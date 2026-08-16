"""Opt-in benchmark/profiling harness for the frozen Qwen3-TTS generation path.

This module measures the *frozen* sequential generation against two plausible
speedups **without changing production defaults or the frozen runner**. It is
a benchmark-only tool: nothing here is used by ``cli.py``/``runner.py``.

Candidates compared (never adopted, never promoted):

- ``baseline``     : the frozen single ``model.generate`` path, per sentence.
- ``speaker-cache``: monkeypatches ``model.extract_speaker_embedding`` to return
  the exact already-evaluated embedding for the canonical reference, so the
  per-sentence speaker pre-processing is computed once instead of per call.
  **Benchmark only** — the monkeypatch changes the call graph.
- ``batch2``/``batch4``: ``model.batch_generate`` with a shared reference at
  batch sizes 2 and 4. **Output-changing and never silently adopted**: the
  batch path forces ``repetition_penalty = max(rep, 1.5)`` for reference-clone
  batches and caps ``per_seq_max_tokens = min(max_tokens, max(75, len(tokens)*6))``,
  both different from the frozen single path. Requires ``--accept-output-changing``.

Commands::

    uv run python -m audiobook.perf profile   [--limit N]
    uv run python -m audiobook.perf benchmark [--include-batch]
                                              [--accept-output-changing]
    uv run python -m audiobook.perf selfcheck

All artifacts are written under ``<root>/outputs/perf`` (gitignored). The
production runner, its state, the configured book, and its output are never
touched.

Qwen3-TTS has no supported seed, so exact PCM equality across reruns is
impossible; objective structure gates + persistent ASR gates here are a
necessary but not sufficient precondition, and blind human A/B samples are
mandatory before any adoption.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import time

from . import runner
from .config import ConfigError, load_config
from .runner import RunError, SAMPLE_RATE

# --- frozen model / reference facts (single source of truth = config) --------
# canonical reference and model are read from audiobook.toml at runtime; the
# defaults here are only used when --limit is not given, and must match the
# frozen pilot contract.

SENTENCES = {
    "s1": "The death of Tamerlane in fourteen oh five was a turning point in world history.",
    "s2": "Tamerlane was the last of the series of world conquerors, in the tradition of Attila and Genghis Khan, who strove to bring the whole of Eurasia, the world island, under the rule of a single vast empire.",
    "s3": "Within fifty years of his death, the maritime states of the Eurasian Far West, with Portugal in the van, were exploring the sea routes that became the nerves and arteries of great maritime empires.",
    "s4": "This is the story of what happened next.",
}
DURATION_BOUNDS = {"s1": (3.5, 8.0), "s2": (7.0, 18.0), "s3": (7.0, 18.0), "s4": (1.5, 6.0)}
SENTENCE_ORDER = ("s1", "s2", "s3", "s4")
# --- candidates ---------------------------------------------------------------
BASELINE = "baseline"
SPEAKER_CACHE = "speaker-cache"
BATCH = ("batch2", "batch4")
DEFAULT_BENCH_CANDIDATES = (BASELINE, SPEAKER_CACHE)
OUT_DIR_REL = "outputs/perf"
REPEATS = 3
# batch output-changing semantics, for the manifest / gate
BATCH_SEMANTIC_NOTES = {
    "repetition_penalty": "batch_generate sets repetition_penalty = max(rep, 1.5) when a shared reference is used (frozen single path uses 1.05)",
    "per_seq_max_tokens": "batch_generate caps per_seq_max_tokens = min(max_tokens, max(75, len(tokenizer.encode(text))*6)) per sequence (frozen single path has no such cap)",
    "output_changing": True,
    "never_silently_adopt": "requires --accept-output-changing; human A/B mandatory before any adoption",
}


# --- tiny helpers (stdlib) ----------------------------------------------------
def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _pkg_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def percentile(values, q):
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float), q))


def stats(values):
    if not values:
        return None
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(np.std(arr)),
    }


def delta_pct(base, cand):
    """% change of candidate vs baseline; positive = candidate faster."""
    return round((base - cand) / base * 100.0, 3) if base else None


def build_workload(root, cfg, limit) -> list:
    """Ordered [(item_id, text, (lo, hi))].

    limit=None -> canonical four-sentence paragraph; otherwise the first
    ``limit`` sentences from the configured book plan (pure stdlib).
    """
    if limit is None:
        return [(k, SENTENCES[k], DURATION_BOUNDS[k]) for k in SENTENCE_ORDER]
    if limit < 0:
        raise RunError(f"limit must be >= 0, got {limit}")
    plan = runner.build_plan(root, cfg, limit=limit)
    return [
        (f"lim{i}", c["text"], (1.5, 60.0))
        for i, c in enumerate(plan["chunks"], 1)
    ]


def paired_orders(candidates, repeats) -> dict:
    """Rotated per-repeat order so order bias is spread across candidates."""
    n = len(candidates)
    out = {}
    for r in range(1, repeats + 1):
        rot = (r - 1) % n
        out[r] = list(candidates[rot:]) + list(candidates[:rot])
    return out


def _require_accept(include_batch, accept) -> None:
    if include_batch and not accept:
        raise SystemExit(
            "batch candidates (batch2/batch4) change generation semantics "
            "(repetition_penalty >= 1.5 and a per-seq max-token cap) and are "
            "output-changing; refusing to run without --accept-output-changing"
        )


# --- worker subprocess -------------------------------------------------------
def worker(candidate, repeat, *, root, out_dir, limit, offline) -> int:
    cfg = load_config(root)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    import soundfile as sf

    from . import asr as _asr

    out_dir = pathlib.Path(out_dir)
    worker_dir = out_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    wav_root = out_dir / "audio" / f"repeat{repeat}_{candidate}"
    wav_root.mkdir(parents=True, exist_ok=True)
    run_started = time.time()

    errors = []
    try:
        runner.verify_inputs(root, cfg.inputs)
    except RunError as e:
        errors.append(f"verify_inputs: {e}")
        _write_worker(worker_dir, candidate, repeat, run_started, cfg, errors, None)
        return 1

    ref_wav = cfg.abs(cfg.audio)
    ref_text = cfg.abs(cfg.transcript).read_text()  # verbatim, trailing newline
    workload = build_workload(root, cfg, limit)
    cache_state = runner.model_cache_state(cfg.model_repo, cfg.model_revision)
    if cache_state["missing"]:
        errors.append(f"model cache incomplete: {cache_state['missing']}")
        _write_worker(worker_dir, candidate, repeat, run_started, cfg, errors, None)
        return 1

    # --- fresh model load (one per worker subprocess) -------------------------
    load_t0 = time.perf_counter()
    model, load_seconds = runner.load_model_once(cfg, offline=offline)
    load_seconds = max(load_seconds, time.perf_counter() - load_t0)

    setup = {"candidate": candidate}
    if candidate == BASELINE:
        ref_audio_arg = str(ref_wav)
        # timing wrapper around the real speaker pre-processing (returns the
        # identical embedding; only measures, never rewrites the result).
        _orig = model.extract_speaker_embedding
        _acc = {"calls": 0, "seconds": 0.0}

        def _wrap(audio, sr=SAMPLE_RATE):
            t0 = time.perf_counter()
            r = _orig(audio, sr=sr)
            _acc["seconds"] += time.perf_counter() - t0
            _acc["calls"] += 1
            return r

        model.extract_speaker_embedding = _wrap
        setup["ref_audio_argument"] = "str(ref_wav) (model.generate reloads WAV and re-extracts speaker embedding per sentence)"
        setup["speaker_cache"] = False
    elif candidate == SPEAKER_CACHE:
        ref_audio_arg = str(ref_wav)
        model.extract_speaker_embedding, _orig = _cache_speaker_embedding(
            model, ref_wav, cfg, setup, errors
        )
    elif candidate in BATCH:
        ref_audio_arg = str(ref_wav)  # shared reference across the batch
        setup["batch"] = {"size": int(candidate[-1]), "shared_reference": True}
        setup["semantics"] = BATCH_SEMANTIC_NOTES
    else:
        errors.append(f"unknown candidate: {candidate}")
        _write_worker(worker_dir, candidate, repeat, run_started, cfg, errors, setup)
        return 1

    # --- generation -----------------------------------------------------------
    per_item = {}
    gen_total = 0.0
    batch_groups = []
    try:
        if candidate in BATCH:
            gen_total, per_item, batch_groups = _gen_batch(
                model, workload, ref_audio_arg, ref_text, cfg,
                wav_root, run_started, int(candidate[-1]), errors,
            )
        else:
            gen_total, per_item = _gen_single(
                model, workload, ref_audio_arg, ref_text, cfg,
                wav_root, run_started, (_acc if candidate == BASELINE else None), errors,
            )
    except Exception as e:  # e.g. batch path unsupported on this model
        errors.append(f"generation failed: {type(e).__name__}: {e}")

    # --- persistent ASR gate (objective, per sentence) ------------------------
    asr_summary = {"status": "not_run"}
    if not errors and per_item:
        try:
            v = _asr.AsrValidator(
                model_repo=cfg.asr_repo, revision=cfg.asr_revision,
                cache_path=out_dir / "asr-cache.json",
            )
            for item_id in per_item:
                rec = v.validate_chunk(wav_root / f"{item_id}.wav", per_item[item_id]["text"], chunk_id=item_id)
                per_item[item_id]["asr"] = rec
                if rec["verdict"] != "PASS":
                    errors.append(f"ASR {item_id} {rec['verdict']}: " + "; ".join(rec.get("reasons") or []))
            passes = sum(1 for it in per_item.values() if it.get("asr", {}).get("verdict") == "PASS")
            asr_summary = {"status": "ran", "passed": passes, "total": len(per_item)}
        except Exception as e:
            asr_summary = {"status": "error", "detail": f"{type(e).__name__}: {e}"}
            errors.append(f"ASR gate failed: {asr_summary['detail']}")

    wm = {
        "repeat": repeat,
        "candidate": candidate,
        "started_unix": run_started,
        "workload": {"limit": limit, "items": [i[0] for i in workload]},
        "model": {
            "repo": cfg.model_repo, "revision": cfg.model_revision,
            "snapshot_dir": cache_state["snapshot_dir"],
            "offline_env": {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE")},
        },
        "reference": {
            "wav": str(ref_wav), "wav_sha256": sha256_file(ref_wav),
            "transcript": str(cfg.abs(cfg.transcript)), "transcript_sha256": sha256_text(ref_text),
        },
        "setup": setup,
        "load_seconds": round(load_seconds, 4),
        "generation_seconds": round(gen_total, 4),
        "generation_seconds_cold_total": round(
            gen_total + float(setup.get("precompute_seconds_live") or 0.0), 4),
        "generation_seconds_amortized": round(gen_total, 4),
        "per_item": per_item,
        "asr": asr_summary,
        "peak_rss_bytes": int(_peak_rss()),
        "environment": {
            "python": sys.version.split()[0],
            "packages": {k: _pkg_version(k) for k in
                         ("mlx", "mlx-audio", "mlx-whisper", "numpy", "soundfile",
                          "transformers", "tokenizers", "huggingface-hub")},
        },
        "runtime_seconds": round(time.time() - run_started, 4),
        "verdict": "FAIL" if errors else "PASS",
        "errors": errors,
    }
    _write_worker(worker_dir, candidate, repeat, run_started, cfg, errors, wm)
    return 0 if not errors else 1


def _write_worker(worker_dir, candidate, repeat, run_started, cfg, errors, wm):
    if wm is None:
        wm = {"repeat": repeat, "candidate": candidate, "verdict": "FAIL", "errors": errors, "started_unix": run_started}
    path = worker_dir / f"repeat{repeat}_{candidate}.json"
    path.write_text(json.dumps(wm, indent=1) + "\n")
    print(f"[worker] repeat {repeat} {candidate}: verdict={wm['verdict']} errors={len(errors)}")


def _reference_fingerprint_matches(arr, canon):
    """Cheap pre-check that ``arr`` could be the canonical reference: exact
    shape, dtype, and element-sum must all match. Works for mlx or numpy
    arrays (operates only on the shared .shape/.dtype/.sum() surface)."""
    if getattr(arr, "shape", None) != getattr(canon, "shape", None):
        return False
    if getattr(arr, "dtype", None) != getattr(canon, "dtype", None):
        return False
    try:
        if float(arr.sum()) != float(canon.sum()):
            return False
    except Exception:
        return False
    return True


def _cache_speaker_embedding(model, ref_wav, cfg, setup, errors):
    """Compute+eval the canonical reference embedding once, verify it is
    deterministic, then patch extract_speaker_embedding to return that exact
    embedding — but ONLY when the input matches the canonical reference
    (fingerprint + exact element equality); any other input delegates to the
    original extractor. Benchmark-only call-graph change."""
    from mlx_audio.utils import load_audio

    ref_array = load_audio(str(ref_wav), SAMPLE_RATE)
    if int(ref_array.ndim) != 1:
        errors.append(f"ref array ndim {ref_array.ndim} != 1")
        return model.extract_speaker_embedding, None
    t0 = time.perf_counter()
    e1 = model.extract_speaker_embedding(ref_array)
    mx_eval(e1)
    t1 = time.perf_counter()
    e2 = model.extract_speaker_embedding(ref_array)
    mx_eval(e2)
    t2 = time.perf_counter()

    import numpy as np

    a1, a2 = np.asarray(e1), np.asarray(e2)
    equal = bool(a1.shape == a2.shape and np.array_equal(a1, a2))
    if not equal:
        errors.append("speaker embedding not deterministic across two calls (cannot cache safely)")
    cached = e1
    calls = {"generation_calls_served_from_cache": 0,
             "delegated_to_original": 0,
             "fingerprint_matched": 0,
             "exact_matched": 0}

    def _patched(audio, sr=SAMPLE_RATE):
        # Never serve a cache hit for an unknown input: gate on a cheap
        # shape/dtype/sum fingerprint, then exact element equality against the
        # canonical preloaded reference array. Nonmatch delegates to the real
        # extract_speaker_embedding (correctness preserved, cache only wins
        # when the input really is the canonical reference).
        import mlx.core as mx

        if _reference_fingerprint_matches(audio, ref_array):
            calls["fingerprint_matched"] += 1
            if bool(mx.array_equal(audio, ref_array)):
                calls["exact_matched"] += 1
                calls["generation_calls_served_from_cache"] += 1
                return cached
        calls["delegated_to_original"] += 1
        return orig(audio, sr=sr)

    orig = model.extract_speaker_embedding
    model.extract_speaker_embedding = _patched
    setup.update({
        "speaker_cache": True,
        "ref_audio_argument": "str(ref_wav) (model.generate reloads WAV, but extract_speaker_embedding returns the cached canonical embedding)",
        "precompute_seconds": round(t2 - t0, 4),
        "precompute_seconds_live": round(t1 - t0, 4),
        "precompute_seconds_duplicate": round(t2 - t1, 4),
        "embedding_shape": list(a1.shape),
        "elementwise_equal_to_duplicate_call": equal,
        "cache_guard": calls,
    })
    return _patched, orig


def mx_eval(arr):
    import mlx.core as mx

    mx.eval(arr)


def _gen_single(model, workload, ref_audio_arg, ref_text, cfg, wav_root, run_started, acc, errors):
    """Frozen per-sentence path (baseline / speaker-cache). Returns
    (total_seconds, per_item)."""
    import numpy as np
    import soundfile as sf

    total = 0.0
    per_item = {}
    for item_id, text, (lo, hi) in workload:
        entry = {"text": text}
        t0 = time.perf_counter()
        try:
            results = list(model.generate(
                text=text,
                ref_audio=ref_audio_arg,
                ref_text=ref_text,
                lang_code=cfg.language,
                stream=False,
                max_tokens=cfg.max_tokens,
            ))
            for r in results:
                mx_eval(r.audio)  # realize lazy MLX graph inside the stopwatch
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            per_item[item_id] = entry
            errors.append(f"{item_id}: {entry['error']}")
            continue
        call_seconds = time.perf_counter() - t0
        total += call_seconds
        entry["generation_seconds"] = round(call_seconds, 4)
        if acc is not None:
            entry["ref_processing_seconds"] = round(acc["seconds"], 4)
            entry["ref_processing_calls"] = int(acc["calls"])
            acc["seconds"] = 0.0  # this call's portion
        _record_result(entry, results, item_id, text, lo, hi, errors)
        w = _write_wav(entry, results, wav_root / f"{item_id}.wav", run_started, errors)
        if w is not None:
            entry["output"] = w
        per_item[item_id] = entry
    return total, per_item


def _gen_batch(model, workload, ref_audio_arg, ref_text, cfg, wav_root, run_started, batch_size, errors):
    """batch_generate shared-reference path at the given batch size."""
    import numpy as np
    import soundfile as sf

    total = 0.0
    per_item = {}
    groups = []
    for start in range(0, len(workload), batch_size):
        group = workload[start:start + batch_size]
        keys = [g[0] for g in group]
        texts = [g[1] for g in group]
        t0 = time.perf_counter()
        results = list(model.batch_generate(
            texts=texts,
            ref_audio=ref_audio_arg,
            ref_text=ref_text,
            lang_code=cfg.language,
            max_tokens=cfg.max_tokens,
            temperature=0.9, top_k=50, top_p=1.0,
            repetition_penalty=1.05,  # frozen single-path value; batch forces max(1.5, this) when cloning
            stream=False,
        ))
        for r in results:
            mx_eval(r.audio)  # realize lazy MLX graph inside the stopwatch
        group_seconds = time.perf_counter() - t0
        total += group_seconds
        by_seq = {}
        for r in results:
            by_seq[int(r.sequence_idx)] = r
        for idx, (item_id, text, (lo, hi)) in enumerate(group):
            entry = {"text": text, "group_seconds": round(group_seconds, 4)}
            entry["generation_seconds"] = round(group_seconds, 4)
            r = by_seq.get(idx)
            if r is None:
                entry["error"] = f"no batch result for sequence_idx {idx}"
                errors.append(f"{item_id}: {entry['error']}")
                per_item[item_id] = entry
                continue
            _record_batch_result(entry, r, item_id, text, lo, hi, errors)
            w = _write_wav(entry, [r], wav_root / f"{item_id}.wav", run_started, errors)
            if w is not None:
                entry["output"] = w
            per_item[item_id] = entry
        groups.append({"keys": keys, "seconds": round(group_seconds, 4), "size": len(keys)})
    return total, per_item, groups


def _record_result(entry, results, item_id, text, lo, hi, errors):
    if len(results) != 1:
        entry["error"] = f"generation results = {len(results)}, expected exactly 1"
        errors.append(f"{item_id}: {entry['error']}")
        return
    _record_single_result(entry, results[0], item_id, lo, hi, errors)


def _record_batch_result(entry, r, item_id, text, lo, hi, errors):
    _record_single_result(entry, r, item_id, lo, hi, errors)


def _record_single_result(entry, r, item_id, lo, hi, errors):
    # Derive fields from the installed dataclass rather than hardcoding the
    # GenerationResult layout: BatchGenerationResult drops real_time_factor and
    # segment_idx (it carries sequence_idx instead).
    names = {f.name for f in dataclasses.fields(r)}

    def g(name, default=None):
        return getattr(r, name, default)

    sample_rate = int(g("sample_rate"))
    samples = int(g("samples"))
    segment_idx = int(g("segment_idx")) if "segment_idx" in names and g("segment_idx") is not None else None
    sequence_idx = int(g("sequence_idx")) if "sequence_idx" in names and g("sequence_idx") is not None else None
    rtf = None
    if "real_time_factor" in names and g("real_time_factor") is not None:
        rtf = float(g("real_time_factor"))
    entry["result"] = {
        "segment_idx": segment_idx,
        "sequence_idx": sequence_idx,
        "sample_rate": sample_rate,
        "samples": samples,
        "audio_duration_seconds": float(samples / sample_rate) if sample_rate else None,
        "token_count": int(g("token_count")),
        "real_time_factor": rtf,
        "processing_time_seconds": float(g("processing_time_seconds")),
        "peak_memory_usage_gb": float(g("peak_memory_usage")),
    }
    if sample_rate != SAMPLE_RATE:
        errors.append(f"{item_id} result sample_rate {sample_rate} != {SAMPLE_RATE}")
        entry.setdefault("errors", []).append(f"sample_rate {sample_rate} != {SAMPLE_RATE}")
    dur = entry["result"]["audio_duration_seconds"]
    if dur is not None:
        entry["duration_seconds"] = round(dur, 4)
        within = lo <= dur <= hi
        entry["duration_gate"] = "PASS" if within else "FAIL"
        if not within:
            errors.append(f"{item_id} duration {dur:.3f}s outside bounds [{lo}, {hi}]")
            entry.setdefault("errors", []).append(f"duration {dur:.3f}s outside bounds [{lo}, {hi}]")


def _write_wav(entry, results, out_path, run_started, errors):
    """Write results[0].audio to out_path as PCM16 24k mono; returns facts."""
    import numpy as np
    import soundfile as sf

    if not results or not getattr(results[0], "audio", None) is not None:
        return {"exists": False}
    audio_np = np.asarray(results[0].audio)
    if audio_np.size == 0:
        errors.append(f"{out_path.name} result audio empty")
        return {"exists": False, "error": "empty audio"}
    if not bool(np.isfinite(audio_np).all()):
        errors.append(f"{out_path.name} result audio non-finite")
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if not peak < 1.0:
        errors.append(f"{out_path.name} raw float peak {peak} >= 1.0 (clipping)")
    t0 = time.perf_counter()
    sf.write(str(out_path), audio_np, int(results[0].sample_rate), subtype="PCM_16")
    entry["write_seconds"] = round(time.perf_counter() - t0, 4)
    if not out_path.exists():
        errors.append(f"{out_path.name} output missing")
        return {"exists": False}
    from . import runner as _runner

    w = _runner.wav_facts(out_path, run_started, errors)
    return w


def _peak_rss() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _worker_argv(root, out_dir, candidate, repeat, limit, offline):
    """argv for the benchmark->worker subprocess.

    Global options (--root/--out/--offline) live on the top-level parser and
    therefore MUST precede the subcommand token; worker-local options follow it.
    """
    return (
        [sys.executable, "-m", "audiobook.perf",
         "--root", str(root), "--out", str(out_dir)]
        + (["--offline"] if offline else [])
        + ["worker",
           "--candidate", candidate, "--repeat", str(repeat)]
        + (["--limit", str(limit)] if limit is not None else [])
    )


# --- driver (benchmark) ------------------------------------------------------
def benchmark(root, out_dir, *, repeats, include_batch, accept, limit, offline) -> int:
    _require_accept(include_batch, accept)
    if include_batch:
        candidates = list(DEFAULT_BENCH_CANDIDATES) + list(BATCH)
    else:
        candidates = list(DEFAULT_BENCH_CANDIDATES)
    orders = paired_orders(candidates, repeats)

    out_dir = pathlib.Path(out_dir)
    worker_dir = out_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    worker_manifests = {}
    failures = []
    for repeat in range(1, repeats + 1):
        for candidate in orders[repeat]:
            print(f"[benchmark] repeat {repeat} candidate {candidate} ...", flush=True)
            argv = _worker_argv(root, out_dir, candidate, repeat, limit, offline)
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
            wjson = worker_dir / f"repeat{repeat}_{candidate}.json"
            if proc.returncode != 0 or not wjson.exists():
                failures.append({"repeat": repeat, "candidate": candidate, "returncode": proc.returncode})
                print(f"[benchmark] FAILED repeat {repeat} {candidate} rc={proc.returncode}")
                print((proc.stderr or proc.stdout)[-3000:])
                continue
            wm = json.loads(wjson.read_text())
            worker_manifests[f"repeat{repeat}_{candidate}"] = wm
            print(f"[benchmark] repeat {repeat} {candidate}: gen={wm['generation_seconds']:.3f}s verdict={wm['verdict']}")

    runtime_seconds = time.time() - started
    errors = []
    if failures:
        errors.append(f"worker failures: {failures}")
    if len(worker_manifests) != repeats * len(candidates):
        errors.append(f"only {len(worker_manifests)}/{repeats * len(candidates)} worker results")
    worker_errors = [
        f"r{w['repeat']}_{w['candidate']}: {e}"
        for w in worker_manifests.values() for e in w.get("errors", [])
    ]
    errors.extend(worker_errors)

    metrics = None
    if not errors:
        metrics = _aggregate(worker_manifests, candidates, repeats)

    verdict, verdict_reason = _verdict(metrics, errors, candidates)
    manifest = {
        "benchmark": "perf-speedup",
        "goal": ("Compare frozen sequential Qwen3-TTS generation vs two plausible speedups: "
                 "(A) speaker-embedding cache, (B) Qwen batch_generate shared-reference at batch sizes 2/4. "
                 "Benchmark-only; neither candidate is enabled in production or promoted."),
        "design": {
            "repeats": repeats,
            "candidates": candidates,
            "orders_per_repeat": orders,
            "alternating_note": "order rotates per repeat to reduce warm/cold bias",
            "fresh_process_per_workload": True,
            "fresh_model_per_workload": True,
            "model_reuse": "none (fresh subprocess + fresh model per workload)",
            "batch_gating": "batch2/batch4 only with --accept-output-changing; output-changing semantics recorded per worker",
            "default_benchmark": "baseline vs speaker-cache only",
        },
        "model": {"repo": load_config(root).model_repo, "revision": load_config(root).model_revision, "sample_rate": SAMPLE_RATE},
        "reference": {"wav": str(load_config(root).abs(load_config(root).audio))},
        "workers": worker_manifests,
        "metrics": metrics,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "verdict_criterion": ">=10% median generation_seconds/workload delta vs baseline => material; no equivalence claim (no seed); blind human A/B mandatory before adoption",
        "runtime_seconds": runtime_seconds,
        "started_unix": started,
        "errors": errors,
    }
    (out_dir / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[benchmark] verdict: {verdict} ({verdict_reason})")
    if metrics:
        for cand in candidates:
            if cand == BASELINE:
                continue
            m = metrics[cand]
            cd = m.get("delta_pct_vs_baseline", {}).get("cold", {}).get("median")
            ad = m.get("delta_pct_vs_baseline", {}).get("amortized", {}).get("median")
            print(f"[benchmark] {cand} vs baseline delta% cold_median={cd} amortized_median={ad}")
    print(f"[benchmark] manifest: {out_dir / 'benchmark-manifest.json'}")
    return 0 if not errors else 1


def _workload_seconds(w, mode):
    """Per-worker workload timing.

    ``cold`` counts the one-time live speaker precompute for cached candidates
    (never hide it); the explicit duplicate determinism check is benchmark
    overhead and is excluded. ``amortized`` is pure generation only.
    """
    gen = float(w["generation_seconds"])
    if mode == "amortized":
        return gen
    pre = w.get("setup", {}).get("precompute_seconds_live")
    return gen + (float(pre) if pre else 0.0)


def _aggregate(worker_manifests, candidates, repeats):
    hashes = {}
    for w in worker_manifests.values():
        for item_id, it in w.get("per_item", {}).items():
            if "output" in it and isinstance(it["output"], dict) and it["output"].get("sha256"):
                hashes[f"r{w['repeat']}_{w['candidate']}_{item_id}"] = it["output"]["sha256"]

    # candidate -> repeat -> seconds (cold & amortized)
    cold = {c: {} for c in candidates}
    amort = {c: {} for c in candidates}
    for w in worker_manifests.values():
        r = int(w["repeat"])
        cold.setdefault(w["candidate"], {})[r] = _workload_seconds(w, "cold")
        amort.setdefault(w["candidate"], {})[r] = _workload_seconds(w, "amortized")

    def median_of(d, c):
        return stats([d[c][r] for r in sorted(d[c])]) if d.get(c) else None

    metrics = {}
    for cand in candidates:
        entry = {
            "generation_seconds_per_workload": {
                "cold": stats([cold[cand][r] for r in sorted(cold[cand])]) if cold.get(cand) else None,
                "amortized": stats([amort[cand][r] for r in sorted(amort[cand])]) if amort.get(cand) else None,
            },
        }
        if cand != BASELINE:
            cold_deltas = [delta_pct(cold[BASELINE][r], cold[cand][r]) for r in sorted(set(cold[BASELINE]) & set(cold[cand]))]
            amort_deltas = [delta_pct(amort[BASELINE][r], amort[cand][r]) for r in sorted(set(amort[BASELINE]) & set(amort[cand]))]
            entry["delta_pct_vs_baseline"] = {
                "cold": stats(cold_deltas) if cold_deltas else None,
                "amortized": stats(amort_deltas) if amort_deltas else None,
                "method": "same-repeat paired deltas",
            }
        metrics[cand] = entry

    # preserve paired-repeat deltas: baseline vs candidate for the SAME repeat
    pairs = {}
    for r in range(1, repeats + 1):
        if BASELINE not in cold or r not in cold[BASELINE]:
            break
        for c in candidates:
            if c == BASELINE or r not in cold.get(c, {}):
                continue
            pairs.setdefault(r, {})[c] = {
                "baseline_cold_seconds": round(cold[BASELINE][r], 4),
                "candidate_cold_seconds": round(cold[c][r], 4),
                "cold_delta_pct": delta_pct(cold[BASELINE][r], cold[c][r]),
                "baseline_amortized_seconds": round(amort[BASELINE][r], 4),
                "candidate_amortized_seconds": round(amort[c][r], 4),
                "amortized_delta_pct": delta_pct(amort[BASELINE][r], amort[c][r]),
            }
    metrics["pairs"] = pairs
    metrics["_hashes_provenance_only_no_equivalence"] = hashes
    return metrics


def _verdict(metrics, errors, candidates):
    if errors or metrics is None:
        return "FAIL", "worker or aggregation errors; no metrics"
    deltas = []
    for cand in candidates:
        if cand == BASELINE:
            continue
        md = metrics[cand].get("delta_pct_vs_baseline", {}).get("cold", {}).get("median")
        if md is None:
            continue
        deltas.append(md)
    if not deltas:
        return "INCONCLUSIVE", "no measurable baseline median"
    best = max(deltas)
    if best >= 10.0:
        return "MATERIAL_IMPROVEMENT", f"best median candidate delta {best}% >= 10% (benchmark only; human A/B required before adoption)"
    if best <= -10.0:
        return "MATERIAL_REGRESSION", f"best median candidate delta {best}% <= -10%"
    return "NOT_MATERIAL", f"best median candidate delta {best}% within +/-10%"


# --- profile -----------------------------------------------------------------
def profile(root, out_dir, *, limit, offline) -> int:
    cfg = load_config(root)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_root = out_dir / "audio" / "profile"
    wav_root.mkdir(parents=True, exist_ok=True)
    run_started = time.time()

    errors = []
    ref_wav = cfg.abs(cfg.audio)
    ref_text = cfg.abs(cfg.transcript).read_text()
    workload = build_workload(root, cfg, limit)

    model, load_seconds = runner.load_model_once(cfg, offline=offline)
    _orig = model.extract_speaker_embedding
    _acc = {"calls": 0, "seconds": 0.0}

    def _wrap(audio, sr=SAMPLE_RATE):
        t0 = time.perf_counter()
        r = _orig(audio, sr=sr)
        _acc["seconds"] += time.perf_counter() - t0
        _acc["calls"] += 1
        return r

    model.extract_speaker_embedding = _wrap

    per_item = {}
    total = 0.0
    for item_id, text, (lo, hi) in workload:
        t0 = time.perf_counter()
        results = list(model.generate(
            text=text, ref_audio=str(ref_wav), ref_text=ref_text,
            lang_code=cfg.language, stream=False, max_tokens=cfg.max_tokens,
        ))
        for r in results:
            mx_eval(r.audio)  # realize lazy MLX graph inside the stopwatch
        call_seconds = time.perf_counter() - t0
        total += call_seconds
        entry = {
            "text": text,
            "generation_seconds": round(call_seconds, 4),
            "ref_processing_seconds": round(_acc["seconds"], 4),
            "ref_processing_calls": int(_acc["calls"]),
            # AR and decoder phases are inside the opaque model.generate body;
            # splitting them would require patching the installed source, so they
            # are reported unavailable rather than fabricated.
            "ar_generation_seconds": None,
            "decoder_seconds": None,
        }
        _acc["seconds"] = 0.0
        if len(results) != 1:
            errors.append(f"{item_id} generation results = {len(results)}")
            entry["error"] = f"generation results = {len(results)}"
            per_item[item_id] = entry
            continue
        _record_single_result(entry, results[0], item_id, lo, hi, errors)
        w = _write_wav(entry, results, wav_root / f"{item_id}.wav", run_started, errors)
        if w is not None:
            entry["output"] = w
        per_item[item_id] = entry
    model.extract_speaker_embedding = _orig

    asr_summary = {"status": "not_run"}
    if not errors:
        from . import asr as _asr

        try:
            v = _asr.AsrValidator(model_repo=cfg.asr_repo, revision=cfg.asr_revision,
                                  cache_path=out_dir / "asr-cache.json")
            for item_id in per_item:
                rec = v.validate_chunk(wav_root / f"{item_id}.wav", per_item[item_id]["text"], chunk_id=item_id)
                per_item[item_id]["asr"] = rec
                if rec["verdict"] != "PASS":
                    reasons = "; ".join(rec.get("reasons") or [])
                    errors.append(f"ASR {item_id} {rec['verdict']}: {reasons}")
            asr_summary = {"status": "ran",
                           "passed": sum(1 for it in per_item.values() if it.get("asr", {}).get("verdict") == "PASS"),
                           "total": len(per_item)}
        except Exception as e:
            asr_summary = {"status": "error", "detail": f"{type(e).__name__}: {e}"}
            errors.append(f"ASR gate failed: {asr_summary['detail']}")

    manifest = {
        "benchmark": "perf-profile",
        "mode": "baseline (frozen sequential path), single run",
        "phases_measured": ["model_load", "ref_processing_seconds (speaker encode via wrapper)", "generation_seconds (whole generate call)", "write_seconds"],
        "phases_unavailable": {
            "ar_generation": "inside opaque model.generate; would require patching installed source — reported unavailable, not fabricated",
            "decoder": "inside opaque model.generate; would require patching installed source — reported unavailable, not fabricated",
        },
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision, "sample_rate": SAMPLE_RATE},
        "reference": {"wav": str(ref_wav), "wav_sha256": sha256_file(ref_wav),
                      "transcript_sha256": sha256_text(ref_text)},
        "load_seconds": round(load_seconds, 4),
        "generation_seconds": round(total, 4),
        "per_item": per_item,
        "asr": asr_summary,
        "peak_rss_bytes": int(_peak_rss()),
        "environment": {"python": sys.version.split()[0],
                        "packages": {k: _pkg_version(k) for k in
                                     ("mlx", "mlx-audio", "mlx-whisper", "numpy", "soundfile",
                                      "transformers", "tokenizers", "huggingface-hub")}},
        "runtime_seconds": round(time.time() - run_started, 4),
        "verdict": "FAIL" if errors else "PASS",
        "errors": errors,
    }
    (out_dir / "profile-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[profile] verdict: {manifest['verdict']} generation={total:.3f}s load={load_seconds:.3f}s")
    print(f"[profile] manifest: {out_dir / 'profile-manifest.json'}")
    return 0 if not errors else 1


# --- selfcheck (no model, no book) ------------------------------------------
def selfcheck() -> int:
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")

    # canonical workload well-formed
    check("SENTENCES s1..s4 nonempty", all(bool(SENTENCES[k]) for k in SENTENCE_ORDER))
    check("DURATION_BOUNDS cover s1..s4 with lo<hi",
          all(DURATION_BOUNDS[k][0] < DURATION_BOUNDS[k][1] for k in SENTENCE_ORDER))

    # build_workload with a tiny in-memory plan (no config/book) via a stub
    def _fake_plan(root, config=None, chapters=None, limit=None, resume_from=None):
        return {"chunks": [{"id": f"c{i}", "chapter": "ch01", "idx": i,
                            "text": f"sentence {i}", "text_sha256": sha256_text(f"sentence {i}")}
                           for i in range(limit or 0)]}

    orig = runner.build_plan
    runner.build_plan = _fake_plan
    try:
        wl = build_workload(".", None, 2)
        runner.build_plan = orig
        check("build_workload(limit=2) -> 2 items", len(wl) == 2 and wl[1][0] == "lim2")
    finally:
        runner.build_plan = orig
    canonical = build_workload(".", None, None)
    check("build_workload canonical -> 4 items", len(canonical) == 4)

    # paired order rotation spreads candidates
    orders = paired_orders([BASELINE, SPEAKER_CACHE], 3)
    check("paired_orders alt 2-cand",
          orders[1] == [BASELINE, SPEAKER_CACHE] and orders[2] == [SPEAKER_CACHE, BASELINE] and orders[3] == [BASELINE, SPEAKER_CACHE])
    check("paired_orders covers all candidates",
          set(sum(orders.values(), [])) == {BASELINE, SPEAKER_CACHE})

    # stats / delta_pct math on known inputs
    s = stats([1.0, 2.0, 3.0])
    check("stats median/mean", s["median"] == 2.0 and abs(s["mean"] - 2.0) < 1e-9)
    check("delta_pct sign convention (cand faster => +)",
          delta_pct(2.0, 1.0) == 50.0 and delta_pct(4.0, 6.0) == -50.0)

    # batch accept gating
    def _require(include, accept):
        try:
            _require_accept(include, accept)
            return "ok"
        except SystemExit:
            return "blocked"

    check("batch blocked without --accept-output-changing", _require(True, False) == "blocked")
    check("batch allowed with --accept-output-changing", _require(True, True) == "ok")
    check("default benchmark excludes batch", DEFAULT_BENCH_CANDIDATES == (BASELINE, SPEAKER_CACHE))

    # worker subprocess argv: global options MUST precede the subcommand token,
    # and the whole argv must dry-parse through the CLI parser.
    wag = _worker_argv("/r", "/o", SPEAKER_CACHE, 2, 5, True)
    widx = wag.index("worker")
    globals_before = all(f in wag[:widx] for f in ("--root", "--out", "--offline"))
    locals_after = all(f not in wag[:widx] for f in ("--candidate", "--repeat", "--limit"))
    pa = build_parser().parse_args(wag[3:])  # skip python, -m, module
    check("worker argv: globals before subcommand, locals after",
          globals_before and locals_after and wag.count("worker") == 1)
    check("worker argv dry-parses via CLI",
          pa.root == "/r" and pa.out == "/o" and pa.offline
          and pa.candidate == SPEAKER_CACHE and pa.repeat == 2 and pa.limit == 5)

    # pair stats: per-repeat paired deltas (cold includes live precompute, never
    # hidden; amortized is pure generation).
    def _mk(repeat, cand, gen, pre=None):
        w = {"repeat": repeat, "candidate": cand, "generation_seconds": gen, "per_item": {}}
        if pre is not None:
            w["setup"] = {"precompute_seconds_live": pre}
        else:
            w["setup"] = {}
        return w

    wms = {
        "r1_baseline": _mk(1, BASELINE, 10.0),
        "r1_speaker-cache": _mk(1, SPEAKER_CACHE, 6.0, pre=2.0),
        "r2_baseline": _mk(2, BASELINE, 12.0),
        "r2_speaker-cache": _mk(2, SPEAKER_CACHE, 8.0, pre=2.0),
    }
    agg = _aggregate(wms, (BASELINE, SPEAKER_CACHE), 2)
    p1 = agg["pairs"][1][SPEAKER_CACHE]
    p2 = agg["pairs"][2][SPEAKER_CACHE]
    check("pair stats: cold counts live precompute (10 vs 8 -> +20%)",
          p1["baseline_cold_seconds"] == 10.0 and p1["candidate_cold_seconds"] == 8.0 and p1["cold_delta_pct"] == 20.0)
    check("pair stats: amortized is pure generation (10 vs 6 -> +40%)",
          p1["baseline_amortized_seconds"] == 10.0 and p1["candidate_amortized_seconds"] == 6.0 and p1["amortized_delta_pct"] == 40.0)
    check("pair stats: per-repeat preserved (repeat2 distinct)",
          p2["cold_delta_pct"] == delta_pct(12.0, 10.0))

    # duration-bounds gate via the real recorder (introspected schema).
    @dataclasses.dataclass
    class _FakeGen:
        audio: object
        samples: int
        sample_rate: int
        segment_idx: int
        token_count: int
        audio_duration: str
        real_time_factor: float
        prompt: dict
        audio_samples: dict
        processing_time_seconds: float
        peak_memory_usage: float
        is_streaming_chunk: bool = False
        is_final_chunk: bool = False

    def _fake_dur(seconds, sr=24000):
        return _FakeGen(audio=None, samples=int(seconds * sr), sample_rate=sr,
                        segment_idx=0, token_count=10, audio_duration="", real_time_factor=1.0,
                        prompt={}, audio_samples={}, processing_time_seconds=1.0, peak_memory_usage=0.5)

    e_in, e_out = {"text": ""}, {"text": ""}
    errs_in, errs_out = [], []
    _record_single_result(e_in, _fake_dur(5.0), "s1", 3.5, 8.0, errs_in)
    _record_single_result(e_out, _fake_dur(9.0), "s1", 3.5, 8.0, errs_out)
    check("duration gate: within bounds -> PASS", e_in["duration_gate"] == "PASS" and not errs_in)
    check("duration gate: outside bounds -> FAIL + error",
          e_out["duration_gate"] == "FAIL" and any("outside bounds" in x for x in errs_out))
    check("result schema introspection (sample_rate/samples/token_count)",
          e_in["result"]["sample_rate"] == 24000 and e_in["result"]["samples"] == 120000 and e_in["result"]["token_count"] == 10)

    # cache guard: fingerprint rejects shape/dtype/sum mismatches.
    import numpy as _np

    _a = _np.array([1.0, 2.0, 3.0])
    check("cache guard fingerprint: exact match", _reference_fingerprint_matches(_np.array([1.0, 2.0, 3.0]), _a))
    check("cache guard fingerprint: reject sum mismatch",
          not _reference_fingerprint_matches(_np.array([1.0, 2.0, 4.0]), _a))
    check("cache guard fingerprint: reject shape mismatch",
          not _reference_fingerprint_matches(_np.array([1.0, 2.0]), _a))
    check("cache guard fingerprint: reject dtype mismatch",
          not _reference_fingerprint_matches(_np.array([1, 2, 3]), _a))

    failed = sum(1 for _, ok in results if not ok)
    print(f"perf selfcheck: {'FAIL' if failed else 'ok'} ({len(results) - failed}/{len(results)})")
    return 1 if failed else 0


# --- CLI ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="audiobook.perf",
        description="Benchmark/profiling harness for the frozen Qwen3-TTS path (opt-in; changes nothing in production).",
    )
    ap.add_argument("--root", help="project root (default: auto-discovered via audiobook.toml)")
    ap.add_argument("--out", help="output dir (default: <root>/outputs/perf)")
    ap.add_argument("--offline", action="store_true", help="forbid any model downloads")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="measure phase timings of the frozen sequential path (single run)")
    p.add_argument("--limit", type=int, default=None, help="use first N sentences from the configured book instead of the canonical 4-sentence paragraph")
    p.set_defaults(run=_run_profile)

    b = sub.add_parser("benchmark", help="paired comparison: baseline vs speaker-cache (default), plus batch with --include-batch")
    b.add_argument("--repeats", type=int, default=REPEATS, help=f"paired runs per candidate (default {REPEATS})")
    b.add_argument("--include-batch", action="store_true", help="also compare batch2/batch4 (requires --accept-output-changing)")
    b.add_argument("--accept-output-changing", action="store_true",
                   help="acknowledge batch candidates change generation semantics (repetition_penalty>=1.5, per-seq max-token cap)")
    b.add_argument("--limit", type=int, default=None, help="use first N sentences from the configured book")
    b.set_defaults(run=_run_benchmark)

    s = sub.add_parser("selfcheck", help="pure stdlib checks; no model, no book")
    s.set_defaults(run=_run_selfcheck)

    w = sub.add_parser("worker", help="internal: run one fresh-process workload (used by benchmark driver)")
    w.add_argument("--candidate", required=True, choices=list(DEFAULT_BENCH_CANDIDATES) + list(BATCH))
    w.add_argument("--repeat", type=int, required=True)
    w.add_argument("--limit", type=int, default=None)
    w.set_defaults(run=_run_worker)
    return ap


def _resolve_out(root, out):
    return pathlib.Path(out) if out else root / OUT_DIR_REL


def _run_profile(args):
    try:
        root = pathlib.Path(args.root) if args.root else runner.find_root()
    except (RunError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return profile(root, _resolve_out(root, args.out), limit=args.limit, offline=args.offline)


def _run_benchmark(args):
    try:
        root = pathlib.Path(args.root) if args.root else runner.find_root()
    except (RunError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return benchmark(root, _resolve_out(root, args.out), repeats=args.repeats,
                     include_batch=args.include_batch, accept=args.accept_output_changing,
                     limit=args.limit, offline=args.offline)


def _run_worker(args):
    try:
        root = pathlib.Path(args.root) if args.root else runner.find_root()
    except (RunError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return worker(args.candidate, args.repeat, root=root, out_dir=_resolve_out(root, args.out),
                  limit=args.limit, offline=args.offline)


def _run_selfcheck(args):
    return selfcheck()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
