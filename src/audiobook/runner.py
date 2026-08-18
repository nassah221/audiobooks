"""Resumable Qwen3-TTS + Bragg-reference generation runner.

Uses one model load per run and one non-streaming ``generate`` call per
paragraph-bounded sentence chunk (``ref_audio``/``ref_text`` verbatim,
configured language, token limit, and deterministic MLX seed). Each 24kHz
mono PCM16 chunk is checkpointed, hashed, validated with
``audiobook.asr.AsrValidator``, and byte-concatenated without inserted samples.
ETA reports generation and ASR costs separately.

Module import is stdlib-only. numpy/soundfile/mlx-audio/huggingface load
lazily on first use with actionable errors, mirroring asr.py. Paths recorded
in artifacts are relative (portable repo); only hashes identify inputs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import struct
import sys
import time

from . import asr, config, epub, qwenfix
from .config import Config, ConfigError, load_config

__all__ = [
    "RunError",
    "Config",
    "ConfigError",
    "load_config",
    "MODEL_REPO",
    "MODEL_REVISION",
    "SAMPLE_RATE",
    "PILOT_SENTENCE",
    "REQUIRED_MODEL_FILES",
    "find_root",
    "verify_inputs",
    "model_cache_state",
    "run_fingerprint",
    "parse_chapters",
    "build_plan",
    "preflight",
    "Generator",
    "validate_generated",
    "estimate",
    "selfcheck",
]

# --- model / generation contract --------------------------------------------
MODEL_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
MODEL_REVISION = "a6eb4f68e4b056f1215157bb696209bc82a6db48"
SAMPLE_RATE = 24_000
PILOT_SENTENCE = (
    "The death of Tamerlane in fourteen oh five was a turning point in world history."
)
# The Qwen API has no seed argument; generation uses MLX's global RNG.
# Each paragraph attempt resets that RNG to a stable, distinct seed.
CLAUSE_SPLIT_POLICY = "planned-strong-clause-v1"
REQUIRED_PUNCTUATION_KINDS = frozenset({
    "sentence_end", "sentence-ending", "sentence-ending-punctuation",
    "period", "question", "exclamation", ".", "?", "!",
    "colon", "semicolon", "colon_semicolon",
    "parenthetical_comma", "parenthetical-comma",
})


def _punctuation_passes(record: dict | None) -> bool:
    """Return whether the ASR punctuation metrics contain no aligned failure."""
    punctuation = (record or {}).get("punctuation")
    if not isinstance(punctuation, dict):
        return False
    boundaries = punctuation.get("boundaries")
    if not isinstance(boundaries, list):
        return False
    for boundary in boundaries:
        if not isinstance(boundary, dict):
            return False
        if boundary.get("aligned") is True and boundary.get("passed") is False:
            return False
    for summary in punctuation.values():
        if not isinstance(summary, dict) or "failed" not in summary:
            continue
        failed = summary["failed"]
        if isinstance(failed, bool) or not isinstance(failed, int) or failed < 0:
            return False
        if failed:
            return False
    return True


def _take_passes(record: dict | None) -> bool:
    if not isinstance(record, dict) or record.get("verdict") != "PASS":
        return False
    terminal = record.get("terminal")
    return (isinstance(terminal, dict) and terminal.get("matched") is True
            and _punctuation_passes(record))

def _planned_clauses(chunk: dict) -> list[dict]:
    """Return the planner's eligible, source-ordered clause spans.

    The planner owns eligibility and the eight-word side constraints. Runner
    only validates the supplied spans, so it never invents lexical splits.
    """
    raw = (chunk.get("eligible_clause_spans") or chunk.get("clause_spans")
           or chunk.get("clauses") or [])
    out = []
    for item in raw:
        if not isinstance(item, dict) or item.get("eligible", True) is False:
            continue
        start = item.get("start", item.get("source_start"))
        end = item.get("end", item.get("source_end"))
        text = item.get("text")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            continue
        if not isinstance(text, str):
            base = chunk.get("text", "")
            text = base[start:end]
        if not text.strip():
            continue
        out.append({**item, "start": start, "end": end, "text": text})
    out.sort(key=lambda item: (item["start"], item["end"]))
    return out
PILOT_DURATION_BOUNDS = (3.5, 8.0)

SENTENCE_PAUSE_SECONDS = 0.25
PARAGRAPH_PAUSE_SECONDS = 0.5

def _invocation(cfg: Config) -> dict:
    """Generation knobs read from config; stream is fixed False."""
    return {
        "lang_code": cfg.language,
        "stream": False,
        "max_tokens": cfg.max_tokens,
        "seed": cfg.seed,
        "speed": "not supplied (defaults to 1.0; MLX path ignores speed)",
    }


REQUIRED_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
]

ASR_CACHE_REL = "validation/asr-cache.json"
RECORDS_REL = "validation/records.json"
BENCHMARK_REL = "benchmark/benchmark.json"
PILOT_WAV_REL = "benchmark/pilot.wav"
STATE_REL = "state.json"
CHUNKS_JSONL_REL = "chunks.jsonl"
BOOK_WAV_REL = "book.wav"
BOOK_JSON_REL = "book.json"

_WAV_HEADER = struct.Struct("<4sI4s4sIHHIIHH4sI")


class RunError(Exception):
    """User-facing error with a clear message (caught by the CLI)."""


# --- hashing -----------------------------------------------------------------
def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _chunk_identity_hash(chunk: dict) -> str:
    """Hash every source fact that determines a planned generation unit."""
    payload = {key: chunk.get(key) for key in (
        "text_sha256", "paragraph_sha256", "source_span", "sentence_span",
        "sentence_indexes", "sentence_count", "sentence_spans",
        "clause_indexes", "clause_count", "clause_spans",
        "eligible_clause_spans", "word_count",
    )}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _state_plan_matches(st_plan: dict | None, plan: dict) -> bool:
    """Return whether persisted state describes this exact source plan."""
    if not isinstance(st_plan, dict):
        return False
    chapters = [{"id": c["id"], "title": c["title"],
                 "paragraphs": c["paragraphs"], "groups": c["groups"]}
                for c in plan["chapters"]]
    return (st_plan.get("planner") == plan.get("planner")
            and st_plan.get("chunk_ids") == [c["id"] for c in plan["chunks"]]
            and st_plan.get("text_hashes") == [_chunk_identity_hash(c) for c in plan["chunks"]]
            and st_plan.get("total_paragraphs") == plan["total_paragraphs"]
            and st_plan.get("total_groups") == plan["total_groups"]
            and st_plan.get("chapters") == chapters)


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_json(path: pathlib.Path, obj) -> None:
    _atomic_write(path, json.dumps(obj, indent=1) + "\n")

def run_fingerprint(cfg: Config, ref_wav_sha: str, ref_text_sha: str) -> str:
    """Hash inputs, generation settings, and the exact chunk policy."""
    planner = {"policy": epub.SENTENCE_GROUP_POLICY,
               "version": epub.SENTENCE_GROUP_VERSION,
               "limits": dict(epub.SENTENCE_GROUP_LIMITS),
               "clause_policy": CLAUSE_SPLIT_POLICY}
    payload = {
        "book": {"path": cfg.book, "sha256": cfg.book_sha256},
        "voice": {"audio": cfg.audio, "wav_sha256": ref_wav_sha,
                  "transcript": cfg.transcript, "text_sha256": ref_text_sha},
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision,
                  "language": cfg.language, "max_tokens": cfg.max_tokens,
                  "seed": cfg.seed},
        "asr": {"repo": cfg.asr_repo, "revision": cfg.asr_revision},
        "planner": planner, "sample_rate": SAMPLE_RATE, "stream": False,
        "generation": {"policy": "icl-eos-hold-v1",
                       "eos_hold_frames": qwenfix.EOS_HOLD_FRAMES,
                       "tail_max_silence_seconds": qwenfix.TAIL_MAX_SILENCE_SECONDS,
                       "tail_fade_seconds": qwenfix.TAIL_FADE_SECONDS,
                       "sibilant_high_frac_min": qwenfix.SIBILANT_HIGH_FRAC_MIN},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --- root / inputs -----------------------------------------------------------
def find_root(start=None) -> pathlib.Path:
    """Nearest ancestor of `start` (default cwd) containing audiobook.toml."""
    try:
        return config.find_root(start)
    except ConfigError as e:
        raise RunError(str(e)) from e


def verify_inputs(root, checksums=None) -> dict:
    """Presence + sha256 of every configured input; raises RunError with detail."""
    root = pathlib.Path(root)
    if checksums is None:
        checksums = load_config(root).inputs
    facts, errors = {}, []
    for rel, want in checksums.items():
        p = root / rel
        if not p.is_file():
            errors.append(f"{rel}: missing (copy the input onto this machine; see README)")
            continue
        got = sha256_file(p)
        facts[rel] = {"path": str(p), "sha256": got, "ok": got == want}
        if got != want:
            errors.append(f"{rel}: sha256 {got} != expected {want}")
    if errors:
        raise RunError("input verification failed:\n  " + "\n  ".join(errors))
    return facts


# --- model cache -------------------------------------------------------------
def model_cache_state(repo=MODEL_REPO, revision=MODEL_REVISION) -> dict:
    """Configured model snapshot presence/refs state in the HF cache."""
    hf_home = pathlib.Path(
        os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface")
    )
    hub = hf_home / "hub" / f"models--{repo.replace('/', '--')}"
    snap = hub / "snapshots" / revision
    refs_main = hub / "refs" / "main"
    missing = [f for f in REQUIRED_MODEL_FILES if not (snap / f).is_file()]
    refs = refs_main.read_text().strip() if refs_main.is_file() else None
    return {
        "repo": repo,
        "revision": revision,
        "snapshot_dir": str(snap),
        "cache_complete": not missing,
        "missing": missing,
        "refs_main": refs,
        "refs_match": refs == revision,
    }




def _pkg_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions() -> dict:
    return {name: _pkg_version(name) for name in
            ("mlx", "mlx-audio", "mlx-whisper", "numpy", "soundfile", "huggingface-hub")}


# mlx 0.32.0 (frozen) ships no arm64 wheel before macOS 14; newer mlx wheels
# would change the frozen dependency set, so the fix is an OS upgrade, not a
# downgrade.
MIN_MACOS = (14, 0)


def _macos_version() -> tuple | None:
    """(major, minor) of the running macOS, or None when not on macOS."""
    if sys.platform != "darwin":
        return None
    m = re.fullmatch(r"(\d+)\.(\d+).*", platform.mac_ver()[0] or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _check_macos() -> None:
    """Fail fast with an actionable message when the frozen mlx wheel can't run."""
    ver = _macos_version()
    if ver is not None and ver < MIN_MACOS:
        raise RunError(
            f"mlx 0.32.0 (frozen) has no arm64 wheel before macOS 14; this Mac is "
            f"on macOS {ver[0]}.{ver[1]}. Upgrade to macOS 14 (Sonoma) or later — "
            f"the dependency versions are frozen and cannot be downgraded."
        )


# --- extraction plan ---------------------------------------------------------
_RANGE_RE = re.compile(r"^(\d{1,2})-(\d{1,2})$")
_SINGLE_RE = re.compile(r"^(\d{1,2})$")


def parse_chapters(spec, chapter_ids) -> list:
    """Expand a --chapters spec ("1-9,preface,names") to spine-ordered ids.

    Numbers refer to chapters ("1" -> "ch01"); unknown ids raise RunError.
    Result order follows the book's spine order, not the spec order.
    """
    ids = list(chapter_ids)
    wanted = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if _RANGE_RE.fullmatch(part):
            a, b = map(int, _RANGE_RE.fullmatch(part).groups())
            if a > b:
                raise RunError(f"chapter range {part}: start > end")
            wanted.extend(f"ch{i:02d}" for i in range(a, b + 1))
        elif _SINGLE_RE.fullmatch(part):
            wanted.append(f"ch{int(part):02d}")
        else:
            wanted.append(part)
    unknown = [c for c in wanted if c not in ids]
    if unknown:
        raise RunError(f"unknown chapter id(s): {', '.join(unknown)} (available: {', '.join(ids)})")
    return [c for c in ids if c in wanted]

def build_plan(root, config=None, chapters=None, limit=None) -> dict:
    """Build deterministic paragraph-bounded sentence and clause groups."""
    root = pathlib.Path(root)
    cfg = config if config is not None else load_config(root)
    book = root / cfg.book
    chapters_data = epub.extract_chapters(book)
    ids = [c["id"] for c in chapters_data]
    if chapters is None:
        selected = chapters_data
    else:
        want = parse_chapters(chapters, ids)
        by_id = {c["id"]: c for c in chapters_data}
        selected = [by_id[cid] for cid in want]
    chunks = []
    source_paragraphs = 0
    chapter_group_counts = {}
    for c in selected:
        chapter_group_counts[c["id"]] = 0
        source_paragraphs += len(c["paragraphs"])
        for paragraph_index, text in enumerate(c["paragraphs"]):
            spans = epub.sentence_spans(text)
            for group in epub.group_sentences(text, spans):
                sentence_indexes = list(group["sentence_indexes"])
                clause_indexes = [list(index) for index in group["clause_indexes"]]
                clause_spans = [dict(span) for span in group["clause_spans"]]
                chunk = {
                    "chapter": c["id"], "paragraph_index": paragraph_index,
                    "idx": len(chunks),
                    "id": (f"{c['id']}:p{paragraph_index:04d}:"
                           f"s{group['sentence_span'][0]:04d}-{group['sentence_span'][1]:04d}:"
                           f"o{int(group['start']):06d}-{int(group['end']):06d}"),
                    "sentence_span": list(group["sentence_span"]),
                    "sentence_indexes": sentence_indexes,
                    "sentence_count": int(group["sentence_count"]),
                    "sentence_spans": [dict(spans[k]) for k in sentence_indexes],
                    "clause_indexes": clause_indexes,
                    "clause_count": int(group["clause_count"]),
                    "clause_spans": clause_spans,
                    "eligible_clause_spans": [dict(span) for span in clause_spans],
                    "source_span": [int(group["start"]), int(group["end"])],
                    "word_count": int(group["words"]), "text": group["text"],
                    "text_sha256": group["text_sha256"],
                    "paragraph_sha256": sha256_text(text),
                }
                chunks.append(chunk)
                chapter_group_counts[c["id"]] += 1
    if limit is not None:
        if limit < 0:
            raise RunError(f"limit must be >= 0, got {limit}")
        chunks = chunks[:limit]
        chapter_group_counts = {}
        for chunk in chunks:
            chapter_group_counts[chunk["chapter"]] = chapter_group_counts.get(chunk["chapter"], 0) + 1
    return {
        "book": str(book),
        "planner": {"policy": epub.SENTENCE_GROUP_POLICY,
                     "version": epub.SENTENCE_GROUP_VERSION,
                     "limits": dict(epub.SENTENCE_GROUP_LIMITS),
                     "clause_policy": CLAUSE_SPLIT_POLICY},
        "chapters": [{"id": c["id"], "title": c["title"],
                      "paragraphs": len(c["paragraphs"]),
                      "groups": chapter_group_counts.get(c["id"], 0)} for c in selected],
        "total_paragraphs": source_paragraphs,
        "total_groups": len(chunks),
        "chunks": chunks,
    }
def _plan_for_resume(root, config=None, chapters=None, limit=None,
                     resume_from=None) -> dict:
    """Build the requested chapter set, trimming it before ``limit`` applies."""
    plan = build_plan(root, config, chapters=chapters, limit=None)
    if resume_from is None:
        return build_plan(root, config, chapters=chapters, limit=limit)
    ids = [c["id"] for c in plan["chapters"]]
    if resume_from not in ids:
        raise RunError(
            f"resume_from chapter {resume_from!r} not in plan "
            f"(available: {', '.join(ids)})"
        )
    suffix = ids[ids.index(resume_from):]
    return build_plan(root, config, chapters=','.join(suffix), limit=limit)


# --- wav facts / gates (mirrors the frozen scripts' structural gates) --------
def wav_facts(path: pathlib.Path, run_started: float, errors: list) -> dict:
    """Reload a written WAV; structural facts plus gate errors appended."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    data = sf.read(str(path), dtype="int16")[0]
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak_int16 = int(np.max(np.abs(data))) if data.size else 0
    facts = {
        "exists": True,
        "bytes": path.stat().st_size,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "samples": int(data.size),
        "seconds": float(data.size / info.samplerate),
        "float_peak": float(peak_int16 / 32768.0),
        "rms": float(np.sqrt(np.mean(np.square(data.astype(np.float64) / 32768.0)))) if data.size else 0.0,
        "active_ratio": float(np.mean(np.abs(data.astype(np.float64) / 32768.0) > 0.01)) if data.size else 0.0,
        "full_scale_samples": int((np.abs(data) >= 32768).sum()),
        "finite": bool(np.isfinite(data).all()),
        "fresh": bool(path.stat().st_mtime >= run_started - 1),
        "sha256": sha256_file(path),
    }
    if facts["sample_rate"] != SAMPLE_RATE:
        errors.append(f"{path.name} sr {facts['sample_rate']} != {SAMPLE_RATE}")
    if facts["channels"] != 1:
        errors.append(f"{path.name} channels {facts['channels']} != 1")
    if facts["subtype"] != "PCM_16":
        errors.append(f"{path.name} subtype {facts['subtype']} != PCM_16")
    if facts["bytes"] == 0 or facts["samples"] == 0:
        errors.append(f"{path.name} empty")
    if not facts["fresh"]:
        errors.append(f"{path.name} not fresh")
    if not facts["finite"]:
        errors.append(f"{path.name} non-finite")
    if not facts["rms"] > 1e-4:
        errors.append(f"{path.name} rms {facts['rms']} <= 1e-4")
    if not facts["active_ratio"] > 0.01:
        errors.append(f"{path.name} active_ratio {facts['active_ratio']} <= 0.01")
    if not facts["float_peak"] < 1.0:
        errors.append(f"{path.name} float_peak {facts['float_peak']} >= 1.0")
    if facts["full_scale_samples"] > 0:
        errors.append(f"{path.name} has full-scale samples")
    return facts


# --- model loader ------------------------------------------------------------
def _set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_model_once(cfg: Config, offline=None):
    """Load the configured Qwen model exactly once per process (caller caches).

    offline: None = auto (offline when the cache is complete, else allow
    downloads); True = forbid downloads (fail fast listing missing files);
    False = allow downloads. Env is pinned offline before the import whenever
    the cache is complete or offline is forced, exactly like the frozen
    scripts.
    """
    cache = model_cache_state(cfg.model_repo, cfg.model_revision)
    if cache["missing"]:
        missing = ", ".join(cache["missing"])
        if offline is True:
            raise RunError(
                f"model cache incomplete and --offline: missing required files "
                f"in {cache['snapshot_dir']}: {missing}"
            )
        print(f"  model cache incomplete ({len(cache['missing'])} files); downloads allowed", file=sys.stderr)
    # A complete pinned snapshot is loaded by exact revision, so refs/main is
    # informational only (reported in preflight) and never blocks generation,
    # even if it points elsewhere.
    use_offline = offline if offline is not None else (not cache["missing"])
    if use_offline:
        _set_offline_env()
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as e:
        raise RunError(
            "mlx-audio not importable; install the pinned env with `uv sync --locked` "
            "(mlx-audio==0.4.8). It is imported lazily — this error only appears when "
            "generation is first used."
        ) from e
    t0 = time.perf_counter()
    model = load_model(cfg.model_repo, revision=cfg.model_revision)
    return model, time.perf_counter() - t0


# --- preflight ---------------------------------------------------------------
def _extraction_summary(root, cfg: Config) -> dict:
    plan = build_plan(root, cfg)
    return {
        "chapters": plan["chapters"],
        "total_paragraphs": plan["total_paragraphs"],
        "book": plan["book"],
    }


def _benchmark_run(root, out_dir, cfg: Config, sentence, offline, validate,
                   ref_wav: pathlib.Path, ref_text: str) -> dict:
    """One pilot generation (one model load) -> benchmark.json + wav.

    Writes <out>/benchmark/pilot.wav and benchmark.json (atomic); the pilot
    wav is deleted on gate failure. Broad duration bounds apply only to the
    default pilot sentence; custom --sentence text gets structural gates only.
    """
    out_dir = pathlib.Path(out_dir)
    bm_path = out_dir / BENCHMARK_REL
    pilot_wav = out_dir / PILOT_WAV_REL
    bm_path.parent.mkdir(parents=True, exist_ok=True)

    inv = _invocation(cfg)
    model, load_seconds = load_model_once(cfg, offline=offline)
    started = time.time()
    import mlx.core as mx
    mx.random.seed(inv["seed"])
    gen_started = time.perf_counter()
    results = list(model.generate(
        text=sentence,
        ref_audio=str(ref_wav),
        ref_text=ref_text,
        lang_code=inv["lang_code"],
        stream=inv["stream"],
        max_tokens=inv["max_tokens"],
    ))
    gen_seconds = time.perf_counter() - gen_started

    errors = []
    result_facts = None
    audio = None
    if len(results) != 1:
        errors.append(f"generation results = {len(results)}, expected exactly 1")
    else:
        r = results[0]
        result_facts = {
            "segment_idx": int(r.segment_idx),
            "sample_rate": int(r.sample_rate),
            "samples": int(r.samples),
            "audio_duration_seconds": r.audio_duration,
            "token_count": int(r.token_count),
            "real_time_factor": float(r.real_time_factor),
            "processing_time_seconds": float(r.processing_time_seconds),
            "peak_memory_usage_gb": float(r.peak_memory_usage),
            "is_streaming_chunk": bool(r.is_streaming_chunk),
            "is_final_chunk": bool(r.is_final_chunk),
        }
        if int(r.sample_rate) != SAMPLE_RATE:
            errors.append(f"result sample_rate {r.sample_rate} != {SAMPLE_RATE}")
        audio = r.audio

    if pilot_wav.exists():
        pilot_wav.unlink()
    if audio is not None:
        import numpy as np
        import soundfile as sf

        sf.write(str(pilot_wav), np.asarray(audio), SAMPLE_RATE, subtype="PCM_16")
    out_facts = wav_facts(pilot_wav, started, errors) if pilot_wav.exists() else {"exists": False}
    if sentence == PILOT_SENTENCE and out_facts.get("exists"):
        lo, hi = PILOT_DURATION_BOUNDS
        if not (lo <= out_facts["seconds"] <= hi):
            errors.append(f"pilot seconds {out_facts['seconds']} outside {lo}-{hi}")

    words = len(sentence.split())
    tokens = result_facts["token_count"] if result_facts else None
    audio_seconds = out_facts.get("seconds") if out_facts.get("exists") else (
        result_facts["audio_duration_seconds"] if result_facts else None
    )
    metrics = {
        "words": words,
        "tokens": tokens,
        "generation_seconds": round(gen_seconds, 4),
        "audio_seconds": round(audio_seconds, 4) if audio_seconds is not None else None,
    }
    if tokens and audio_seconds:
        metrics["generation_seconds_per_token"] = round(gen_seconds / tokens, 6)
        metrics["tokens_per_audio_second"] = round(tokens / audio_seconds, 4)
    if audio_seconds and words:
        metrics["audio_seconds_per_word"] = round(audio_seconds / words, 6)
        metrics["generation_seconds_per_word"] = round(gen_seconds / words, 6)

    asr = None
    if validate:
        from . import asr as _asr_mod

        try:
            v = _asr_mod.AsrValidator(
                model_repo=cfg.asr_repo, revision=cfg.asr_revision,
                cache_path=out_dir / ASR_CACHE_REL,
            )
            record = v.validate_chunk(pilot_wav, sentence, chunk_id="pilot")
            asr = {
                "verdict": record["verdict"],
                "reasons": record["reasons"],
                "rtf": record.get("rtf"),
                "asr_seconds": record.get("asr_seconds"),
                "model_repo": record["asr"]["model_repo"],
                "model_revision": record["asr"]["model_revision"],
            }
        except Exception as e:
            errors.append(f"benchmark ASR failed: {type(e).__name__}: {e}")
    if asr is not None and asr["verdict"] != "PASS":
        errors.append(f"benchmark ASR {asr['verdict']}: " + "; ".join(asr.get("reasons") or []))

    benchmark = {
        "sentence": sentence,
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision,
                  "snapshot_dir": model_cache_state(cfg.model_repo, cfg.model_revision)["snapshot_dir"]},
        "reference": {"wav_sha256": sha256_file(ref_wav), "text_sha256": sha256_text(ref_text)},
        "invocation": _invocation(cfg),
        "result": result_facts,
        "output": out_facts,
        "metrics": metrics,
        "asr": asr,
        "runtime_seconds": round(gen_seconds + load_seconds, 4),
        "load_seconds": round(load_seconds, 4),
        "generation_seconds": round(gen_seconds, 4),
        "started_unix": started,
        "verdict": "FAIL" if errors else "PASS",
        "errors": errors,
    }
    _atomic_write_json(bm_path, benchmark)
    # Benchmark stays on disk (metrics + asr) for diagnosis/ETA even on gate
    # failure; the overall preflight exit status is decided by the CLI from
    # benchmark["verdict"].
    return benchmark


def preflight(root, out_dir, *, sentence=PILOT_SENTENCE, benchmark=True,
              offline=None, validate=True, config=None) -> dict:
    """Inputs + hashes, model cache, env packages, extraction counts, and the
    optional measured pilot (one model load). No model is loaded when
    benchmark=False (pure dry run)."""
    root = pathlib.Path(root)
    out_dir = pathlib.Path(out_dir)
    _check_macos()
    cfg = config if config is not None else load_config(root)
    inputs = verify_inputs(root, cfg.inputs)
    cache = model_cache_state(cfg.model_repo, cfg.model_revision)
    packages = package_versions()
    extract = _extraction_summary(root, cfg)
    report = {
        "root": str(root),
        "out": str(out_dir),
        "inputs": inputs,
        "model_cache": cache,
        "packages": packages,
        "extraction": extract,
        "benchmark": None,
        "verdict": "PASS",
    }
    if benchmark:
        report["benchmark"] = _benchmark_run(
            root, out_dir, cfg, sentence, offline=offline, validate=validate,
            ref_wav=root / cfg.audio, ref_text=(root / cfg.transcript).read_text())
        if report["benchmark"]["verdict"] != "PASS":
            report["verdict"] = "FAIL"
    return report



class Generator:
    """Resumable full-book generation with atomic PCM16 checkpoints."""

    def __init__(self, root, out_dir, *, chapters=None, limit=None, force=False,
                 resume_from=None, offline=None, validate=True, config=None):
        self.root = pathlib.Path(root)
        self.out_dir = pathlib.Path(out_dir)
        self.force = force
        self.validate = validate
        self.offline = offline
        self.config = config if config is not None else load_config(self.root)
        self.inputs = verify_inputs(self.root, self.config.inputs)
        self.ref_wav = self.root / self.config.audio
        self.ref_text = (self.root / self.config.transcript).read_text()  # verbatim, trailing newline kept
        self.ref_wav_sha = self.inputs[self.config.audio]["sha256"]
        self.ref_text_sha = sha256_text(self.ref_text)
        self.plan = _plan_for_resume(
            self.root, self.config, chapters=chapters, limit=limit,
            resume_from=resume_from,
        )
        self.fingerprint = run_fingerprint(
            self.config, self.ref_wav_sha, self.ref_text_sha
        )
        self.start_idx = 0
        self.chunks_dir = self.out_dir / "chunks"
        self.state_path = self.out_dir / STATE_REL
        self.jsonl_path = self.out_dir / CHUNKS_JSONL_REL
        self._model = None
        self.load_seconds = None
        self._validator_obj = None
        self._records = self._load_records()
        self._attempt_failures = {}
        self._forced_parents = set()
        self._load_state()

    def _invalidate_forced_parent(self, parent: dict, *, persist=True) -> None:
        """Forget forced checkpoint choices while retaining existing WAV files."""
        parent_id = parent["id"]
        ids = {parent_id}
        ids.update(child["id"] for child in self._child_chunks(parent))
        self._forced_parents.add(parent_id)
        for chunk_id in ids:
            self.done.pop(chunk_id, None)
            self._records.pop(chunk_id, None)
        if persist:
            self._save_records()
            self._save_state()

    def _state_chapters(self) -> list:
        return [{"id": c["id"], "title": c["title"],
                 "paragraphs": c["paragraphs"], "groups": c["groups"]}
                for c in self.plan["chapters"]]

    def _plan_identity(self) -> dict:
        return {
            "planner": self.plan["planner"],
            "chapters": self._state_chapters(),
            "chunk_ids": [c["id"] for c in self.plan["chunks"]],
            "text_hashes": [_chunk_identity_hash(c) for c in self.plan["chunks"]],
            "total_paragraphs": self.plan["total_paragraphs"],
            "total_groups": self.plan["total_groups"],
        }

    def _load_state(self):
        self.done = {}
        self.started_unix = time.time()
        if self.state_path.is_file():
            st = json.loads(self.state_path.read_text())
            if st.get("fingerprint") != self.fingerprint or \
                    not _state_plan_matches(st.get("plan"), self.plan):
                if not self.force:
                    raise RunError(
                        f"existing state at {self.state_path} does not match this run "
                        "(inputs/model/reference/chapters/limit changed). Use --force to "
                        "discard resume and regenerate, or a fresh --out."
                    )
            else:
                self.done = st.get("done", {})
                self.started_unix = st.get("started_unix", self.started_unix)
        self._save_state()

    def _save_state(self):
        state = {
            "fingerprint": self.fingerprint,
            "model": {"repo": self.config.model_repo, "revision": self.config.model_revision},
            "reference": {"wav_sha256": self.ref_wav_sha, "text_sha256": self.ref_text_sha},
            "invocation": _invocation(self.config),
            "asr": {"repo": self.config.asr_repo, "revision": self.config.asr_revision},
            "sample_rate": SAMPLE_RATE,
            "plan": self._plan_identity(),
            "done": self.done,
            "started_unix": self.started_unix,
        }
        _atomic_write_json(self.state_path, state)

    def _records_path(self):
        return self.out_dir / RECORDS_REL

    def _load_records(self) -> dict:
        p = self._records_path()
        if not p.is_file():
            return {}
        try:
            return {r["chunk_id"]: r for r in json.loads(p.read_text())["records"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            return {}

    def _save_records(self):
        payload = {
            "asr": {"model_repo": self.config.asr_repo,
                    "model_revision": self.config.asr_revision},
            "records": sorted(self._records.values(), key=lambda r: r.get("chunk_id") or ""),
        }
        _atomic_write_json(self._records_path(), payload)

    def _pending_chunks(self):
        if self.force:
            for parent in self.plan["chunks"]:
                if parent["id"] not in self._forced_parents:
                    self._invalidate_forced_parent(parent)
            return list(self.plan["chunks"])
        return [chunk for chunk in self.plan["chunks"] if not self._is_done(chunk)]
    @staticmethod
    def _chunk_status(chunk: dict, done: dict, force: bool) -> str:
        if force:
            return "pending"
        d = done.get(chunk["id"])
        return "done" if d and d.get("text_sha256") == chunk["text_sha256"] else "pending"

    def _is_done(self, chunk: dict) -> bool:
        return self._unit_done(chunk) or self._children_complete(chunk)


    def _validator(self):
        if self._validator_obj is None:
            from . import asr

            self._validator_obj = asr.AsrValidator(
                model_repo=self.config.asr_repo, revision=self.config.asr_revision,
                cache_path=self.out_dir / ASR_CACHE_REL,
            )
        return self._validator_obj

    def _child_chunks(self, parent: dict) -> list[dict]:
        spans = _planned_clauses(parent)
        if len(spans) <= 1:
            return []
        out = []
        for n, span in enumerate(spans):
            child = dict(parent)
            child.update({
                "id": f"{parent['id']}:c{n:04d}",
                "chunk_id": f"{parent['id']}:c{n:04d}",
                "parent_id": parent["id"], "child_index": n,
                "idx": parent.get("idx", 0) * 10000 + n,
                "source_span": [span["start"], span["end"]],
                "text": span["text"], "text_sha256": sha256_text(span["text"]),
                "word_count": span.get("words", len(span["text"].split())),
                "clause_spans": [span], "clause_indexes": [[
                    span.get("sentence_index", span.get("sentence_start", 0)),
                    span.get("clause_index", n)]], "clause_count": 1,
            })
            out.append(child)
        return out

    def _unit_done(self, chunk: dict) -> bool:
        d = self.done.get(chunk["id"])
        if not d or d.get("text_sha256") != chunk["text_sha256"]:
            return False
        if float(d.get("tail_frame_peak", 0.0)) > qwenfix.TAIL_FRAME_PEAK_MAX:
            return False
        wav = self.out_dir / d["wav"]
        if not wav.is_file() or d.get("wav_sha256") != sha256_file(wav):
            return False
        if not self.validate:
            return True
        return _take_passes(self._records.get(chunk["id"]))

    def _children_complete(self, parent: dict) -> bool:
        children = self._child_chunks(parent)
        return bool(children) and all(self._unit_done(c) for c in children)

    def _record_chunk(self, record: dict) -> None:
        self.done[record["id"]] = {
            "text_sha256": record["text_sha256"], "wav": record["wav"],
            "wav_sha256": record["wav_sha256"], "samples": record["samples"],
            "seconds": record["seconds"],
            "terminal_silence_seconds": record["terminal_silence_seconds"],
            "tail_frame_peak": record.get("tail_frame_peak", 0.0),
        }
        # Persist the record before state marks this unit done.
        self._save_records()
        self._save_state()

    @staticmethod
    def _terminal_silence_seconds(audio) -> float:
        """Measure trailing quiet for diagnostics; lexical gates decide acceptance."""
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        nonquiet = np.flatnonzero(np.abs(samples) > 0.01)
        if not nonquiet.size:
            return samples.size / SAMPLE_RATE
        return (samples.size - int(nonquiet[-1]) - 1) / SAMPLE_RATE

    @staticmethod
    def _seed(base: int, chunk_id: str, attempt: int) -> int:
        digest = hashlib.sha256(f"{chunk_id}:{attempt}".encode()).digest()
        return (base + int.from_bytes(digest[:4], "big")) % (2 ** 32)

    def _ref_audio_array(self):
        """Reference waveform as an mx.array, loaded once per run."""
        if getattr(self, "_ref_audio_mx", None) is None:
            import mlx.core as mx
            import numpy as np
            import soundfile as sf

            data, sr = sf.read(str(self.ref_wav))
            if int(sr) != SAMPLE_RATE:
                raise RunError(f"reference sample_rate {sr} != {SAMPLE_RATE}")
            self._ref_audio_mx = mx.array(np.asarray(data, dtype=np.float32))
        return self._ref_audio_mx

    def _generate(self, chunk: dict, model, attempt: int = 0) -> dict:
        import mlx.core as mx

        inv = _invocation(self.config)
        if int(model.sample_rate) != SAMPLE_RATE:
            raise RunError(f"model sample_rate {model.sample_rate} != {SAMPLE_RATE}")
        t0 = time.perf_counter()
        seed = self._seed(inv["seed"], chunk["id"], attempt)
        mx.random.seed(seed)
        audio = qwenfix.generate_icl_tail_safe(
            model, chunk["text"], self._ref_audio_array(), self.ref_text,
            inv["lang_code"], inv["max_tokens"])
        gen_seconds = time.perf_counter() - t0
        import numpy as np
        import soundfile as sf

        tail_frame_peak = qwenfix.tail_frame_peak(audio, SAMPLE_RATE)
        sibilant_frac = qwenfix.final_sibilant_high_frac(
            audio, SAMPLE_RATE, chunk["text"])
        rel = f"chunks/{chunk['chapter']}/{chunk['id'].replace(':', '-')}.wav"
        wav = self.out_dir / rel
        wav.parent.mkdir(parents=True, exist_ok=True)
        # .tmp.wav suffix: soundfile infers format from the extension; rename is atomic.
        tmp = wav.with_name(wav.name + ".tmp.wav")
        try:
            sf.write(str(tmp), audio, SAMPLE_RATE, subtype="PCM_16")
            os.replace(tmp, wav)
        except BaseException:
            if tmp.exists():
                tmp.unlink()
            raise
        started = time.time()
        errors = []
        facts = wav_facts(wav, started, errors)
        if errors:
            raise RunError(f"{chunk['id']}: invalid output:\n  " + "\n  ".join(errors))
        return {
            "id": chunk["id"], "chunk_id": chunk["id"],
            "chapter": chunk["chapter"], "idx": chunk.get("idx"),
            "parent_id": chunk.get("parent_id"), "child_index": chunk.get("child_index"),
            "paragraph_index": chunk["paragraph_index"],
            "sentence_span": chunk["sentence_span"],
            "sentence_indexes": chunk["sentence_indexes"],
            "sentence_count": chunk["sentence_count"],
            "sentence_spans": chunk["sentence_spans"],
            "clause_indexes": chunk.get("clause_indexes", []),
            "clause_count": chunk.get("clause_count", 0),
            "clause_spans": chunk.get("clause_spans", []),
            "source_span": chunk["source_span"], "word_count": chunk["word_count"],
            "text": chunk["text"], "text_sha256": chunk["text_sha256"],
            "wav": rel, "wav_sha256": facts["sha256"],
            "sample_rate": facts["sample_rate"], "channels": facts["channels"],
            "subtype": facts["subtype"], "samples": facts["samples"],
            "terminal_silence_seconds": round(self._terminal_silence_seconds(audio), 4),
            "eos_hold_frames": qwenfix.EOS_HOLD_FRAMES,
            "tail_frame_peak": round(tail_frame_peak, 6),
            "final_sibilant_high_frac": (
                None if sibilant_frac is None else round(sibilant_frac, 4)),
            "seconds": facts["seconds"], "generation_seconds": round(gen_seconds, 4),
            "model": {"repo": self.config.model_repo, "revision": self.config.model_revision},
            "reference_wav_sha256": self.ref_wav_sha,
            "reference_text_sha256": self.ref_text_sha,
            "seed": seed, "attempt": attempt, "started_unix": time.time(),
        }
    def _validate_chunk(self, chunk: dict, record: dict):
        """Validate one take without checkpointing its verdict."""
        return self._validator().validate_many([{
            "wav": str(self.out_dir / record["wav"]),
            "expected_text": chunk["text"],
            "chunk_id": record["id"],
        }])[0]

    @staticmethod
    def _validation_failure(vrec: dict | None) -> str:
        reasons = [str(r) for r in (vrec or {}).get("reasons", []) if r]
        if not reasons and isinstance(vrec, dict):
            terminal = vrec.get("terminal") or {}
            if terminal.get("matched") is False:
                reasons.append(f"terminal phrase missing: {terminal.get('expected')!r}")
            for boundary in (vrec.get("punctuation") or {}).get("boundaries", []):
                if boundary.get("aligned") and boundary.get("passed") is False:
                    reasons.append(
                        f"{boundary.get('kind', 'punctuation')} pause failed"
                    )
        return "; ".join(reasons) or "terminal or punctuation gate failed"

    def _generate_with_retries(self, chunk: dict, model):
        """Retry structural defects up to four takes, ASR failures twice.

        Generation is not bit-reproducible even with fixed seeds, so each
        attempt is a fresh draw. Structural gates (truncated tail, missing
        final sibilant) can only be fixed by another draw and get the full
        attempt budget; ASR validation failures fall back to clause
        splitting after two, as before.
        """
        failures = []
        validation_failures = 0
        for attempt in range(4):
            if validation_failures >= 2:
                break
            try:
                record = self._generate(chunk, model, attempt=attempt)
                structural = None
                if float(record.get("tail_frame_peak", 0.0)) > qwenfix.TAIL_FRAME_PEAK_MAX:
                    structural = "speech reaches into the room-tone tail"
                sib = record.get("final_sibilant_high_frac")
                if structural is None and sib is not None \
                        and float(sib) < qwenfix.SIBILANT_HIGH_FRAC_MIN:
                    structural = (
                        f"final sibilant missing (high-band {float(sib):.3f} "
                        f"< {qwenfix.SIBILANT_HIGH_FRAC_MIN})")
                if structural is not None:
                    failures.append(f"attempt {attempt}: {structural}")
                    print(f"  {chunk['id']} {failures[-1]}", file=sys.stderr)
                    continue
                vrec = self._validate_chunk(chunk, record) if self.validate else None
                if vrec and not _take_passes(vrec):
                    validation_failures += 1
                    reason = self._validation_failure(vrec)
                    failures.append(f"attempt {attempt}: {reason}")
                    print(f"  {chunk['id']} {failures[-1]}", file=sys.stderr)
                    continue
                if vrec:
                    self._records[record["id"]] = vrec
                return record, failures
            except Exception as exc:
                validation_failures += 1
                reason = f"{type(exc).__name__}: {exc}"
                failures.append(f"attempt {attempt}: {reason}")
                print(f"  {chunk['id']} {failures[-1]}", file=sys.stderr)
        self._attempt_failures.setdefault(chunk["id"], []).extend(failures)
        return None, failures

    @staticmethod
    def _retry_error(chunk: dict, failures: list[str]) -> RunError:
        detail = "; ".join(failures) or "no diagnostic available"
        return RunError(f"{chunk['id']}: every retry attempt failed: {detail}")


    def _plan_complete(self) -> bool:
        """Return whether every parent has a valid unit or complete children."""
        for parent in self.plan["chunks"]:
            children = self._child_chunks(parent)
            if children and all(self._unit_done(child) for child in children):
                continue
            if not self._unit_done(parent):
                return False
        return True

    def _all_done(self) -> bool:
        return self._plan_complete()

    # -- release gate ---------------------------------------------------------
    @staticmethod
    def _record_ok(rec, text_sha256: str, wav_sha256: str,
                   asr_repo: str, asr_revision: str) -> bool:
        if not rec or rec.get("verdict") != "PASS":
            return False
        if not _take_passes(rec) or rec.get("validation_policy") != asr.VALIDATION_POLICY:
            return False
        if rec.get("wav_sha256") != wav_sha256 or rec.get("expected_sha256") != text_sha256:
            return False
        cfg = rec.get("asr") or {}
        return cfg.get("model_repo") == asr_repo and cfg.get("model_revision") == asr_revision

    def _release_gate(self) -> dict:
        blocked = []
        for parent in self.plan["chunks"]:
            children = self._child_chunks(parent)
            if children and all(self._unit_done(c) for c in children):
                for child in children:
                    done = self.done[child["id"]]
                    if not self._record_ok(self._records.get(child["id"]),
                                           child["text_sha256"], done["wav_sha256"],
                                           self.config.asr_repo, self.config.asr_revision):
                        blocked.append(child["id"])
                continue
            if not self._unit_done(parent):
                blocked.append(parent["id"])
                continue
            done = self.done[parent["id"]]
            if not self._record_ok(self._records.get(parent["id"]),
                                   parent["text_sha256"], done["wav_sha256"],
                                   self.config.asr_repo, self.config.asr_revision):
                blocked.append(parent["id"])
        return {"release": not blocked, "blocked": blocked}

    def _remove_book_artifacts(self) -> None:
        """Delete stale final book artifacts so an old book cannot masquerade
        as current when the validation gate blocks release."""
        for rel in (BOOK_WAV_REL, BOOK_JSON_REL):
            p = self.out_dir / rel
            if p.exists():
                try:
                    p.unlink()
                    print(f"  removed stale {rel} (validation gate failed)")
                except OSError as e:
                    print(f"  warning: could not remove {rel}: {e}", file=sys.stderr)

    # -- concatenation --------------------------------------------------------
    @staticmethod
    def _wav_header_bytes(total_frames: int) -> bytes:
        data_size = total_frames * 2  # mono 16-bit
        return _WAV_HEADER.pack(
            b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1, 1,
            SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b"data", data_size,
        )

    @staticmethod
    def _payload_bytes_equal(book: pathlib.Path, sources, header_size=44) -> bool:
        with open(book, "rb") as fb:
            fb.seek(header_size)
            for src in sources:
                with open(src, "rb") as fs:
                    fs.seek(header_size)
                    while True:
                        cs = fs.read(1 << 20)
                        if not cs:
                            break
                        if fb.read(len(cs)) != cs:
                            return False
            return fb.read(1) == b""

    @staticmethod
    def _boundary_padding_frames(left_chunk, left_wav, right_chunk, right_wav) -> int:
        """Return silence needed to reach the source-aware minimum pause."""
        import numpy as np
        import soundfile as sf

        left, left_sr = sf.read(str(left_wav), dtype="float32")
        right, right_sr = sf.read(str(right_wav), dtype="float32")
        if left_sr != SAMPLE_RATE or right_sr != SAMPLE_RATE:
            raise RunError("boundary WAV sample rate mismatch")
        left = np.asarray(left).reshape(-1)
        right = np.asarray(right).reshape(-1)
        left_active = np.flatnonzero(np.abs(left) > 0.01)
        right_active = np.flatnonzero(np.abs(right) > 0.01)
        trailing = left.size - int(left_active[-1]) - 1 if left_active.size else left.size
        leading = int(right_active[0]) if right_active.size else right.size
        minimum = (PARAGRAPH_PAUSE_SECONDS
                   if left_chunk["paragraph_index"] != right_chunk["paragraph_index"]
                   else SENTENCE_PAUSE_SECONDS)
        return max(0, int(round(minimum * SAMPLE_RATE)) - trailing - leading)

    @classmethod
    def _assembly_parts(cls, chunks, wavs):
        """Return ordered WAVs and source-aware zero-padding frame counts."""
        parts = []
        for i, (chunk, wav) in enumerate(zip(chunks, wavs)):
            parts.append((wav, 0))
            if i + 1 < len(wavs):
                pad = cls._boundary_padding_frames(chunk, wav, chunks[i + 1], wavs[i + 1])
                if pad:
                    parts.append((None, pad))
        return parts

    @staticmethod
    def _write_assembly(path, total_frames, parts):
        with open(path, "wb") as out:
            out.write(Generator._wav_header_bytes(total_frames))
            for wav, silence_frames in parts:
                if silence_frames:
                    out.write(b"\0\0" * silence_frames)
                    continue
                with open(wav, "rb") as src:
                    src.seek(44)
                    for data in iter(lambda: src.read(1 << 20), b""):
                        out.write(data)

    def _assembly_units(self, parent: dict) -> list[dict]:
        children = self._child_chunks(parent)
        if children and self._children_complete(parent) and not self._unit_done(parent):
            return children
        return [parent]

    def _concatenate(self) -> dict:
        """Assemble each parent, or its complete ordered child units."""
        import soundfile as sf
        chunks, wavs = [], []
        for parent in self.plan["chunks"]:
            for chunk in self._assembly_units(parent):
                done = self.done.get(chunk["id"])
                if not done:
                    raise RunError(f"cannot concatenate: {chunk['id']} not generated")
                wav = self.out_dir / done["wav"]
                if not wav.is_file():
                    raise RunError(f"cannot concatenate: {chunk['id']} wav missing ({wav})")
                chunks.append(chunk)
                wavs.append(wav)
        parts = self._assembly_parts(chunks, wavs)
        inserted = sum(frames for _, frames in parts)
        total_frames = sum(self.done[c["id"]]["samples"] for c in chunks) + inserted
        book_wav = self.out_dir / BOOK_WAV_REL
        tmp = book_wav.with_name(book_wav.name + ".tmp")
        self._write_assembly(tmp, total_frames, parts)
        os.replace(tmp, book_wav)
        info = sf.info(str(book_wav))
        errors = []
        if (int(info.samplerate) != SAMPLE_RATE or int(info.channels) != 1 or
                info.subtype != "PCM_16" or int(info.frames) != total_frames):
            errors.append(f"book.wav header mismatch: frames={info.frames} != {total_frames}")
        chapter_ids = []
        for chunk in chunks:
            if chunk["chapter"] not in chapter_ids:
                chapter_ids.append(chunk["chapter"])
        manifest = {
            "book": BOOK_WAV_REL,
            "method": "PCM16 chunk concatenation with minimum 250 ms sentence and 500 ms paragraph pauses",
            "sample_rate": SAMPLE_RATE, "channels": 1, "subtype": "PCM_16",
            "samples": total_frames, "inserted_silence_samples": inserted,
            "seconds": round(total_frames / SAMPLE_RATE, 4),
            "bytes": book_wav.stat().st_size, "sha256": sha256_file(book_wav),
            "verdict": "FAIL" if errors else "PASS", "errors": errors,
            "chapters": chapter_ids, "chunks": [c["id"] for c in chunks],
        }
        _atomic_write_json(self.out_dir / BOOK_JSON_REL, manifest)
        if errors:
            raise RunError("concatenation failed:\n  " + "\n  ".join(errors))
        return manifest

    # -- run ------------------------------------------------------------------
    def run(self) -> dict:
        pending = self._pending_chunks()
        total = len(pending)
        if total == 0:
            print(f"  nothing to generate ({len(self.plan['chunks'])} planned, all done)")
        else:
            self._model, self.load_seconds = load_model_once(self.config, offline=self.offline)
            print(f"  model loaded in {self.load_seconds:.1f}s; {total} chunk(s) to generate")
        gen_cum = audio_cum = 0.0
        for i, chunk in enumerate(pending, 1):
            record, failures = self._generate_with_retries(chunk, self._model)
            if record is None:
                children = self._child_chunks(chunk)
                if not children:
                    raise self._retry_error(chunk, failures)
                records = []
                for child in children:
                    child_record, child_failures = self._generate_with_retries(child, self._model)
                    if child_record is None:
                        raise self._retry_error(child, child_failures)
                    self._record_chunk(child_record)
                    records.append(child_record)
            else:
                self._record_chunk(record)
                records = [record]
            gen_cum += sum(r["generation_seconds"] for r in records)
            audio_cum += sum(r["seconds"] for r in records)
            print(f"  {chunk['id']}  gen {sum(r['generation_seconds'] for r in records):5.2f}s "
                  f"audio {sum(r['seconds'] for r in records):6.2f}s  ({i}/{total})")
        try:
            self._save_records()
        except Exception as e:
            print(f"  warning: could not write validation records: {e}", file=sys.stderr)
        summary = {
            "plan": {"chapters": len(self.plan["chapters"]),
                     "source_paragraphs": self.plan["total_paragraphs"],
                     "groups": self.plan["total_groups"]},
            "generated": len(pending),
            "done_total": sum(1 for c in self.plan["chunks"] if self._is_done(c)),
            "generation_seconds": round(gen_cum, 2),
            "audio_seconds": round(audio_cum, 2),
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds is not None else None,
        }
        if self.plan["chunks"] and self._all_done():
            gate = self._release_gate()
            if gate["release"]:
                concat = self._concatenate()
                print(f"  concatenated {len(concat['chapters'])} chapters -> {self.out_dir / BOOK_WAV_REL} ({concat['seconds']:.1f}s audio)")
                summary["book"] = concat
            else:
                self._remove_book_artifacts()
                summary["book"] = None
                summary["blocked"] = gate["blocked"]
        elif self.plan["chunks"]:
            print("  book.wav not built: plan incomplete; re-run to resume")
        return summary


# --- validate ----------------------------------------------------------------
def validate_generated(root, out_dir, *, chapters=None, limit=None, config=None) -> dict:
    """Validate generated chunks through the persistent AsrValidator.

    Requires matching state and source text, skips missing chunks, and
    atomically writes ``<out>/validation/records.json``.
    """
    from . import asr

    root = pathlib.Path(root)
    out_dir = pathlib.Path(out_dir)
    cfg = config if config is not None else load_config(root)
    inputs = verify_inputs(root, cfg.inputs)
    cur_fingerprint = run_fingerprint(
        cfg,
        inputs[cfg.audio]["sha256"],
        sha256_text((root / cfg.transcript).read_text()),
    )
    plan = build_plan(root, cfg, chapters=chapters, limit=limit)
    state_path = out_dir / STATE_REL
    if not state_path.is_file():
        raise RunError(f"no generation state at {state_path}; run `audiobook generate` first")
    st = json.loads(state_path.read_text())
    if st.get("fingerprint") != cur_fingerprint:
        raise RunError(
            "state fingerprint differs from current inputs/model/reference; "
            "re-run generate"
        )
    if not _state_plan_matches(st.get("plan"), plan):
        raise RunError("state plan differs from requested chapters/limit; re-run generate")
    g = Generator(root, out_dir, config=cfg, chapters=chapters, limit=limit)
    specs = []
    for parent in plan["chunks"]:
        parent_done = st.get("done", {}).get(parent["id"])
        parent_wav = out_dir / parent_done["wav"] if parent_done and parent_done.get("wav") else None
        children = g._child_chunks(parent)
        candidates = [parent] if parent_done and parent_wav and parent_wav.is_file() else children
        for chunk in candidates:
            d = st.get("done", {}).get(chunk["id"])
            if not d:
                continue
            wav = out_dir / d["wav"]
            if not wav.is_file():
                continue
            specs.append({"wav": str(wav), "expected_text": chunk["text"],
                          "chunk_id": chunk["id"]})
    if not specs:
        raise RunError("no generated chunks to validate")
    v = asr.AsrValidator(model_repo=cfg.asr_repo, revision=cfg.asr_revision,
                         cache_path=out_dir / ASR_CACHE_REL)
    records = v.validate_many(specs)
    current = {}
    records_path = out_dir / RECORDS_REL
    if records_path.is_file():
        try:
            current = {r["chunk_id"]: r
                       for r in json.loads(records_path.read_text()).get("records", [])
                       if isinstance(r, dict) and r.get("chunk_id")}
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            current = {}
    current.update({r["chunk_id"]: r for r in records})
    _atomic_write_json(records_path, {
        "asr": {"model_repo": cfg.asr_repo, "model_revision": cfg.asr_revision},
        "records": sorted(current.values(), key=lambda r: r.get("chunk_id") or ""),
    })
    failed = [r for r in records if r["verdict"] != "PASS"]
    cached = sum(1 for r in records if r.get("cache_hit"))
    result = {
        "chunks": len(records),
        "passed": len(records) - len(failed),
        "failed": len(failed),
        "cached": cached,
        "failures": [{"chunk_id": r.get("chunk_id"), "reasons": r["reasons"]}
                     for r in failed],
        "asr": v.stats(),
        "records": str(out_dir / RECORDS_REL),
    }
    # Assembly: reuse the Generator to get the byte-exact concatenation when
    # every planned chunk was validated and none failed. Requires the state to
    # match this run (already checked above), so resuming the generator loads
    # the same done set. Loads no model.
    if not failed:
        g = Generator(root, out_dir, config=cfg, chapters=chapters, limit=limit)
        if g._plan_complete() and g._release_gate()["release"]:
            book = g._concatenate()
            result["book"] = {
                "wav": BOOK_WAV_REL,
                "seconds": book["seconds"],
                "chapters": len(book["chapters"]),
            }
    return result


# --- eta ---------------------------------------------------------------------
def estimate(root, out_dir, config=None, plan=None) -> dict:
    """Project generation and ASR wall time from the measured pilot and the
    extraction paragraph plan. Costs are reported separately."""
    out_dir = pathlib.Path(out_dir)
    root = pathlib.Path(root)
    cfg = config if config is not None else load_config(root)
    bm_path = out_dir / BENCHMARK_REL
    if not bm_path.is_file():
        raise RunError(
            f"no benchmark at {bm_path}; run `audiobook preflight` (measured pilot) first"
        )
    bm = json.loads(bm_path.read_text())
    if bm.get("verdict") != "PASS":
        raise RunError("benchmark did not pass; re-run `audiobook preflight`")
    if plan is None:
        plan = build_plan(root, cfg)
    m = bm["metrics"]
    words = sum(len(c["text"].split()) for c in plan["chunks"])
    if m.get("audio_seconds_per_word") is None or m.get("generation_seconds_per_token") is None \
            or m.get("tokens_per_audio_second") is None:
        raise RunError("benchmark metrics incomplete; re-run `audiobook preflight`")
    est_audio = words * m["audio_seconds_per_word"]
    est_tokens = est_audio * m["tokens_per_audio_second"]
    est_gen = est_tokens * m["generation_seconds_per_token"]
    asr_rtf = (bm.get("asr") or {}).get("rtf")
    est_asr = est_audio * asr_rtf if asr_rtf else None
    load_seconds = bm.get("load_seconds")
    total = load_seconds + est_gen + (est_asr or 0.0)
    return {
        "chapters": len(plan["chapters"]),
        "paragraphs": len(plan["chunks"]),
        "words": words,
        "estimated_audio_seconds": round(est_audio, 1),
        "generation_seconds": round(est_gen, 1),
        "asr_seconds": round(est_asr, 1) if est_asr is not None else None,
        "model_load_seconds": round(load_seconds, 1) if load_seconds is not None else None,
        "total_wall_seconds": round(total, 1),
        "basis": {"sentence": bm["sentence"],
                  "audio_seconds_per_word": m["audio_seconds_per_word"],
                  "generation_seconds_per_token": m["generation_seconds_per_token"],
                  "asr_rtf": asr_rtf},
        "note": "estimate only (measured on the frozen pilot sentence and "
                "extrapolated by word/token counts)",
    }


# --- self-check (no model, no book) ------------------------------------------
def _make_wav(path: pathlib.Path, samples: list) -> None:
    """Write a minimal PCM16 24k mono WAV with the same header layout used
    for concatenation (pure stdlib, for the self-check)."""
    frames = len(samples)
    with open(path, "wb") as f:
        f.write(_WAV_HEADER.pack(
            b"RIFF", 36 + frames * 2, b"WAVE", b"fmt ", 16, 1, 1,
            SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b"data", frames * 2,
        ))
        f.write(struct.pack(f"<{frames}h", *samples))


def selfcheck() -> int:
    """Unit-level checks of hashing/fingerprint/plan/resume/concatenation.
    Pure stdlib; never loads a model or the book."""
    import tempfile
    import soundfile as sf
    import numpy as np

    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")

    print("audiobook.runner selfcheck")
    check("sha256_text deterministic", sha256_text("abc") == sha256_text("abc")
          and sha256_text("abc") != sha256_text("abd"))

    _root = pathlib.Path("/tmp/no-such-root")
    _base = Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                   "repo", "rev", "English", 4096, 42, "a-repo", "a-rev")
    fp1 = run_fingerprint(_base, "w", "t")
    fp2 = run_fingerprint(_base, "w", "t")
    fp3 = run_fingerprint(_base, "w", "t2")
    check("fingerprint deterministic", fp1 == fp2)
    check("fingerprint sensitive to reference", fp1 != fp3)
    check("fingerprint sensitive to max_tokens",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 8192, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to seed",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, 43, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to model",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo2", "rev", "English", 4096, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to asr",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, 42, "a-repo2", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to book hash",
          fp1 != run_fingerprint(Config(_root, "b.epub", "2" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to voice path",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "other.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, 42, "a-repo", "a-rev"),
                                 "w", "t"))

    # config validation (no model, no book)
    _GOOD = """[book]\npath = "books/b.epub"\nsha256 = "%s"\n[voice]\naudio = "v.wav"\naudio_sha256 = "%s"\ntranscript = "v.txt"\ntranscript_sha256 = "%s"\n[model]\nrepo = "r"\nrevision = "rev"\nlanguage = "English"\nmax_tokens = 4096\nseed = 42\n[asr]\nrepo = "a"\nrevision = "arev"\n""" % ("1" * 64, "2" * 64, "3" * 64)
    with tempfile.TemporaryDirectory() as td:
        tp = pathlib.Path(td) / config.CONFIG_NAME

        def _load_ok(txt):
            tp.write_text(txt)
            try:
                load_config(td)
                return True
            except ConfigError:
                return False

        def _load_err(txt):
            return not _load_ok(txt)

        check("config parses", _load_ok(_GOOD))
        check("config rejects missing section", _load_err(
            _GOOD.replace("[voice]", "[voice2]")))
        check("config rejects missing field", _load_err(
            _GOOD.replace("max_tokens = 4096", "")))
        check("config rejects bad sha", _load_err(
            _GOOD.replace("sha256 = \"%s\"" % ("1" * 64), "sha256 = \"zz\"")))
        check("config rejects absolute path", _load_err(
            _GOOD.replace('books/b.epub', '/etc/passwd')))
        check("config rejects dotdot path", _load_err(
            _GOOD.replace('books/b.epub', '../evil.epub')))

    check("seed deterministic", Generator._seed(42, "ch01:0008", 0) ==
          Generator._seed(42, "ch01:0008", 0))
    check("retry seed differs", Generator._seed(42, "ch01:0008", 0) !=
          Generator._seed(42, "ch01:0008", 1))
    ids = ["preface", "names", "ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09"]
    check("parse_chapters range+named",
          parse_chapters("1-3,preface", ids) == ["preface", "ch01", "ch02", "ch03"])
    check("parse_chapters single", parse_chapters("09", ids) == ["ch09"])
    check("parse_chapters raises on unknown",
          _raises(RunError, parse_chapters, "ch99", ids))
    check("parse_chapters raises on bad range",
          _raises(RunError, parse_chapters, "5-2", ids))

    plan_chunks = [
        {"id": "ch01:0000", "text_sha256": "a", "chapter": "ch01"},
        {"id": "ch01:0001", "text_sha256": "b", "chapter": "ch01"},
        {"id": "ch02:0000", "text_sha256": "c", "chapter": "ch02"},
    ]
    done = {"ch01:0000": {"text_sha256": "a"}}
    check("chunk done on hash match",
          Generator._chunk_status(plan_chunks[0], done, False) == "done")
    check("chunk pending on text change",
          Generator._chunk_status({"id": "ch01:0000", "text_sha256": "zz"}, done, False) == "pending")
    check("chunk pending when absent",
          Generator._chunk_status(plan_chunks[1], done, False) == "pending")
    check("force ignores resume",
          Generator._chunk_status(plan_chunks[0], done, True) == "pending")

    _planner = {"policy": epub.SENTENCE_GROUP_POLICY, "version": epub.SENTENCE_GROUP_VERSION,
                "limits": dict(epub.SENTENCE_GROUP_LIMITS), "clause_policy": CLAUSE_SPLIT_POLICY}
    _span_a = {"start": 0, "end": 12, "text": "First clause.", "words": 2,
               "sentence_index": 0, "clause_index": 0}
    _span_b = {"start": 13, "end": 26, "text": "Second clause.", "words": 2,
               "sentence_index": 1, "clause_index": 0}
    def _test_chunk(cid, text, span, clause):
        return {"id": cid, "text": text, "text_sha256": sha256_text(text),
                "paragraph_sha256": "p", "source_span": span,
                "sentence_span": [0, 1], "sentence_indexes": [0],
                "sentence_count": 1, "sentence_spans": [dict(clause)],
                "clause_indexes": [[0, 0]], "clause_count": 1,
                "clause_spans": [dict(clause)], "word_count": 2}
    _ca = _test_chunk("ch01:p0000:s0000-0001", _span_a["text"], [0, 12], _span_a)
    _cb = _test_chunk("ch01:p0001:s0000-0001", _span_b["text"], [13, 26], _span_b)
    cur_plan = {"planner": _planner,
                "chapters": [{"id": "ch01", "title": "T", "paragraphs": 2, "groups": 2}],
                "chunks": [_ca, _cb], "total_paragraphs": 2, "total_groups": 2}
    st_plan = {"planner": _planner, "chapters": cur_plan["chapters"],
               "chunk_ids": [_ca["id"], _cb["id"]],
               "text_hashes": [_chunk_identity_hash(_ca), _chunk_identity_hash(_cb)],
               "total_paragraphs": 2, "total_groups": 2}
    check("state plan identity matches", _state_plan_matches(st_plan, cur_plan))
    check("state plan drift (limit) detected", not _state_plan_matches(
        dict(st_plan, chunk_ids=[_ca["id"]], text_hashes=[st_plan["text_hashes"][0]], total_groups=1), cur_plan))
    check("state plan drift (text) detected", not _state_plan_matches(
        dict(st_plan, text_hashes=["x", st_plan["text_hashes"][1]]), cur_plan))
    check("state plan rejects non-plan", not _state_plan_matches(None, cur_plan))

    original_extract = epub.extract_chapters
    epub.extract_chapters = lambda _: [
        {"id": "ch01", "title": "One", "paragraphs": ["First. Second.", "Third."]},
        {"id": "ch02", "title": "Two", "paragraphs": ["Second chapter."]},
        {"id": "ch03", "title": "Three", "paragraphs": ["Third chapter."]},
    ]
    try:
        paragraph_plan = build_plan(_root, _base)
        suffix_plan = _plan_for_resume(_root, _base, resume_from="ch02")
        check("resume suffix selects chapters before plan identity",
              [c["id"] for c in suffix_plan["chapters"]] == ["ch02", "ch03"]
              and [c["chapter"] for c in suffix_plan["chunks"]] == ["ch02", "ch03"])
        check("resume suffix rejects chapter absent from selected set",
              _raises(RunError, lambda: _plan_for_resume(
                  _root, _base, chapters="ch01", resume_from="ch02")))
    finally:
        epub.extract_chapters = original_extract
    check("plan preserves paragraph provenance and exact text",
          [c["text"] for c in paragraph_plan["chunks"]] ==
          ["First. Second.", "Third.", "Second chapter.", "Third chapter."]
          and paragraph_plan["total_paragraphs"] == 4
          and paragraph_plan["total_groups"] == 4
          and paragraph_plan["chapters"][0]["paragraphs"] == 2)
    target_paragraphs = [
        "one two three four five six seven eight nine ten (eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen), nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five.",
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen: eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six.",
    ]
    original_extract = epub.extract_chapters
    epub.extract_chapters = lambda _: [{"id": "ch01", "title": "One",
                                        "paragraphs": target_paragraphs}]
    try:
        target_plan = build_plan(_root, _base)
    finally:
        epub.extract_chapters = original_extract
    target_chunks = target_plan["chunks"]
    check("strong clause metadata is persisted",
          all(c["clause_count"] == len(c["clause_spans"]) > 0
              and c["clause_indexes"]
              and c["eligible_clause_spans"] for c in target_chunks)
          and any(((s.get("boundary_after") or {}).get("kind") == "parenthetical-comma")
                  for c in target_chunks for s in c["clause_spans"])
          and any(((s.get("boundary_after") or {}).get("punctuation") == ":")
                  for c in target_chunks for s in c["clause_spans"]))
    check("planned chunks reconstruct paragraphs",
          all("".join(c["text"] for c in target_chunks
                       if c["paragraph_index"] == i) == text
              for i, text in enumerate(target_paragraphs)))
    # release-gate predicate (no model, no book)
    from . import asr as _asr
    _ar, _av = _asr.DEFAULT_MODEL_REPO, _asr.DEFAULT_MODEL_REVISION
    _punct = {"boundaries": [{"kind": "sentence_end", "aligned": True,
                              "passed": True, "expected_token_index": 0,
                              "asr_token_index": 0, "next_asr_token_index": 1,
                              "gap_s": 0.2, "threshold_s": 0.15}],
              "sentence_end": {"checked": 1, "aligned": 1, "passed": 1,
                               "failed": 0, "unaligned": 0, "gaps_s": [0.2]}}
    _cur_cfg = {"model_repo": _ar, "model_revision": _av}
    _pass = {"verdict": "PASS", "wav_sha256": "W", "expected_sha256": "E",
             "validation_policy": _asr.VALIDATION_POLICY, "asr": dict(_cur_cfg),
             "terminal": {"matched": True}, "punctuation": _punct}
    check("missing record stays pending", not _take_passes(None))
    check("FAIL record rejected", not _take_passes(dict(_pass, verdict="FAIL")))
    check("aligned punctuation failure rejected", not _punctuation_passes(
        dict(_pass, punctuation={"boundaries": [{"aligned": True, "passed": False}]})
    ))
    check("unaligned punctuation remains diagnostic", _punctuation_passes(
        dict(_pass, punctuation={"boundaries": [{"aligned": False, "passed": None}]})
    ))
    check("record_ok accepts current PASS",
          Generator._record_ok(dict(_pass), "E", "W", _ar, _av))
    check("record_ok rejects missing record",
          not Generator._record_ok(None, "E", "W", _ar, _av))
    check("record_ok rejects FAIL record",
          not Generator._record_ok(dict(_pass, verdict="FAIL"), "E", "W", _ar, _av))
    check("record_ok rejects stale wav hash",
          not Generator._record_ok(dict(_pass), "E", "W2", _ar, _av))
    check("record_ok rejects stale text hash",
          not Generator._record_ok(dict(_pass), "E2", "W", _ar, _av))
    check("record_ok rejects stale ASR config",
          not Generator._record_ok(dict(_pass, asr={"model_repo": "x", "model_revision": "y"}), "E", "W", _ar, _av))
    check("record_ok rejects empty ASR config",
          not Generator._record_ok(dict(_pass, asr={}), "E", "W", _ar, _av))
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        left = td / "left.wav"
        right = td / "right.wav"
        sf.write(left, np.r_[np.full(2400, .2), np.zeros(0)], SAMPLE_RATE, subtype="PCM_16")
        sf.write(right, np.r_[np.zeros(1200), np.full(2400, .2)], SAMPLE_RATE, subtype="PCM_16")
        same = [{"paragraph_index": 1}, {"paragraph_index": 1}]
        other = [{"paragraph_index": 1}, {"paragraph_index": 2}]
        check("sentence boundary pads to 250ms",
              Generator._boundary_padding_frames(same[0], left, same[1], right) == 4800)
        check("paragraph boundary pads to 500ms",
              Generator._boundary_padding_frames(other[0], left, other[1], right) == 10800)
    check("record_ok rejects stale validation policy",
          not Generator._record_ok(dict(_pass, validation_policy="old"), "E", "W", _ar, _av))
    quiet_tail = np.r_[np.full(SAMPLE_RATE // 5, 0.2, dtype=np.float32),
                       np.zeros(SAMPLE_RATE // 5, dtype=np.float32)]
    clipped_tail = np.full(SAMPLE_RATE // 5, 0.2, dtype=np.float32)
    check("terminal silence measured",
          Generator._terminal_silence_seconds(quiet_tail) >= 0.1)
    check("boundary speech is diagnostic only",
          Generator._terminal_silence_seconds(clipped_tail) == 0.0)

    header = Generator._wav_header_bytes(6)
    check("wav header RIFF/fmt/data", header[:4] == b"RIFF" and header[8:12] == b"WAVE"
          and header[12:16] == b"fmt " and header[36:40] == b"data")
    check("wav header sizes", len(header) == 44 and int.from_bytes(header[40:44], "little") == 12)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        a, b, book = td / "a.wav", td / "b.wav", td / "book.wav"
        _make_wav(a, [1, 2, 3])
        _make_wav(b, [4, 5, 6])
        with open(book, "wb") as out:
            out.write(Generator._wav_header_bytes(6))
            out.write(a.read_bytes()[44:])
            out.write(b.read_bytes()[44:])
        check("concatenation bytes exact",
              Generator._payload_bytes_equal(book, [a, b]) and
              book.read_bytes() == Generator._wav_header_bytes(6) + a.read_bytes()[44:] + b.read_bytes()[44:])
        check("concatenation rejects wrong payload",
              not Generator._payload_bytes_equal(book, [a, a]))

    # Child fallback must assemble children when the parent has no checkpoint.
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        parent = dict(_ca, eligible_clause_spans=[_span_a, _span_b])
        child_gen = object.__new__(Generator)
        child_gen.out_dir = td
        child_gen.validate = False
        child_gen.force = False
        child_gen.done = {}
        child_gen._records = {}
        for child in child_gen._child_chunks(parent):
            wav = td / (child["id"].replace(":", "-") + ".wav")
            _make_wav(wav, [1, 2])
            child_gen.done[child["id"]] = {
                "text_sha256": child["text_sha256"], "wav": wav.name,
                "wav_sha256": sha256_file(wav), "samples": 2,
                "terminal_silence_seconds": 0.0001,
            }
        child_gen.plan = {"chunks": [parent]}
        check("fallback assembly selects complete children",
              child_gen._children_complete(parent) and
              [c["id"] for c in child_gen._assembly_units(parent)] ==
              [c["id"] for c in child_gen._child_chunks(parent)])
        check("completed plan accepts child set",
              child_gen._plan_complete() and not child_gen._unit_done(parent))

    # A forced parent failure must invalidate its old choice before children win.
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        parent = dict(_ca, eligible_clause_spans=[_span_a, _span_b])
        force_gen = object.__new__(Generator)
        force_gen.out_dir = td
        force_gen.validate = True
        force_gen.force = True
        force_gen.done = {}
        force_gen._records = {}
        force_gen._forced_parents = set()
        force_gen.config = type("Cfg", (), {"asr_repo": _ar, "asr_revision": _av})()
        force_gen.plan = {"chunks": [parent]}
        force_gen._save_records = lambda: None
        force_gen._save_state = lambda: None
        old_wav = td / "old-parent.wav"
        _make_wav(old_wav, [1, 2])
        force_gen.done[parent["id"]] = {
            "text_sha256": parent["text_sha256"], "wav": old_wav.name,
            "wav_sha256": sha256_file(old_wav), "samples": 2,
            "terminal_silence_seconds": 0.0001,
        }
        force_gen._records[parent["id"]] = dict(
            _pass, chunk_id=parent["id"], wav_sha256=sha256_file(old_wav),
            expected_sha256=parent["text_sha256"])
        check("force selects parent for regeneration",
              [c["id"] for c in force_gen._pending_chunks()] == [parent["id"]])
        check("force invalidates old parent checkpoint and validation",
              parent["id"] not in force_gen.done and parent["id"] not in force_gen._records
              and old_wav.is_file())
        fresh = []
        for child in force_gen._child_chunks(parent):
            wav = td / (child["id"].replace(":", "-") + ".wav")
            _make_wav(wav, [3, 4])
            fresh_hash = sha256_file(wav)
            force_gen.done[child["id"]] = {
                "text_sha256": child["text_sha256"], "wav": wav.name,
                "wav_sha256": fresh_hash, "samples": 2,
                "terminal_silence_seconds": 0.0001,
            }
            force_gen._records[child["id"]] = dict(
                _pass, chunk_id=child["id"], wav_sha256=fresh_hash,
                expected_sha256=child["text_sha256"])
            fresh.append(child)
        check("fresh fallback children override stale parent",
              force_gen._children_complete(parent)
              and [c["id"] for c in force_gen._assembly_units(parent)] ==
              [c["id"] for c in fresh]
              and parent["id"] not in [c["id"] for c in force_gen._assembly_units(parent)])

    # Retry helper accepts the second deterministic child take.
    retry_gen = object.__new__(Generator)
    retry_gen.validate = True
    retry_gen._records = {}
    retry_gen._attempt_failures = {}
    retry_gen.out_dir = pathlib.Path("/tmp")
    retry_child = {"id": "ch01:p0003:s0000-0003:c0001", "text": "Child."}
    retry_attempts = []
    retry_records = [
        {"id": retry_child["id"], "attempt": 0, "tail_frame_peak": 0.24,
         "terminal_silence_seconds": 0.0, "generation_seconds": 0.0, "seconds": 1.0},
        {"id": retry_child["id"], "attempt": 1, "tail_frame_peak": 0.0089,
         "terminal_silence_seconds": 0.24, "generation_seconds": 0.0, "seconds": 1.0},
    ]
    retry_gen._generate = lambda chunk, model, attempt=0: (
        retry_attempts.append(attempt) or dict(retry_records[attempt]))
    retry_gen._validate_chunk = lambda chunk, record: {
        "verdict": "PASS", "reasons": [],
        "terminal": {"matched": True},
        "punctuation": {"boundaries": [], "summary": {"failed": 0}},
    }
    accepted, retry_failures = retry_gen._generate_with_retries(retry_child, object())
    check("truncated take retries and quiet tail passes",
          accepted["tail_frame_peak"] == 0.0089
          and retry_attempts == [0, 1]
          and retry_failures == ["attempt 0: speech reaches into the room-tone tail"]
          and retry_child["id"] in retry_gen._records)
    gate_gen = object.__new__(Generator)
    gate_gen.validate = False
    gate_gen._records = {}
    gate_gen._attempt_failures = {}
    gate_gen.out_dir = pathlib.Path("/tmp")
    gate_gen._generate = lambda chunk, model, attempt=0: {
        "id": chunk["id"], "tail_frame_peak": 0.24}
    rejected, gate_failures = gate_gen._generate_with_retries(retry_child, object())
    check("persistent truncation exhausts four takes",
          rejected is None and len(gate_failures) == 4
          and all(reason.endswith("speech reaches into the room-tone tail")
                  for reason in gate_failures)
          and retry_child["id"] not in gate_gen._records)
    pass_gen = object.__new__(Generator)
    pass_gen.validate = False
    pass_gen._records = {}
    pass_gen._attempt_failures = {}
    pass_gen.out_dir = pathlib.Path("/tmp")
    pass_gen._generate = lambda chunk, model, attempt=0: {
        "id": chunk["id"], "tail_frame_peak": 0.0089}
    positive, positive_failures = pass_gen._generate_with_retries(retry_child, object())
    check("room-tone tail passes", positive is not None and not positive_failures)
    check("sibilant-final words detected",
          qwenfix.SIBILANT_FINAL.search("collapsed") is not None
          and qwenfix.SIBILANT_FINAL.search("centuries") is not None
          and qwenfix.SIBILANT_FINAL.search("delhi") is None
          and qwenfix.SIBILANT_FINAL.search("planned") is None)
    sib_gen = object.__new__(Generator)
    sib_gen.validate = False
    sib_gen._records = {}
    sib_gen._attempt_failures = {}
    sib_gen.out_dir = pathlib.Path("/tmp")
    sib_gen._generate = lambda chunk, model, attempt=0: {
        "id": chunk["id"], "tail_frame_peak": 0.001,
        "final_sibilant_high_frac": 0.012}
    sib_rejected, sib_failures = sib_gen._generate_with_retries(retry_child, object())
    check("missing final sibilant exhausts four takes",
          sib_rejected is None and len(sib_failures) == 4
          and all("final sibilant missing" in reason for reason in sib_failures))
    fail_gen = object.__new__(Generator)
    fail_gen.validate = True
    fail_gen._records = {}
    fail_gen._attempt_failures = {}
    fail_gen.out_dir = pathlib.Path("/tmp")
    fail_attempts = []
    fail_gen._generate = lambda chunk, model, attempt=0: (
        fail_attempts.append(attempt) or dict(retry_records[1], attempt=attempt))
    fail_gen._validate_chunk = lambda chunk, record: {
        "verdict": "FAIL", "reasons": ["punctuation pause failed"]}
    failed_record, failed_reasons = fail_gen._generate_with_retries(retry_child, object())
    retry_error = fail_gen._retry_error(retry_child, failed_reasons)
    check("child retry failure names both reasons",
          failed_record is None and fail_attempts == [0, 1]
          and "every retry attempt failed" in str(retry_error)
          and "punctuation pause failed" in str(retry_error))

    failed = [name for name, ok in results if not ok]
    print(f"selfcheck: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


def _raises(exc, fn, *args) -> bool:
    try:
        fn(*args)
        return False
    except exc:
        return True


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="audiobook.runner", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck", help="run the no-model runner self-check")
    args = ap.parse_args(argv)
    return selfcheck()


if __name__ == "__main__":
    sys.exit(main())
