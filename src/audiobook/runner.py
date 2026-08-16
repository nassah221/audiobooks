"""Resumable Qwen3-TTS + Bragg-reference generation runner (frozen behavior).

Implements the frozen pilot invocation exactly (see scripts/qwen/
run_qwen_bragg_paragraph.py): one model load per run, sentence-by-sentence
non-streaming ``generate`` calls (``ref_audio``/``ref_text`` verbatim,
``lang_code="English"``, ``stream=False``, ``max_tokens=4096``, no seed, no
speed), 24k mono PCM16 atomic checkpoints, hashed resume (``state.json``
keyed by text/model/reference hashes), byte-exact chapter concatenation with
zero inserted samples, optional persistent mlx-whisper validation via
``audiobook.asr.AsrValidator``, and ETA estimates that report generation and
ASR costs separately.

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

from . import config, epub
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

# --- frozen model / generation contract --------------------------------------
MODEL_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
MODEL_REVISION = "a6eb4f68e4b056f1215157bb696209bc82a6db48"
SAMPLE_RATE = 24_000
PILOT_SENTENCE = (
    "The death of Tamerlane in fourteen oh five was a turning point in world history."
)
# Broad sanity gate: Qwen has no supported seed, and approved runs of this
# frozen sentence vary in duration while preserving the same generation contract.
PILOT_DURATION_BOUNDS = (3.5, 8.0)


def _invocation(cfg: Config) -> dict:
    """Generation knobs read from config. stream is fixed False; seed/speed
    are unsupported by Qwen3-TTS and deliberately not exposed."""
    return {
        "lang_code": cfg.language,
        "stream": False,
        "max_tokens": cfg.max_tokens,
        "seed": "unsupported (no seed parameter in Qwen3TTSModel.generate)",
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


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_json(path: pathlib.Path, obj) -> None:
    _atomic_write(path, json.dumps(obj, indent=1) + "\n")


# --- root / inputs -----------------------------------------------------------
def find_root(start=None) -> pathlib.Path:
    """Nearest ancestor of `start` (default cwd) containing audiobook.toml."""
    try:
        return config.find_root(start)
    except ConfigError as e:
        raise RunError(str(e)) from e


def verify_inputs(root, checksums=None) -> dict:
    """Presence + sha256 of every configured input; raises RunError with detail.

    `checksums` is a {relative_path: expected_sha256} map; when omitted it is
    derived from audiobook.toml under `root`.
    """
    root = pathlib.Path(root)
    if checksums is None:
        checksums = load_config(root).inputs
    facts = {}
    errors = []
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


def run_fingerprint(cfg: Config, ref_wav_sha: str, ref_text_sha: str) -> str:
    """Hash of everything that identifies a run: config inputs + generation knobs.

    Resume is only valid when this matches the state on disk; changing any
    input, model, language, max_tokens, or ASR setting (or the actual voice
    files) yields a different fingerprint and a clear error.
    """
    payload = {
        "book": {"path": cfg.book, "sha256": cfg.book_sha256},
        "voice": {"audio": cfg.audio, "transcript": cfg.transcript,
                  "audio_sha256": cfg.audio_sha256,
                  "transcript_sha256": cfg.transcript_sha256,
                  "wav_ref_sha256": ref_wav_sha,
                  "text_ref_sha256": ref_text_sha},
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision,
                  "language": cfg.language, "max_tokens": cfg.max_tokens},
        "asr": {"repo": cfg.asr_repo, "revision": cfg.asr_revision},
        "sample_rate": SAMPLE_RATE,
        "stream": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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


def build_plan(root, config=None, chapters=None, limit=None, resume_from=None) -> dict:
    """Full sentence plan from the configured book (pure stdlib extraction).

    Returns {"book", "chapters": [{id, title, sentences}], "total_sentences",
    "chunks": [{chapter, idx, id, text, text_sha256}]} in spine order.
    Raises ValueError for a non-book and RunError for bad selections.
    """
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
    if resume_from is not None:
        ids = [c["id"] for c in selected]
        if resume_from not in ids:
            raise RunError(
                f"resume_from chapter {resume_from!r} not in selection "
                f"(available: {', '.join(ids)})"
            )
        selected = selected[ids.index(resume_from):]
    chunks = []
    for c in selected:
        for i, s in enumerate(c["sentences"]):
            chunks.append({
                "chapter": c["id"],
                "idx": i,
                "id": f"{c['id']}:{i:04d}",
                "text": s,
                "text_sha256": sha256_text(s),
            })
    if limit is not None:
        if limit < 0:
            raise RunError(f"limit must be >= 0, got {limit}")
        chunks = chunks[:limit]
    return {
        "book": str(book),
        "chapters": [{"id": c["id"], "title": c["title"], "sentences": len(c["sentences"])}
                     for c in selected],
        "total_sentences": len(chunks),
        "chunks": chunks,
    }


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
        "total_sentences": plan["total_sentences"],
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
            ref_wav=root / cfg.audio, ref_text=(root / cfg.transcript).read_text(),
        )
        if report["benchmark"]["verdict"] != "PASS":
            report["verdict"] = "FAIL"
    return report


# --- generator ---------------------------------------------------------------
def _state_plan_matches(st_plan, plan) -> bool:
    """True iff stored plan identity equals the current plan.

    Identity = ordered selected chunk ids + their text hashes + chapter spec,
    so a changed --limit or --chapters (or edited book text) yields False and
    forces resume invalidation instead of silently skipping/reordering.
    """
    if not isinstance(st_plan, dict):
        return False
    return (st_plan.get("chunk_ids") == [c["id"] for c in plan["chunks"]]
            and st_plan.get("text_hashes") == [c["text_sha256"] for c in plan["chunks"]]
            and st_plan.get("chapters") == [
                {"id": c["id"], "title": c["title"], "sentences": c["sentences"]}
                for c in plan["chapters"]])


class Generator:
    """Resumable full-book generation with atomic PCM16 checkpoints.

    One model load per run; every planned chunk is generated with the frozen
    invocation, written atomically (tmp + rename), recorded in chunks.jsonl
    (append-only log) and state.json (atomic, keyed by text/model/reference
    hashes), optionally ASR-validated through the persistent AsrValidator,
    and finally concatenated byte-exactly into book.wav when the plan is
    complete.
    """

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
        self.fingerprint = run_fingerprint(
            self.config, self.ref_wav_sha, self.ref_text_sha
        )
        self.plan = build_plan(
            self.root, self.config, chapters=chapters, limit=limit
        )
        self.start_idx = 0
        if resume_from is not None:
            ids = [c["id"] for c in self.plan["chapters"]]
            if resume_from not in ids:
                raise RunError(
                    f"resume_from chapter {resume_from!r} not in plan "
                    f"(available: {', '.join(ids)})"
                )
            self.start_idx = ids.index(resume_from)
        self.chunks_dir = self.out_dir / "chunks"
        self.state_path = self.out_dir / STATE_REL
        self.jsonl_path = self.out_dir / CHUNKS_JSONL_REL
        self._model = None
        self.load_seconds = None
        self._validator_obj = None
        self._records = self._load_records()
        self._load_state()

    # -- state ----------------------------------------------------------------
    def _state_chapters(self) -> list:
        return [{"id": c["id"], "title": c["title"], "sentences": c["sentences"]}
                for c in self.plan["chapters"]]

    def _plan_identity(self) -> dict:
        return {
            "chapters": self._state_chapters(),
            "chunk_ids": [c["id"] for c in self.plan["chunks"]],
            "text_hashes": [c["text_sha256"] for c in self.plan["chunks"]],
            "total_sentences": self.plan["total_sentences"],
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
        self._save_state()  # initial checkpoint (also records plan/fingerprint)

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

    # -- chunk processing -----------------------------------------------------
    @staticmethod
    def _chunk_status(chunk: dict, done: dict, force: bool) -> str:
        """Pure hash/record check: "done" | "pending". Disk is checked by
        the caller (file presence + wav hash)."""
        if force:
            return "pending"
        d = done.get(chunk["id"])
        if not d or d.get("text_sha256") != chunk["text_sha256"]:
            return "pending"
        return "done"

    def _is_done(self, chunk: dict) -> bool:
        if self._chunk_status(chunk, self.done, self.force) != "done":
            return False
        d = self.done[chunk["id"]]
        wav = self.out_dir / d["wav"]
        if not wav.is_file():
            return False
        if d.get("wav_sha256") != sha256_file(wav):
            return False  # corrupted on disk: regenerate
        return True

    def _pending_chunks(self):
        chapter_idx = {c["id"]: i for i, c in enumerate(self.plan["chapters"])}
        pending = []
        for chunk in self.plan["chunks"]:
            if chapter_idx[chunk["chapter"]] < self.start_idx:
                continue
            if not self._is_done(chunk):
                pending.append(chunk)
        return pending

    def _validator(self):
        if self._validator_obj is None:
            from . import asr

            self._validator_obj = asr.AsrValidator(
                model_repo=self.config.asr_repo, revision=self.config.asr_revision,
                cache_path=self.out_dir / ASR_CACHE_REL,
            )
        return self._validator_obj

    def _generate(self, chunk: dict, model) -> dict:
        inv = _invocation(self.config)
        t0 = time.perf_counter()
        results = list(model.generate(
            text=chunk["text"],
            ref_audio=str(self.ref_wav),
            ref_text=self.ref_text,
            lang_code=inv["lang_code"],
            stream=inv["stream"],
            max_tokens=inv["max_tokens"],
        ))
        gen_seconds = time.perf_counter() - t0
        if len(results) != 1:
            raise RunError(f"{chunk['id']}: generation results = {len(results)}, expected exactly 1")
        r = results[0]
        if int(r.sample_rate) != SAMPLE_RATE:
            raise RunError(f"{chunk['id']}: result sample_rate {r.sample_rate} != {SAMPLE_RATE}")
        import numpy as np
        import soundfile as sf

        rel = f"chunks/{chunk['chapter']}/{chunk['chapter']}-{chunk['idx']:04d}.wav"
        wav = self.out_dir / rel
        wav.parent.mkdir(parents=True, exist_ok=True)
        # .tmp.wav suffix: soundfile infers format from the extension; the
        # rename below is the atomic publish step.
        tmp = wav.with_name(wav.name + ".tmp.wav")
        try:
            sf.write(str(tmp), np.asarray(r.audio), SAMPLE_RATE, subtype="PCM_16")
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
            "id": chunk["id"],
            "chapter": chunk["chapter"],
            "idx": chunk["idx"],
            "text": chunk["text"],
            "text_sha256": chunk["text_sha256"],
            "wav": rel,
            "wav_sha256": facts["sha256"],
            "sample_rate": facts["sample_rate"],
            "channels": facts["channels"],
            "subtype": facts["subtype"],
            "samples": facts["samples"],
            "seconds": facts["seconds"],
            "generation_seconds": round(gen_seconds, 4),
            "model": {"repo": self.config.model_repo, "revision": self.config.model_revision},
            "reference_wav_sha256": self.ref_wav_sha,
            "reference_text_sha256": self.ref_text_sha,
            "started_unix": time.time(),
        }

    def _record_chunk(self, record: dict):
        self.done[record["id"]] = {
            "text_sha256": record["text_sha256"],
            "wav": record["wav"],
            "wav_sha256": record["wav_sha256"],
            "samples": record["samples"],
            "seconds": record["seconds"],
            "generation_seconds": record["generation_seconds"],
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        self._save_state()

    def _validate_chunk(self, chunk: dict, record: dict):
        """Non-raising per-chunk validation; verdict recorded, never aborts."""
        from . import asr

        v = self._validator()
        vrec = v.validate_many([{
            "wav": str(self.out_dir / record["wav"]),
            "expected_text": chunk["text"],
            "chunk_id": chunk["id"],
        }])[0]
        self._records[chunk["id"]] = vrec
        return vrec

    def _all_done(self) -> bool:
        return all(self._chunk_status(c, self.done, False) == "done" and
                   (self.out_dir / self.done[c["id"]]["wav"]).is_file()
                   for c in self.plan["chunks"])

    # -- release gate ---------------------------------------------------------
    @staticmethod
    def _record_ok(rec, text_sha256: str, wav_sha256: str,
                   asr_repo: str, asr_revision: str) -> bool:
        """True iff a stored ASR record is a *current* PASS for this exact
        chunk: verdict PASS, matching wav/text hashes, and the current ASR
        model config. A stale record (different wav, text, or ASR model) can
        never satisfy the release gate."""
        if not rec or rec.get("verdict") != "PASS":
            return False
        if rec.get("wav_sha256") != wav_sha256:
            return False
        if rec.get("expected_sha256") != text_sha256:
            return False
        cfg = rec.get("asr") or {}
        return (cfg.get("model_repo") == asr_repo
                and cfg.get("model_revision") == asr_revision)

    def _release_gate(self) -> dict:
        """Release only when every chunk has a current PASS validation record."""
        blocked = []
        for chunk in self.plan["chunks"]:
            if not self._is_done(chunk):
                blocked.append(chunk["id"])
                continue
            if not self._record_ok(
                    self._records.get(chunk["id"]), chunk["text_sha256"],
                    self.done[chunk["id"]]["wav_sha256"],
                    self.config.asr_repo, self.config.asr_revision):
                blocked.append(chunk["id"])
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
        """True iff book.wav payload bytes == concatenation of source payloads
        (streamed in lockstep, no giant array; chunk boundaries follow the
        source reads)."""
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

    def _concatenate(self) -> dict:
        """Byte-exact concatenation of per-sentence PCM16 payloads in plan
        order (zero inserted samples) into book.wav + book.json."""
        import soundfile as sf

        wavs = []
        total_frames = 0
        for chunk in self.plan["chunks"]:
            d = self.done.get(chunk["id"])
            if not d:
                raise RunError(f"cannot concatenate: {chunk['id']} not generated")
            wav = self.out_dir / d["wav"]
            if not wav.is_file():
                raise RunError(f"cannot concatenate: {chunk['id']} wav missing ({wav})")
            total_frames += d["samples"]
            wavs.append(wav)

        book_wav = self.out_dir / BOOK_WAV_REL
        tmp = book_wav.with_name(book_wav.name + ".tmp")
        with open(tmp, "wb") as out:
            out.write(self._wav_header_bytes(total_frames))
            for wav in wavs:
                with open(wav, "rb") as f:
                    f.seek(44)
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        out.write(b)
        os.replace(tmp, book_wav)

        errors = []
        if not self._payload_bytes_equal(book_wav, wavs):
            errors.append("book.wav payload != exact concatenation of chunk payloads")
        info = sf.info(str(book_wav))
        if int(info.samplerate) != SAMPLE_RATE or int(info.channels) != 1 \
                or info.subtype != "PCM_16" or int(info.frames) != total_frames:
            errors.append(
                f"book.wav header mismatch: sr={info.samplerate} ch={info.channels} "
                f"subtype={info.subtype} frames={info.frames} != {total_frames}"
            )

        chapter_ranges = {}
        sentence_offsets = {}
        pos = 0
        for chunk in self.plan["chunks"]:
            n = self.done[chunk["id"]]["samples"]
            sentence_offsets[chunk["id"]] = {
                "start_sample": pos, "end_sample": pos + n,
                "seconds": round(n / SAMPLE_RATE, 4),
            }
            cr = chapter_ranges.setdefault(chunk["chapter"], {"start_sample": pos, "samples": 0})
            cr["samples"] += n
            pos += n
        if pos != total_frames:
            errors.append(f"offset sum {pos} != total frames {total_frames}")
        for cid, cr in chapter_ranges.items():
            cr["end_sample"] = cr["start_sample"] + cr["samples"]
            cr["seconds"] = round(cr["samples"] / SAMPLE_RATE, 4)
            cr["title"] = next(
                c["title"] for c in self.plan["chapters"] if c["id"] == cid
            )

        manifest = {
            "book": BOOK_WAV_REL,
            "method": "byte-exact concatenation of per-sentence PCM16 payloads, zero inserted samples",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "subtype": "PCM_16",
            "samples": total_frames,
            "seconds": round(total_frames / SAMPLE_RATE, 4),
            "bytes": book_wav.stat().st_size,
            "sha256": sha256_file(book_wav),
            "chapters": chapter_ranges,
            "sentences": sentence_offsets,
            "generated_unix": time.time(),
            "verdict": "FAIL" if errors else "PASS",
            "errors": errors,
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
        gen_cum = 0.0
        audio_cum = 0.0
        for i, chunk in enumerate(pending, 1):
            record = self._generate(chunk, self._model)
            self._record_chunk(record)
            gen_cum += record["generation_seconds"]
            audio_cum += record["seconds"]
            line = (f"  {chunk['id']}  gen {record['generation_seconds']:5.2f}s "
                    f"audio {record['seconds']:6.2f}s  ({i}/{total})")
            if self.validate:
                vrec = self._validate_chunk(chunk, record)
                verdict = vrec["verdict"]
                line += f"  [{verdict}]"
                if verdict != "PASS":
                    line += f" {vrec['reasons'][0]}"
            print(line)
            if i % 25 == 0 or i == total:
                elapsed = gen_cum
                remaining = total - i
                if remaining and elapsed:
                    print(f"  ... {remaining} left, ~{elapsed / i * remaining / 60:.1f} min (generation only)")
        try:
            self._save_records()
        except Exception as e:  # records are best-effort; state/chunks are the durable log
            print(f"  warning: could not write validation records: {e}", file=sys.stderr)
        summary = {
            "plan": {"chapters": len(self.plan["chapters"]),
                     "sentences": len(self.plan["chunks"])},
            "generated": len(pending),
            "done_total": sum(1 for c in self.plan["chunks"]
                              if self._chunk_status(c, self.done, False) == "done"),
            "generation_seconds": round(gen_cum, 2),
            "audio_seconds": round(audio_cum, 2),
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds is not None else None,
        }
        if self.plan["chunks"] and self._all_done():
            gate = self._release_gate()
            if gate["release"]:
                concat = self._concatenate()
                print(f"  concatenated {len(concat['chapters'])} chapters -> "
                      f"{self.out_dir / BOOK_WAV_REL} ({concat['seconds']:.1f}s audio)")
                summary["book"] = concat
            else:
                self._remove_book_artifacts()
                print(f"  book.wav not built: {len(gate['blocked'])} chunk(s) blocked "
                      f"by the validation gate; re-run `audiobook validate` or regenerate")
                summary["book"] = None
                summary["blocked"] = gate["blocked"]
        elif self.plan["chunks"]:
            print("  book.wav not built: plan incomplete; re-run to resume")
        return summary


# --- validate ----------------------------------------------------------------
def validate_generated(root, out_dir, *, chapters=None, limit=None, config=None) -> dict:
    """Validate all generated chunks through the persistent AsrValidator.

    Requires matching state (fingerprint + plan) and the book for expected
    text; skips ungenerated chunks. Records are written atomically to
    <out>/validation/records.json.
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
    specs = []
    for chunk in plan["chunks"]:
        d = st.get("done", {}).get(chunk["id"])
        if not d:
            continue  # only generated chunks
        wav = out_dir / d["wav"]
        if not wav.is_file():
            continue
        specs.append({"wav": str(wav), "expected_text": chunk["text"], "chunk_id": chunk["id"]})
    if not specs:
        raise RunError("no generated chunks to validate")
    v = asr.AsrValidator(model_repo=cfg.asr_repo, revision=cfg.asr_revision,
                         cache_path=out_dir / ASR_CACHE_REL)
    records = v.validate_many(specs)
    _atomic_write_json(out_dir / RECORDS_REL, {
        "asr": {"model_repo": cfg.asr_repo, "model_revision": cfg.asr_revision},
        "records": records,
    })
    failed = [r for r in records if r["verdict"] != "PASS"]
    cached = sum(1 for r in records if r.get("cache_hit"))
    result = {
        "chunks": len(records),
        "passed": len(records) - len(failed),
        "failed": len(failed),
        "cached": cached,
        "failures": [{"chunk_id": r.get("chunk_id"), "reasons": r["reasons"]} for r in failed],
        "asr": v.stats(),
        "records": str(out_dir / RECORDS_REL),
    }
    # Assembly: reuse the Generator to get the byte-exact concatenation when
    # every planned chunk was validated and none failed. Requires the state to
    # match this run (already checked above), so resuming the generator loads
    # the same done set. Loads no model.
    if not failed and len(records) == len(plan["chunks"]):
        g = Generator(root, out_dir, config=cfg, chapters=chapters, limit=limit)
        if g._release_gate()["release"]:
            book = g._concatenate()
            result["book"] = {
                "wav": BOOK_WAV_REL,
                "seconds": book["seconds"],
                "chapters": len(book["chapters"]),
            }
    return result


# --- eta ---------------------------------------------------------------------
def estimate(root, out_dir, config=None, plan=None) -> dict:
    """Projected generation + ASR wall time from the measured pilot benchmark
    and the extraction sentence counts. Generation and ASR costs are reported
    separately."""
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
        "sentences": len(plan["chunks"]),
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
        "note": "estimate only (Qwen3-TTS has no seed; measured on the frozen pilot "
                "sentence, extrapolated by word/token counts)",
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

    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")

    print("audiobook.runner selfcheck")
    check("sha256_text deterministic", sha256_text("abc") == sha256_text("abc")
          and sha256_text("abc") != sha256_text("abd"))

    _root = pathlib.Path("/tmp/no-such-root")
    _base = Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                   "repo", "rev", "English", 4096, "a-repo", "a-rev")
    fp1 = run_fingerprint(_base, "w", "t")
    fp2 = run_fingerprint(_base, "w", "t")
    fp3 = run_fingerprint(_base, "w", "t2")
    check("fingerprint deterministic", fp1 == fp2)
    check("fingerprint sensitive to reference", fp1 != fp3)
    check("fingerprint sensitive to max_tokens",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 8192, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to model",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo2", "rev", "English", 4096, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to asr",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, "a-repo2", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to book hash",
          fp1 != run_fingerprint(Config(_root, "b.epub", "2" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to voice path",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "other.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, "a-repo", "a-rev"),
                                 "w", "t"))

    # config validation (no model, no book)
    _GOOD = """[book]\npath = "books/b.epub"\nsha256 = "%s"\n[voice]\naudio = "v.wav"\naudio_sha256 = "%s"\ntranscript = "v.txt"\ntranscript_sha256 = "%s"\n[model]\nrepo = "r"\nrevision = "rev"\nlanguage = "English"\nmax_tokens = 4096\n[asr]\nrepo = "a"\nrevision = "arev"\n""" % ("1" * 64, "2" * 64, "3" * 64)
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

    st_plan = {
        "chapters": [{"id": "ch01", "title": "T", "sentences": 2}],
        "chunk_ids": ["ch01:0000", "ch01:0001"],
        "text_hashes": ["a", "b"],
        "total_sentences": 2,
    }
    cur_plan = {
        "chapters": [{"id": "ch01", "title": "T", "sentences": 2}],
        "chunks": [{"id": "ch01:0000", "text_sha256": "a"},
                   {"id": "ch01:0001", "text_sha256": "b"}],
    }
    check("state plan identity matches", _state_plan_matches(st_plan, cur_plan))
    check("state plan drift (limit) detected", not _state_plan_matches(
        {"chapters": st_plan["chapters"], "chunk_ids": ["ch01:0000"],
         "text_hashes": ["a"], "total_sentences": 1}, cur_plan))
    check("state plan drift (text) detected", not _state_plan_matches(
        {"chapters": st_plan["chapters"], "chunk_ids": st_plan["chunk_ids"],
         "text_hashes": ["a", "x"], "total_sentences": 2}, cur_plan))
    check("state plan rejects non-plan", not _state_plan_matches(None, cur_plan))

    # release-gate predicate (no model, no book)
    from . import asr as _asr
    _ar, _av = _asr.DEFAULT_MODEL_REPO, _asr.DEFAULT_MODEL_REVISION
    _cur_cfg = {"model_repo": _ar, "model_revision": _av}
    _pass = {"verdict": "PASS", "wav_sha256": "W", "expected_sha256": "E",
             "asr": dict(_cur_cfg)}
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
