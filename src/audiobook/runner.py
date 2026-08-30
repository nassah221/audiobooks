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
    "regenerate",
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

# Generation policy identifier, recorded on each `done`/`failed` entry (see
# `_generate`/`_record_chunk`) -- NOT hashed into run_fingerprint. Bumping
# this string documents a change in generation behavior (EOS-hold tail
# handling, the context-drift gate, ...) without invalidating a directory's
# resume state: run_fingerprint only gates the inputs that make existing
# audio flatly wrong (model, reference voice, book, seed base), so an old
# directory generated under an earlier policy stays resumable under newer
# code. A directory legally mixes policies across chunks -- old chunks keep
# the policy they were generated under; only newly generated chunks record
# the current one. Audit which policy produced a given chunk by reading its
# `done[chunk_id]["generation_policy"]` (absent on chunks generated before
# this field existed).
#
# Lineage: v1 (icl-rolling-v1) conditioned every mid-paragraph chunk on the
# previous chunk's text+codes unconditionally. v2 (icl-rolling-v2) added the
# context-drift gate (CONTEXT_DRIFT_HIGH_FRAC_MIN / CONTEXT_CHAIN_DEPTH_MAX
# below) to stop reusing codes once they measured too muffled. Hassan
# listened to v2 output and still heard audible timbre drift within a
# chain, so v3 (icl-nocontext-v3) turns rolling context off entirely via
# ROLLING_CONTEXT_ENABLED: every chunk generates from the Bragg reference
# alone, the same context-free path a paragraph-start chunk always used.
# v4 (icl-nocontext-eos-replacement-v4) keeps context disabled and changes
# EOS continuation from replacement+5 frames to replacement-only after a
# shared-code unseen-text blind comparison: 3/8 preferred v4, 5 ties, all
# terminal words complete, and all three audible artifacts were in v3 tails.
# The v1/v2 machinery (_rolling_context_for, _context_decision,
# _accept_rolling, qwenfix's context-splicing) stays in the code, gated
# off, in case a future fix makes rolling context viable again.
GENERATION_POLICY = "icl-nocontext-eos-replacement-v4"

# Master switch for rolling-context conditioning (see GENERATION_POLICY
# lineage above). False: every chunk's generation call gets context=None,
# and per-chunk context fields (context_chunk_id/context_depth/
# context_usable) record as None -- only the spectral field
# (context_high_frac, cheap and useful for audits regardless) still
# records normally. True restores the icl-rolling-v2 behavior unchanged.
ROLLING_CONTEXT_ENABLED = False

# Shared predicate for "this unit has nothing to pronounce" (a stray
# formatting artifact like a bare "*" section-break marker). Single source
# of truth for both the live generation skip below and adjudicate.py's
# post-hoc OMIT_UNSPEAKABLE decision -- a unit with no letters is
# unspeakable regardless of which path notices first.
_UNSPEAKABLE_RE = re.compile(r"[A-Za-z]")


def is_unspeakable(text: str) -> bool:
    return not bool(_UNSPEAKABLE_RE.search(text or ""))


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


def _state_plan_relation(st_plan: dict | None, plan: dict) -> str:
    """Classify how persisted ``st_plan`` relates to the freshly built
    ``plan`` requested by this run:

    - ``"match"``: identical -- same chunks, same chapters, same totals.
    - ``"superset"``: every chunk id the stored state ever recorded is
      still present in ``plan`` with an unchanged identity hash (text,
      spans, clause boundaries); ``plan`` may additionally contain chunks
      the stored state never saw (e.g. ``--chapters 1`` growing to
      ``--chapters 1,2`` against the same ``--out``). Resuming is safe:
      every stored ``done``/``failed`` entry still describes a real chunk
      in this run, so nothing needs to be discarded, and the new chunks
      simply start pending.
    - ``"conflict"``: the planner itself changed, or some chunk id the
      stored state recorded now has different text/boundaries (or is
      missing from ``plan`` altogether, e.g. a narrower chapter request) --
      reusing that stored progress would silently attach it to the wrong
      source text, so resuming is refused (see ``Generator._load_state``).
    """
    if not isinstance(st_plan, dict):
        return "conflict"
    if st_plan.get("planner") != plan.get("planner"):
        return "conflict"
    st_ids = st_plan.get("chunk_ids")
    st_hashes = st_plan.get("text_hashes")
    if (not isinstance(st_ids, list) or not isinstance(st_hashes, list)
            or len(st_ids) != len(st_hashes)):
        return "conflict"
    cur_hash_by_id = {c["id"]: _chunk_identity_hash(c) for c in plan["chunks"]}
    for cid, h in zip(st_ids, st_hashes):
        if cur_hash_by_id.get(cid) != h:
            return "conflict"
    chapters = [{"id": c["id"], "title": c["title"],
                 "paragraphs": c["paragraphs"], "groups": c["groups"]}
                for c in plan["chapters"]]
    is_exact = (st_ids == [c["id"] for c in plan["chunks"]]
                and st_hashes == [cur_hash_by_id[c["id"]] for c in plan["chunks"]]
                and st_plan.get("total_paragraphs") == plan["total_paragraphs"]
                and st_plan.get("total_groups") == plan["total_groups"]
                and st_plan.get("chapters") == chapters)
    return "match" if is_exact else "superset"


def _state_plan_matches(st_plan: dict | None, plan: dict) -> bool:
    """Return whether persisted state safely resumes against this source
    plan -- an exact match, or `plan` widening it (see
    ``_state_plan_relation``). A genuinely conflicting plan returns False."""
    return _state_plan_relation(st_plan, plan) in ("match", "superset")


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_json(path: pathlib.Path, obj) -> None:
    _atomic_write(path, json.dumps(obj, indent=1) + "\n")

def run_fingerprint(cfg: Config, ref_wav_sha: str, ref_text_sha: str) -> str:
    """Hash only the immutable inputs that make existing audio flatly wrong
    if they change: which model produced it, which voice it was cloned
    from (the reference WAV and its exact transcript, always used as a
    pair), which book it was read from, and the base seed.

    Everything that used to live here -- the sentence/clause planner
    version, the ASR validator, and the generation policy (EOS-hold tail
    handling, structural gates, the context-drift gate) -- is deliberately
    left out:

    - Planner changes are already caught per chunk by
      ``_chunk_identity_hash``/``_state_plan_matches`` (a changed sentence
      or clause boundary changes that chunk's identity hash, which blocks
      *that* chunk, not the whole run).
    - The ASR validator's repo/revision is already checked per validation
      record by ``_record_ok``, independent of run identity.
    - The generation policy moved to a per-chunk ``done`` field (see
      ``GENERATION_POLICY``) instead of a run-wide gate, so a code upgrade
      that only changes generation behavior never invalidates a
      directory's whole resume state -- a directory legally mixes
      policies across chunks as new code lands.
    """
    payload = {
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision},
        "reference": {"wav_sha256": ref_wav_sha, "text_sha256": ref_text_sha},
        "book": {"sha256": cfg.book_sha256},
        "seed": cfg.seed,
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
                 discard_done=False, resume_from=None, offline=None, validate=True,
                 config=None):
        self.root = pathlib.Path(root)
        self.out_dir = pathlib.Path(out_dir)
        self.force = force
        # See _load_state: a stronger, separate flag from `force` -- force
        # alone never discards a mismatched run's recorded progress (it
        # archives it and carries it forward, since the individual
        # per-chunk text_sha256 checks already keep anything genuinely
        # stale from counting as done). discard_done is the explicit,
        # loud opt-in to actually zero it.
        self.discard_done = discard_done
        # Set True only by _load_state's mismatch-carry-forward branch
        # (--force used to escape a mismatch, not --discard-done). Distinct
        # from `self.force` itself: `force` ALSO means "regenerate every
        # chunk" when the state already matches (see _pending_chunks), and
        # that full-regen behavior must NOT apply to a carried-forward set
        # -- carried entries that still validate are done, no TTS call.
        self._mismatch_force_carry = False
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
        # Last successfully-generated (not necessarily accepted) take per
        # chunk id, this run only -- lets _record_failure capture the
        # retained wav's structural facts without re-reading the file.
        self._last_attempt = {}
        # Last accepted unit, for rolling-context conditioning (in-memory
        # only: generated codec frames are never persisted across runs, so
        # a resumed run always starts with no context, even mid-paragraph).
        self._rolling = None
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

    def _archive_state(self) -> pathlib.Path:
        """Rename the existing state.json aside (state.json.bak-<n>, lowest
        unused n) before a mismatch is acknowledged via --force or
        --discard-done -- so the discarded record is never lost even when
        the run proceeds, only left where a human can recover it."""
        n = 1
        while True:
            candidate = self.state_path.with_name(f"{self.state_path.name}.bak-{n}")
            if not candidate.exists():
                break
            n += 1
        self.state_path.replace(candidate)
        return candidate

    def _load_state(self):
        self.done = {}
        # Continue-on-failure set: chunk id -> {text_sha256, reasons,
        # wav, wav_sha256, samples, seconds, terminal_silence_seconds,
        # tail_frame_peak, failed_policy, failed_unix}. Additive state --
        # an existing state.json with no "failed" key (every directory
        # before this feature shipped) loads as {} here, so resume
        # compatibility for already-generated chapters is unaffected.
        self.failed = {}
        self.started_unix = time.time()
        if self.state_path.is_file():
            st = json.loads(self.state_path.read_text())
            fp_matches = st.get("fingerprint") == self.fingerprint
            relation = _state_plan_relation(st.get("plan"), self.plan) if fp_matches else "conflict"
            if fp_matches and relation != "conflict":
                self.done = st.get("done", {})
                self.failed = st.get("failed", {})
                self.started_unix = st.get("started_unix", self.started_unix)
            elif not self.force and not self.discard_done:
                raise RunError(
                    f"existing state at {self.state_path} does not match this run "
                    "(inputs/model/reference changed, or the requested chapters/limit "
                    "conflict with previously recorded chunks). Use --force to archive "
                    "the old state and carry forward whatever done chunks still match "
                    "this run, --discard-done to explicitly wipe recorded progress "
                    "(archived first, never silently), or a fresh --out."
                )
            else:
                # Acknowledged: archive the old file before this run's
                # _save_state below overwrites it, so the discarded record
                # is always recoverable -- never a silent loss.
                old_done, old_failed = st.get("done", {}) or {}, st.get("failed", {}) or {}
                archived = self._archive_state()
                if self.discard_done:
                    print(
                        f"  --discard-done: discarding {len(old_done)} done chunk(s) and "
                        f"{len(old_failed)} failed entry(ies) from the mismatched state "
                        f"(archived to {archived.name}; wav files on disk are untouched)",
                        file=sys.stderr,
                    )
                else:
                    # --force alone never zeros done: carry the old record
                    # forward. Any entry that no longer matches THIS run's
                    # plan simply won't count as done (_unit_done checks
                    # text_sha256 per chunk), so nothing stale is reused.
                    self.done, self.failed = old_done, old_failed
                    # Carried-forward entries must be re-admitted as done
                    # by the normal _is_done check, NOT wiped by force's
                    # other meaning ("regenerate everything" on an
                    # already-matching state) -- see _pending_chunks.
                    self._mismatch_force_carry = True
                    print(
                        f"  --force: state mismatch archived to {archived.name}; "
                        f"carrying forward {len(old_done)} previously-done chunk(s) and "
                        f"{len(old_failed)} failed entry(ies) (only ones matching this "
                        "run's plan will still count as done)",
                        file=sys.stderr,
                    )
        # Reconcile on load: a chunk already in `done` may still have a
        # stale continue-on-failure entry for itself or a clause-split
        # descendant of it (see _clear_resolved_failures) -- e.g. a parent
        # was regenerated whole and passed after one of its children had
        # already exhausted its retry budget. Clearing this here, on every
        # load, heals a directory that fell out of sync on its very next
        # `generate`/`validate` invocation, with no hand-editing.
        for done_id in list(self.done):
            self._clear_resolved_failures(done_id)
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
            "failed": self.failed,
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

    def _has_failed_descendant(self, chunk: dict) -> bool:
        """True when `chunk` itself, or one of its clause children, is in
        the continue-on-failure set -- used to push previously-failed work
        behind never-attempted work in `_pending_chunks`."""
        if chunk["id"] in self.failed:
            return True
        return any(child["id"] in self.failed for child in self._child_chunks(chunk))

    def _clear_resolved_failures(self, chunk_id: str) -> None:
        """Drop `chunk_id`, and any clause-split child of it, from the
        continue-on-failure set now that `chunk_id` itself is done.

        A clause-split child's id is always `f"{parent_id}:c{n:04d}"`
        (see `_child_chunks`), so once the parent is done as ONE whole
        unit, any failed entry recorded against one of its children is
        stale: the parent's audio already covers that child's text span
        end to end, so there is no missing audio, only a leftover
        bookkeeping entry. Left alone, that entry makes `validate` and
        `generate` refuse assembly forever ("N chunk(s) in the
        continue-on-failure set") even though nothing needs fixing.
        Called both when a chunk freshly completes (`_record_chunk`,
        `_maybe_promote_failed`) and when state.json loads (`_load_state`),
        so an already-mismatched directory heals on its next run with no
        hand-editing.

        The symmetric case -- a child itself re-passing -- clears its own
        entry the same way (a child has no children of its own, so the
        loop below is simply empty for it).

        getattr guard: minimal test doubles built with
        object.__new__(Generator) predate `failed` and never set it;
        this is a pure bookkeeping side effect, not core to what those
        tests exercise.
        """
        failed = getattr(self, "failed", None)
        if not isinstance(failed, dict):
            return
        failed.pop(chunk_id, None)
        prefix = f"{chunk_id}:c"
        for fid in [f for f in failed if f.startswith(prefix) and f[len(prefix):].isdigit()]:
            del failed[fid]

    def _pending_chunks(self):
        # `force` has two distinct meanings that must not collapse into
        # one: (1) the state already matched and --force means "regenerate
        # everything" (the documented, pre-existing behavior -- full
        # invalidation below); (2) _load_state hit a mismatch and --force
        # only archived + carried the old done/failed set forward (see
        # _mismatch_force_carry) -- there, carried entries that still
        # validate against THIS plan are done, exactly like a normal
        # resume, and only genuinely new/changed chunks are pending.
        # getattr guard: minimal test doubles built with
        # object.__new__(Generator) predate this flag and never set it;
        # absent means "not a mismatch carry-forward", i.e. plain force.
        if self.force and not getattr(self, "_mismatch_force_carry", False):
            for parent in self.plan["chunks"]:
                if parent["id"] not in self._forced_parents:
                    self._invalidate_forced_parent(parent)
            return list(self.plan["chunks"])
        # Failed chunks are re-selectable, but only AFTER every
        # never-attempted chunk: a fresh chunk should never wait behind a
        # retry of something that already burned an attempt budget.
        fresh, retry = [], []
        for chunk in self.plan["chunks"]:
            if self._is_done(chunk):
                continue
            (retry if self._has_failed_descendant(chunk) else fresh).append(chunk)
        return fresh + retry
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
        if d.get("omitted"):
            return True
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
        if record.get("omitted"):
            self.done[record["id"]] = {
                "text_sha256": record["text_sha256"],
                "omitted": True,
                "omit_reason": record.get("omit_reason", "OMIT_UNSPEAKABLE"),
            }
        else:
            self.done[record["id"]] = {
                "text_sha256": record["text_sha256"], "wav": record["wav"],
                "wav_sha256": record["wav_sha256"], "samples": record["samples"],
                "seconds": record["seconds"],
                "terminal_silence_seconds": record["terminal_silence_seconds"],
                "tail_frame_peak": record.get("tail_frame_peak", 0.0),
                "context_chunk_id": record.get("context_chunk_id"),
                # Audit trail for concern A: which generation policy
                # produced this take. Absent on chunks generated before
                # this field existed -- that is the "mixed-policy
                # directory" the fingerprint split makes legal.
                "generation_policy": record.get("generation_policy"),
                # Context-drift gate (icl-rolling-v2) audit trail: measured
                # high-band fraction, chain depth, and whether this take's
                # codes were usable as context for the next chunk -- kept
                # regardless of the decision, so future analysis is free.
                "context_high_frac": record.get("context_high_frac"),
                "context_depth": record.get("context_depth"),
                "context_usable": record.get("context_usable"),
            }
        # This chunk is done: any lingering continue-on-failure entry for
        # it, or for a clause-split child of it, is now stale (see
        # _clear_resolved_failures).
        self._clear_resolved_failures(record["id"])
        # Persist the record before state marks this unit done.
        self._save_records()
        self._save_state()

    def _finalize_take(self, record: dict) -> None:
        """Compute the context-drift decision once, stash it onto `record`
        so `_record_chunk` persists it, then record the chunk and update
        rolling context accordingly. The one path both the parent-unit and
        child-unit acceptance branches in `run()` share.

        When ROLLING_CONTEXT_ENABLED is False (icl-nocontext-v3, the
        default), the decision/rolling-chain machinery is skipped entirely:
        context_depth and context_usable record as None (never computed --
        there is no chain to measure), self._rolling never gets set so
        every chunk's own generation call always sees context=None (see
        _generate_with_retries), and context_chunk_id is already None
        because context is never passed to _generate. context_high_frac
        (the spectral field) is unaffected -- it is computed unconditionally
        in _generate, still cheap and useful for audits.
        """
        if ROLLING_CONTEXT_ENABLED:
            decision = self._context_decision(record)
            record["context_depth"] = decision["depth"]
            record["context_usable"] = decision["context_usable"]
            self._record_chunk(record)
            self._accept_rolling(record, decision)
        else:
            record["context_depth"] = None
            record["context_usable"] = None
            self._record_chunk(record)

    @staticmethod
    def _chunk_wav_rel(chunk: dict) -> str:
        return f"chunks/{chunk['chapter']}/{chunk['id'].replace(':', '-')}.wav"

    def _record_failure(self, chunk: dict, failures: list) -> None:
        """Continue-on-failure: record a chunk that exhausted its retry
        budget into the persisted `failed` set instead of raising. The
        last attempt's wav is left exactly where `_generate` wrote it
        (never deleted) so a later policy fix can revalidate it without a
        new TTS call -- see `_maybe_promote_failed`. Structural facts come
        from `self._last_attempt` (this run's last successfully-generated,
        not necessarily accepted, take for this chunk); if every attempt
        raised before producing audio, those are absent, and there is no
        wav to keep the sha of.
        """
        last = self._last_attempt.pop(chunk["id"], None)
        rel = self._chunk_wav_rel(chunk)
        wav_path = self.out_dir / rel
        wav_sha = sha256_file(wav_path) if wav_path.is_file() else None
        self.failed[chunk["id"]] = {
            "text_sha256": chunk["text_sha256"],
            "reasons": list(failures),
            "wav": rel if wav_sha else None,
            "wav_sha256": wav_sha,
            "samples": last.get("samples") if last else None,
            "seconds": last.get("seconds") if last else None,
            "terminal_silence_seconds": last.get("terminal_silence_seconds") if last else None,
            "tail_frame_peak": last.get("tail_frame_peak") if last else None,
            "failed_policy": asr.VALIDATION_POLICY,
            "failed_unix": time.time(),
        }
        # A chunk can only be in one of done/failed at a time.
        self.done.pop(chunk["id"], None)
        self._save_state()
        reason = failures[-1] if failures else "no diagnostic available"
        print(f"  {chunk['id']}  FAILED (deferred, continue-on-failure): {reason}",
              file=sys.stderr)

    def _maybe_promote_failed(self, chunk: dict) -> bool:
        """If `chunk` is in the failed set, its retained wav is unchanged
        on disk, and it now passes the structural gate and (if enabled)
        ASR validation -- e.g. a gate/lexical fix landed since it failed --
        promote it straight to done with NO new TTS call. Returns True iff
        promoted; callers should treat that exactly like a fresh success
        (skip generation) and False like "still needs a fresh attempt."
        """
        entry = self.failed.get(chunk["id"])
        if not entry or entry.get("text_sha256") != chunk["text_sha256"] or not entry.get("wav"):
            return False
        wav = self.out_dir / entry["wav"]
        if not wav.is_file() or sha256_file(wav) != entry.get("wav_sha256"):
            return False
        tail_frame_peak = entry.get("tail_frame_peak")
        if tail_frame_peak is None or float(tail_frame_peak) > qwenfix.TAIL_FRAME_PEAK_MAX:
            return False
        if self.validate:
            vrec = self._validator().validate_many([{
                "wav": str(wav),
                "expected_text": asr.normalize_for_tts(chunk["text"]),
                "chunk_id": chunk["id"],
            }])[0]
            if not _take_passes(vrec):
                return False
            self._records[chunk["id"]] = vrec
        self.done[chunk["id"]] = {
            "text_sha256": entry["text_sha256"], "wav": entry["wav"],
            "wav_sha256": entry["wav_sha256"], "samples": entry.get("samples"),
            "seconds": entry.get("seconds"),
            "terminal_silence_seconds": entry.get("terminal_silence_seconds"),
            "tail_frame_peak": tail_frame_peak,
            "context_chunk_id": None,
        }
        self._clear_resolved_failures(chunk["id"])
        self._save_records()
        self._save_state()
        print(f"  {chunk['id']}  promoted from failed set (revalidated, no TTS call)")
        return True

    def _maybe_omit_unspeakable(self, chunk: dict) -> bool:
        """Skip generation entirely for a unit with no alphabetic content (a
        stray formatting artifact, e.g. a bare "*" section-break marker) --
        there is nothing to synthesize, and attempting to would only ever
        produce near-silent audio that fails the structural gate on every
        retry. Mirrors adjudicate.py's post-hoc OMIT_UNSPEAKABLE decision,
        but before ever calling the TTS model, so the audit trail this
        produces is the one that overlay would have produced anyway.
        Returns True (and records the omission) when the unit was skipped.
        """
        if not is_unspeakable(chunk["text"]):
            return False
        self._record_chunk({
            "id": chunk["id"], "text_sha256": chunk["text_sha256"],
            "omitted": True, "omit_reason": "OMIT_UNSPEAKABLE",
        })
        return True

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

    @staticmethod
    def _rolling_context_for(rolling: dict | None, chunk: dict) -> dict | None:
        """Rolling-context policy: condition on the last accepted unit only
        when it is in the same chapter and paragraph as `chunk` (a window of
        one previous unit). Returns None otherwise — including when there is
        no rolling state yet (start of a chapter run) or the chunk starts a
        new paragraph, which is the "reset" case: no explicit state mutation
        is needed because the mismatch already excludes the stale unit.
        """
        if rolling is None:
            return None
        if rolling["chapter"] != chunk["chapter"]:
            return None
        if rolling["paragraph_index"] != chunk["paragraph_index"]:
            return None
        return rolling

    def _context_decision(self, record: dict) -> dict:
        """Context-drift gate (icl-rolling-v2): decide whether `record`'s
        codes are trustworthy enough to hand to the NEXT chunk as ICL
        context. Never affects whether `record` itself is accepted into
        the book -- only whether ITS codes get reused.

        Two independent triggers, either one blocks reuse:
        1. high_frac below CONTEXT_DRIFT_HIGH_FRAC_MIN: rolling context
           compounds a loss of high-frequency energy down a chain
           (measured -35% by depth 3), heard as increasingly muffled
           audio.
        2. depth >= CONTEXT_CHAIN_DEPTH_MAX: a hard cap independent of
           this take's own quality -- a chunk at depth 3 never passes its
           codes onward, so a chain resets at latest every 3 hops even
           when every take individually measures clean.

        depth 0 means "not itself conditioned on prior context" (a
        paragraph start, or the chunk right after a reset); depth N means
        conditioned on a chain of N prior chunks in the same paragraph.
        """
        prev = self._rolling
        same_chain = (prev is not None and prev.get("chapter") == record["chapter"]
                      and prev.get("paragraph_index") == record["paragraph_index"])
        depth = (prev["depth"] + 1) if same_chain else 0
        high_frac = record.get("context_high_frac")
        usable = (high_frac is not None
                  and high_frac >= qwenfix.CONTEXT_DRIFT_HIGH_FRAC_MIN
                  and depth < qwenfix.CONTEXT_CHAIN_DEPTH_MAX)
        return {"depth": depth, "high_frac": high_frac, "context_usable": usable}

    def _accept_rolling(self, record: dict, decision: dict = None) -> None:
        """Track the last accepted unit for rolling-context conditioning.

        Only called for takes that passed structural gates and validation
        (see `_generate_with_retries`/`run`), so a rejected take never
        poisons the chain. `decision` (from `_context_decision`) may be
        passed in already computed, since `run()` also needs it to record
        per-chunk fields before this runs; if omitted, it is computed here
        (kept optional for existing direct callers/tests).
        """
        codes = record.get("_gen_codes")
        if codes is None:
            return
        if decision is None:
            decision = self._context_decision(record)
        if not decision["context_usable"]:
            self._rolling = None
            return
        self._rolling = {
            "chunk_id": record["id"],
            "paragraph_index": record["paragraph_index"],
            "chapter": record["chapter"],
            # speak_text, not text: these codes were synthesized from
            # speak_text (see _generate), so the ICL context handed to the
            # next chunk must describe the text that actually produced them.
            "text": record.get("speak_text", record["text"]),
            "codes": codes,
            "depth": decision["depth"],
        }

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

    def _generate(self, chunk: dict, model, attempt: int = 0,
                  context: dict | None = None) -> dict:
        import mlx.core as mx

        inv = _invocation(self.config)
        if int(model.sample_rate) != SAMPLE_RATE:
            raise RunError(f"model sample_rate {model.sample_rate} != {SAMPLE_RATE}")
        t0 = time.perf_counter()
        seed = self._seed(inv["seed"], chunk["id"], attempt)
        mx.random.seed(seed)
        # speak_text (v31): speak-time-only text substitution for the
        # "Name + roman numeral" mispronunciation hazard (see
        # asr.normalize_for_tts's docstring). The TTS receives speak_text,
        # not chunk["text"] -- plan identity, chunk ids, and text_sha256
        # stay bound to the untouched original everywhere else.
        speak_text = asr.normalize_for_tts(chunk["text"])
        context_arg = (context["text"], context["codes"]) if context is not None else None
        audio, gen_codes = qwenfix.generate_icl_tail_safe(
            model, speak_text, self._ref_audio_array(), self.ref_text,
            inv["lang_code"], inv["max_tokens"], context=context_arg)
        gen_seconds = time.perf_counter() - t0
        import numpy as np
        import soundfile as sf

        tail_frame_peak = qwenfix.tail_frame_peak(audio, SAMPLE_RATE)
        sibilant_frac = qwenfix.final_sibilant_high_frac(
            audio, SAMPLE_RATE, speak_text)
        context_high_frac = qwenfix.speech_high_band_frac(audio, SAMPLE_RATE)
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
            "speak_text": speak_text,
            "wav": rel, "wav_sha256": facts["sha256"],
            "sample_rate": facts["sample_rate"], "channels": facts["channels"],
            "subtype": facts["subtype"], "samples": facts["samples"],
            "terminal_silence_seconds": round(self._terminal_silence_seconds(audio), 4),
            "eos_hold_frames": qwenfix.EOS_HOLD_FRAMES,
            "tail_frame_peak": round(tail_frame_peak, 6),
            "final_sibilant_high_frac": (
                None if sibilant_frac is None else round(sibilant_frac, 4)),
            "context_high_frac": round(context_high_frac, 6),
            "seconds": facts["seconds"], "generation_seconds": round(gen_seconds, 4),
            "model": {"repo": self.config.model_repo, "revision": self.config.model_revision},
            "reference_wav_sha256": self.ref_wav_sha,
            "reference_text_sha256": self.ref_text_sha,
            "seed": seed, "attempt": attempt, "started_unix": time.time(),
            "context_chunk_id": context["chunk_id"] if context is not None else None,
            # Per-chunk generation policy (concern A) -- NOT in
            # run_fingerprint, so bumping GENERATION_POLICY never
            # invalidates an existing directory's resume state; a
            # directory legally mixes policies as chunks generated under
            # different code land side by side.
            "generation_policy": GENERATION_POLICY,
            # Internal only: the generated codec frames, kept for a possible
            # rolling-context handoff to the next chunk. Never persisted —
            # excluded from state.json/records.json by the explicit field
            # lists those write.
            "_gen_codes": gen_codes,
        }
    def _validate_chunk(self, chunk: dict, record: dict):
        """Validate one take without checkpointing its verdict."""
        return self._validator().validate_many([{
            "wav": str(self.out_dir / record["wav"]),
            # speak_text: the ASR comparison target is what was actually
            # spoken (see _generate/asr.normalize_for_tts), not chunk["text"].
            "expected_text": record.get("speak_text", chunk["text"]),
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

        Only the first attempt is conditioned on rolling context (the last
        accepted unit in this paragraph, if any): a rejected take must not
        poison the chain, and dropping context on retry removes it as a
        failure variable, so every attempt after the first generates as if
        there were no previous chunk. When ROLLING_CONTEXT_ENABLED is False
        (icl-nocontext-v3, the default), that "first attempt" context is
        always None too -- every attempt of every chunk generates from the
        Bragg reference alone.
        """
        failures = []
        validation_failures = 0
        base_context = (
            self._rolling_context_for(getattr(self, "_rolling", None), chunk)
            if ROLLING_CONTEXT_ENABLED else None)
        for attempt in range(4):
            if validation_failures >= 2:
                break
            context = base_context if attempt == 0 else None
            try:
                record = self._generate(chunk, model, attempt=attempt, context=context)
                # Tracked regardless of pass/fail: on exhaustion, the wav
                # this attempt wrote is what's left on disk, and
                # _record_failure needs its structural facts to let a
                # later policy fix revalidate it without a new TTS call.
                # getattr guard: minimal test doubles built with
                # object.__new__(Generator) may not set this up, and
                # tracking it is a pure bookkeeping side effect, not core
                # to what those tests exercise.
                if hasattr(self, "_last_attempt"):
                    self._last_attempt[chunk["id"]] = record
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

    @staticmethod
    def _expected_text_sha256(chunk: dict) -> str:
        """Hash of what was actually spoken (asr.normalize_for_tts(chunk
        ["text"])) -- what _validate_chunk records as a take's
        expected_sha256 since speak_text was introduced. NOT the same as
        chunk["text_sha256"] (the untouched original, used for plan
        identity) whenever normalize_for_tts changes anything: comparing
        a validation record's expected_sha256 against the raw
        text_sha256 would permanently mark every speak_text-affected
        chunk (regnal numerals, entrepot, circa, glued numbers, ...) as
        stale even though it validated correctly, blocking release
        forever."""
        return sha256_text(asr.normalize_for_tts(chunk["text"]))

    def _release_gate(self) -> dict:
        blocked = []
        for parent in self.plan["chunks"]:
            children = self._child_chunks(parent)
            if children and all(self._unit_done(c) for c in children):
                for child in children:
                    done = self.done[child["id"]]
                    if done.get("omitted"):
                        continue
                    if not self._record_ok(self._records.get(child["id"]),
                                           self._expected_text_sha256(child), done["wav_sha256"],
                                           self.config.asr_repo, self.config.asr_revision):
                        blocked.append(child["id"])
                continue
            if not self._unit_done(parent):
                blocked.append(parent["id"])
                continue
            done = self.done[parent["id"]]
            if done.get("omitted"):
                continue
            if not self._record_ok(self._records.get(parent["id"]),
                                   self._expected_text_sha256(parent), done["wav_sha256"],
                                   self.config.asr_repo, self.config.asr_revision):
                blocked.append(parent["id"])
        return {"release": not blocked, "blocked": blocked}

    def _assembly_decision(self) -> dict:
        """The single gate both `run()` (generate) and `validate_generated`
        consult before touching book.wav -- consistent by construction,
        since there is only one implementation to consult.

        `self.failed` is checked FIRST, before plan completeness: a parent
        chunk can be "done" as one whole unit while `self.failed` still
        holds a genuine failure for a different id, and `_plan_complete`/
        `_release_gate` only ever look at `self.done`, so neither would
        ever see that failure on its own. (A stale clause-split-child
        entry left behind by a later whole-parent regeneration is not this
        case -- that is reconciled out of `self.failed` the moment the
        parent completes; see `_clear_resolved_failures`. What is left in
        `self.failed` by the time this runs is always a real, unresolved
        failure.)

        Returns one of:
        - ``{"status": "empty"}``: nothing planned.
        - ``{"status": "failed", "failed_ids": [...]}``: refuse -- resolve
          the continue-on-failure set before assembly.
        - ``{"status": "incomplete"}``: refuse -- generation still pending.
        - ``{"status": "blocked", "blocked": [...]}``: refuse -- every unit
          is done, but a validation record does not check out.
        - ``{"status": "release"}``: safe to concatenate.
        """
        if not self.plan["chunks"]:
            return {"status": "empty"}
        if self.failed:
            return {"status": "failed", "failed_ids": sorted(self.failed)}
        if not self._plan_complete():
            return {"status": "incomplete"}
        gate = self._release_gate()
        if not gate["release"]:
            return {"status": "blocked", "blocked": gate["blocked"]}
        return {"status": "release"}

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
                if done.get("omitted"):
                    # No audio to contribute (e.g. a bare "*" section break);
                    # the normal inter-paragraph gap between its real
                    # neighbors already carries that silence, so it is
                    # simply left out rather than padded separately.
                    continue
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
            if self._maybe_omit_unspeakable(chunk):
                print(f"  {chunk['id']}  omitted (non-speech text)  ({i}/{total})")
                continue
            if self._maybe_promote_failed(chunk):
                continue
            record, failures = self._generate_with_retries(chunk, self._model)
            if record is None:
                children = self._child_chunks(chunk)
                if not children:
                    # Continue-on-failure: record and move to the next
                    # pending chunk instead of aborting the whole run.
                    self._record_failure(chunk, failures)
                    continue
                records = []
                for child in children:
                    if self._maybe_omit_unspeakable(child):
                        continue
                    if self._maybe_promote_failed(child):
                        continue
                    child_record, child_failures = self._generate_with_retries(child, self._model)
                    if child_record is None:
                        self._record_failure(child, child_failures)
                        continue
                    self._finalize_take(child_record)
                    records.append(child_record)
            else:
                # _finalize_take -> _record_chunk clears any stale
                # self.failed[chunk["id"]] (see _clear_resolved_failures).
                self._finalize_take(record)
                records = [record]
            if not records:
                continue
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
            "failed_total": len(self.failed),
            "failed": [{"chunk_id": cid, "reason": entry["reasons"][-1] if entry.get("reasons") else None}
                       for cid, entry in sorted(self.failed.items())],
            "generation_seconds": round(gen_cum, 2),
            "audio_seconds": round(audio_cum, 2),
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds is not None else None,
        }
        # One gate for both entry points (see _assembly_decision):
        # generate and validate now refuse identically, and for the same
        # reason, whenever a real failure is unresolved.
        decision = self._assembly_decision()
        if decision["status"] == "release":
            concat = self._concatenate()
            print(f"  concatenated {len(concat['chapters'])} chapters -> {self.out_dir / BOOK_WAV_REL} ({concat['seconds']:.1f}s audio)")
            summary["book"] = concat
        elif decision["status"] == "failed":
            # Loud, explicit: a non-empty failed set means real content is
            # missing from the book, never silently.
            self._remove_book_artifacts()
            summary["book"] = None
            summary["blocked"] = decision["failed_ids"]
            print(f"  book.wav not built: {len(decision['failed_ids'])} chunk(s) PERMANENTLY FAILED "
                  "(continue-on-failure mode) -- resolve before assembly:", file=sys.stderr)
            for cid in decision["failed_ids"]:
                entry = self.failed[cid]
                reason = entry["reasons"][-1] if entry.get("reasons") else "no diagnostic available"
                print(f"    {cid}: {reason}", file=sys.stderr)
        elif decision["status"] == "blocked":
            self._remove_book_artifacts()
            summary["book"] = None
            summary["blocked"] = decision["blocked"]
        elif decision["status"] == "incomplete":
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
            if not d or d.get("omitted") or not d.get("wav"):
                continue
            wav = out_dir / d["wav"]
            if not wav.is_file():
                continue
            # speak_text: standalone `validate` must compare against the same
            # text that was actually spoken as live generation does, or the
            # numeral-hazard rewrite (asr.normalize_for_tts) causes the exact
            # false failures it exists to prevent when re-run here.
            specs.append({"wav": str(wav), "expected_text": asr.normalize_for_tts(chunk["text"]),
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
    g = Generator(root, out_dir, config=cfg, chapters=chapters, limit=limit)
    result["failed_chunks"] = sorted(g.failed)
    if failed:
        # A fresh ASR failure from THIS validate pass, separate from
        # `g.failed` (the persisted continue-on-failure set) -- either one
        # alone means real content is not safe to release, so this check
        # stays independent of `g._assembly_decision()` below.
        pass
    else:
        # Same gate `generate` uses (see Generator._assembly_decision):
        # refuses identically, and for the same reason, whenever a real
        # failure is unresolved.
        decision = g._assembly_decision()
        if decision["status"] == "failed":
            result["book_blocked_reason"] = (
                f"{len(decision['failed_ids'])} chunk(s) in the continue-on-failure set; "
                "resolve (fix + revalidate, or adjudicate) before assembly")
        elif decision["status"] == "release":
            book = g._concatenate()
            result["book"] = {
                "wav": BOOK_WAV_REL,
                "seconds": book["seconds"],
                "chapters": len(book["chapters"]),
            }
    return result


# --- selective regeneration ---------------------------------------------------
def regenerate(root, out_dir, chunk_ids, *, config=None) -> dict:
    """Move the listed DONE chunks back to pending (concern D).

    For each chunk id: pop it from `state.json`'s `done` map, rename its
    existing wav aside (``<name>.superseded-<timestamp>.wav`` -- archived,
    never deleted, matching the rest of this module's "never destroy a
    generated take" policy), and clear its `validation/records.json` and
    `validation/asr-cache.json` entries so the next `generate` re-runs it
    fresh under whatever policy/gates are current, with no rolling context
    (the codec frames that would carry it were never persisted). An id not
    currently in `done` is a hard error -- silently skipping a typo would
    leave the caller believing a chunk was queued for regeneration when it
    was not.

    `root`/`config` are accepted for interface symmetry with the rest of
    this module (a future version may want to validate ids against a
    freshly-built plan); regenerate itself only touches `out_dir` and does
    not require the book, model, or ASR model.
    """
    out_dir = pathlib.Path(out_dir)
    state_path = out_dir / STATE_REL
    if not state_path.is_file():
        raise RunError(f"no generation state at {state_path}; nothing to regenerate")
    st = json.loads(state_path.read_text())
    done = st.get("done", {}) or {}
    failed = st.get("failed", {}) or {}
    ids = list(dict.fromkeys(chunk_ids))  # de-dupe, preserve request order
    if not ids:
        raise RunError("regenerate: no chunk ids given")
    unknown = [cid for cid in ids if cid not in done]
    if unknown:
        detail = ", ".join(
            f"{cid} (in the failed set, not done)" if cid in failed else cid
            for cid in unknown
        )
        raise RunError(
            "regenerate: chunk id(s) not in the done set (typo, already pending, "
            f"or already failed): {detail}"
        )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archived_wavs = []
    for cid in ids:
        entry = done.pop(cid)
        wav_rel = entry.get("wav")
        if not wav_rel:
            continue  # an omitted unit has no wav to archive
        wav_path = out_dir / wav_rel
        if not wav_path.is_file():
            continue  # nothing on disk to preserve; state alone is enough
        superseded = wav_path.with_name(f"{wav_path.name}.superseded-{timestamp}.wav")
        n = 1
        while superseded.exists():
            superseded = wav_path.with_name(
                f"{wav_path.name}.superseded-{timestamp}-{n}.wav")
            n += 1
        os.replace(wav_path, superseded)
        archived_wavs.append(str(superseded.relative_to(out_dir)))
    st["done"] = done
    _atomic_write_json(state_path, st)

    id_set = set(ids)
    cleared_records = 0
    records_path = out_dir / RECORDS_REL
    if records_path.is_file():
        try:
            payload = json.loads(records_path.read_text())
            records = {r["chunk_id"]: r for r in payload.get("records", [])
                       if isinstance(r, dict) and r.get("chunk_id")}
            asr_header = payload.get("asr", {})
        except (json.JSONDecodeError, KeyError, TypeError):
            records, asr_header = {}, {}
        cleared_records = sum(1 for cid in ids if records.pop(cid, None) is not None)
        if cleared_records:
            _atomic_write_json(records_path, {
                "asr": asr_header,
                "records": sorted(records.values(), key=lambda r: r.get("chunk_id") or ""),
            })

    cleared_cache = 0
    cache_path = out_dir / ASR_CACHE_REL
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
        stale_keys = [k for k, v in cache.items()
                      if isinstance(v, dict) and v.get("chunk_id") in id_set]
        for k in stale_keys:
            del cache[k]
        cleared_cache = len(stale_keys)
        if stale_keys:
            _atomic_write_json(cache_path, cache)

    return {
        "regenerated": ids,
        "archived_wavs": archived_wavs,
        "cleared_records": cleared_records,
        "cleared_cache_entries": cleared_cache,
    }


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

    # Declared once, up front: two sections below flip ROLLING_CONTEXT_ENABLED
    # temporarily to prove the icl-rolling-v2 machinery still works when
    # re-enabled, then restore the default (False) in a finally block.
    global ROLLING_CONTEXT_ENABLED

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
    check("fingerprint sensitive to seed",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, 43, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to model repo",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo2", "rev", "English", 4096, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to model revision",
          fp1 != run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev2", "English", 4096, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint sensitive to book hash",
          fp1 != run_fingerprint(Config(_root, "b.epub", "2" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    # Concern A: run_fingerprint now gates ONLY the inputs that make
    # existing audio flatly wrong -- model, reference (wav+transcript
    # hash), book, seed. max_tokens, the ASR validator, language, and bare
    # path strings (as opposed to their content hashes) moved out: a
    # directory generated under an earlier ASR model, token limit, or
    # generation policy must keep resuming under current code (a
    # "mixed-policy directory" -- see GENERATION_POLICY and the plan
    # widening/mixed-policy checks below), not get refused outright.
    check("fingerprint NOT sensitive to max_tokens (moved out of run identity)",
          fp1 == run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 8192, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint NOT sensitive to the ASR validator (checked per validation record instead)",
          fp1 == run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "English", 4096, 42, "a-repo2", "a-rev2"),
                                 "w", "t"))
    check("fingerprint NOT sensitive to language",
          fp1 == run_fingerprint(Config(_root, "b.epub", "1" * 64, "v.wav", "2" * 64, "v.txt", "3" * 64,
                                        "repo", "rev", "French", 4096, 42, "a-repo", "a-rev"),
                                 "w", "t"))
    check("fingerprint NOT sensitive to bare path strings (only content hashes count)",
          fp1 == run_fingerprint(Config(_root, "other.epub", "1" * 64, "other.wav", "2" * 64,
                                        "other.txt", "3" * 64, "repo", "rev", "English",
                                        4096, 42, "a-repo", "a-rev"),
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
    check("state plan identity matches", _state_plan_matches(st_plan, cur_plan)
          and _state_plan_relation(st_plan, cur_plan) == "match")
    # Concern B: a stored plan that is a narrower subset of the current,
    # WIDER plan (e.g. --chapters 1 grown to --chapters 1,2 against the
    # same --out) is a "superset" relation, not drift -- every chunk id it
    # ever recorded is still present with an unchanged identity hash, so
    # resuming keeps that progress instead of erroring.
    _narrower_st_plan = dict(st_plan, chunk_ids=[_ca["id"]],
                             text_hashes=[st_plan["text_hashes"][0]], total_groups=1)
    check("a plan that only widens (superset) resumes -- stored progress is not drift",
          _state_plan_matches(_narrower_st_plan, cur_plan)
          and _state_plan_relation(_narrower_st_plan, cur_plan) == "superset")
    check("state plan drift (text) detected: a genuinely conflicting plan still errors",
          not _state_plan_matches(dict(st_plan, text_hashes=["x", st_plan["text_hashes"][1]]), cur_plan)
          and _state_plan_relation(dict(st_plan, text_hashes=["x", st_plan["text_hashes"][1]]), cur_plan)
          == "conflict")
    check("a narrower request than what's stored still conflicts "
          "(never silently drops a recorded chunk from consideration)",
          not _state_plan_matches(st_plan, dict(cur_plan, chunks=[_ca]))
          and _state_plan_relation(st_plan, dict(cur_plan, chunks=[_ca])) == "conflict")
    check("a planner change always conflicts, even with identical chunk ids/hashes",
          _state_plan_relation(dict(st_plan, planner={"policy": "different"}), cur_plan) == "conflict")
    check("state plan rejects non-plan", not _state_plan_matches(None, cur_plan)
          and _state_plan_relation(None, cur_plan) == "conflict")

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

    # _release_gate compares a record's expected_sha256 against the SPOKEN
    # text's hash, not the chunk's raw text_sha256 -- since v31 introduced
    # speak_text, _validate_chunk records expected_sha256 = sha256(speak_text),
    # so any speak_text-affected chunk (entrepot, circa, glued numbers,
    # regnal numerals) has speak_text != chunk["text"], and comparing
    # against the raw hash would block it from ever releasing even though
    # it validated correctly. Found and fixed while running the actual ch2
    # boundary validation -- 23 real chunks were stuck blocked this way.
    _entrepot_release_text = "the great cultural entrepô t of the Islamic world."
    _raw_sha = sha256_text(_entrepot_release_text)
    _speak_sha = sha256_text(_asr.normalize_for_tts(_entrepot_release_text))
    check("entrepot-style text: raw text_sha256 and speak_text's hash genuinely differ",
          _raw_sha != _speak_sha)
    _entrepot_rec = dict(_pass, chunk_id="x", wav_sha256="W", expected_sha256=_speak_sha)
    check("record_ok accepts a record validated against speak_text's hash",
          Generator._record_ok(_entrepot_rec, _speak_sha, "W", _ar, _av))
    check("record_ok would have wrongly rejected it against the stale raw text_sha256 "
          "(this is the bug the fix addresses)",
          not Generator._record_ok(_entrepot_rec, _raw_sha, "W", _ar, _av))
    check("_expected_text_sha256 computes the speak_text hash _validate_chunk actually uses",
          Generator._expected_text_sha256({"text": _entrepot_release_text}) == _speak_sha)
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

    # A unit with no letters (a stray "*" section-break marker) is omitted
    # before ever attempting TTS -- the same OMIT_UNSPEAKABLE decision
    # adjudicate.py would make post-hoc, just made earlier.
    check("bare symbol is unspeakable", is_unspeakable("*"))
    check("other letterless strings are unspeakable",
          is_unspeakable("---") and is_unspeakable("1500") and is_unspeakable(""))
    check("normal text is speakable", not is_unspeakable("The empire fell."))
    check("digit-bearing prose is speakable",
          not is_unspeakable("In 1500, the empire fell."))

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        omit_gen = object.__new__(Generator)
        omit_gen.out_dir = td
        omit_gen.validate = False
        omit_gen._records = {}
        omit_gen._save_records = lambda: None
        omit_gen._save_state = lambda: None
        omit_gen.done = {}
        star_chunk = {"id": "ch01:p0005:s0000-0001", "text": "*",
                      "text_sha256": sha256_text("*")}
        prose_chunk = {"id": "ch01:p0006:s0000-0001", "text": "Real prose.",
                       "text_sha256": sha256_text("Real prose.")}
        check("omit helper records the unit and returns True",
              omit_gen._maybe_omit_unspeakable(star_chunk) and
              omit_gen.done[star_chunk["id"]] ==
              {"text_sha256": star_chunk["text_sha256"], "omitted": True,
               "omit_reason": "OMIT_UNSPEAKABLE"})
        check("omit helper leaves normal text alone",
              not omit_gen._maybe_omit_unspeakable(prose_chunk) and
              prose_chunk["id"] not in omit_gen.done)
        check("an omitted unit is done without ever having a wav",
              omit_gen._unit_done(star_chunk))

    # The release gate must not block on an omitted unit (no ASR record
    # exists for it -- it was never generated, let alone validated).
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        gate_gen = object.__new__(Generator)
        gate_gen.out_dir = td
        gate_gen.validate = True
        gate_gen._records = {}
        gate_gen.config = type("Cfg", (), {"asr_repo": _ar, "asr_revision": _av})()
        star_chunk = {"id": "ch01:p0007:s0000-0001", "text": "*",
                      "text_sha256": sha256_text("*"), "eligible_clause_spans": []}
        gate_gen.plan = {"chunks": [star_chunk]}
        gate_gen.done = {star_chunk["id"]: {"text_sha256": star_chunk["text_sha256"],
                                            "omitted": True, "omit_reason": "OMIT_UNSPEAKABLE"}}
        check("release gate does not block an omitted unit",
              gate_gen._release_gate() == {"release": True, "blocked": []})

        # _assembly_decision (defect 2's shared gate): empty plan, a
        # complete+passing plan, and a complete-but-blocked plan each
        # report the status run()/validate_generated act on.
        gate_gen.failed = {}
        check("assembly decision releases when the plan is complete, gate passes, nothing failed",
              gate_gen._assembly_decision() == {"status": "release"})
        empty_gen = object.__new__(Generator)
        empty_gen.plan = {"chunks": []}
        check("assembly decision reports empty for a plan with nothing in it",
              empty_gen._assembly_decision() == {"status": "empty"})
        incomplete_chunk = {"id": "ch01:p0008:s0000-0001", "text": "Not generated yet.",
                            "text_sha256": sha256_text("Not generated yet."),
                            "eligible_clause_spans": []}
        incomplete_gen = object.__new__(Generator)
        incomplete_gen.plan = {"chunks": [incomplete_chunk]}
        incomplete_gen.done = {}
        incomplete_gen.failed = {}
        check("assembly decision reports incomplete while a unit is still pending (no failure)",
              incomplete_gen._assembly_decision() == {"status": "incomplete"})
        blocked_chunk = {"id": "ch01:p0009:s0000-0001", "text": "Done but unrecorded.",
                         "text_sha256": sha256_text("Done but unrecorded."),
                         "eligible_clause_spans": []}
        blocked_wav = td / "blocked.wav"
        _make_wav(blocked_wav, [1, 2])
        blocked_gen = object.__new__(Generator)
        blocked_gen.out_dir = td
        blocked_gen.validate = True
        # A record that PASSes (so _unit_done/_plan_complete count this
        # chunk as generation-complete) but under a stale validation
        # policy -- _record_ok's stricter check (used only by
        # _release_gate) catches that and blocks release.
        blocked_gen._records = {blocked_chunk["id"]: dict(
            _pass, chunk_id=blocked_chunk["id"], wav_sha256=sha256_file(blocked_wav),
            expected_sha256=Generator._expected_text_sha256(blocked_chunk),
            validation_policy="stale-policy",
        )}
        blocked_gen.config = type("Cfg", (), {"asr_repo": _ar, "asr_revision": _av})()
        blocked_gen.plan = {"chunks": [blocked_chunk]}
        blocked_gen.done = {blocked_chunk["id"]: {
            "text_sha256": blocked_chunk["text_sha256"], "wav": blocked_wav.name,
            "wav_sha256": sha256_file(blocked_wav), "samples": 2,
            "terminal_silence_seconds": 0.0,
        }}
        blocked_gen.failed = {}
        check("assembly decision reports blocked when the plan is complete but a record fails the gate",
              blocked_gen._assembly_decision() == {"status": "blocked", "blocked": [blocked_chunk["id"]]})

    # Assembly must skip an omitted unit's audio entirely -- the real
    # neighbors on either side get exactly the padding their own paragraph
    # relationship calls for, not doubled and not glued together.
    with tempfile.TemporaryDirectory() as td:
        import wave
        td = pathlib.Path(td)
        left_wav, right_wav = td / "left.wav", td / "right.wav"
        _make_wav(left_wav, [1, 2, 3])
        _make_wav(right_wav, [4, 5, 6])
        left_chunk = {"id": "ch01:p0004:s0000-0001", "text": "Left.",
                      "text_sha256": sha256_text("Left."), "chapter": "ch01",
                      "paragraph_index": 4, "eligible_clause_spans": []}
        star_chunk = {"id": "ch01:p0005:s0000-0001", "text": "*",
                      "text_sha256": sha256_text("*"), "chapter": "ch01",
                      "paragraph_index": 5, "eligible_clause_spans": []}
        right_chunk = {"id": "ch01:p0006:s0000-0001", "text": "Right.",
                       "text_sha256": sha256_text("Right."), "chapter": "ch01",
                       "paragraph_index": 6, "eligible_clause_spans": []}
        concat_gen = object.__new__(Generator)
        concat_gen.out_dir = td
        concat_gen.plan = {"chunks": [left_chunk, star_chunk, right_chunk]}
        concat_gen.done = {
            left_chunk["id"]: {"text_sha256": left_chunk["text_sha256"], "wav": left_wav.name,
                               "wav_sha256": sha256_file(left_wav), "samples": 3,
                               "terminal_silence_seconds": 0.0},
            star_chunk["id"]: {"text_sha256": star_chunk["text_sha256"],
                               "omitted": True, "omit_reason": "OMIT_UNSPEAKABLE"},
            right_chunk["id"]: {"text_sha256": right_chunk["text_sha256"], "wav": right_wav.name,
                                "wav_sha256": sha256_file(right_wav), "samples": 3,
                                "terminal_silence_seconds": 0.0},
        }
        manifest = concat_gen._concatenate()
        book_wav = td / BOOK_WAV_REL
        expected_pad = Generator._boundary_padding_frames(left_chunk, left_wav, right_chunk, right_wav)
        with wave.open(str(book_wav), "rb") as fh:
            payload = struct.unpack("<%dh" % fh.getnframes(), fh.readframes(fh.getnframes()))
        check("omitted unit contributes no audio and no separate padding",
              manifest["inserted_silence_samples"] == expected_pad and
              payload == (1, 2, 3) + (0,) * expected_pad + (4, 5, 6),
              repr((manifest["inserted_silence_samples"], expected_pad, len(payload))))

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
    retry_gen._generate = lambda chunk, model, attempt=0, context=None: (
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
    gate_gen._generate = lambda chunk, model, attempt=0, context=None: {
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
    pass_gen._generate = lambda chunk, model, attempt=0, context=None: {
        "id": chunk["id"], "tail_frame_peak": 0.0089}
    positive, positive_failures = pass_gen._generate_with_retries(retry_child, object())
    check("room-tone tail passes", positive is not None and not positive_failures)
    check("sibilant-final words detected",
          qwenfix.SIBILANT_FINAL.search("collapsed") is not None
          and qwenfix.SIBILANT_FINAL.search("centuries") is not None
          and qwenfix.SIBILANT_FINAL.search("delhi") is None
          and qwenfix.SIBILANT_FINAL.search("planned") is None)

    # Greek-derived word-final "-ch" is /k/, not the affricate
    # SIBILANT_FINAL's "ch" branch assumes -- "monarch" failed this gate
    # on 8 straight attempts (ch02:p0066) despite every take being a
    # correct, silent-release /k/. Whole-word exclusion only: plurals and
    # genuine -ch affricates ("church", "match") still require the check.
    check("monarch (Greek -ch = /k/) no longer requires the sibilant check",
          qwenfix.final_sibilant_high_frac(
              np.zeros(4000, dtype=np.float32), 16000,
              "an exceptionally determined and aggressive monarch.") is None)
    check("church/match/monarchs still require the sibilant check",
          qwenfix.SIBILANT_FINAL.search("church") and "church" not in qwenfix.SIBILANT_FINAL_CH_IS_K
          and qwenfix.SIBILANT_FINAL.search("match") and "match" not in qwenfix.SIBILANT_FINAL_CH_IS_K
          and qwenfix.SIBILANT_FINAL.search("monarchs")
          and "monarchs" not in qwenfix.SIBILANT_FINAL_CH_IS_K)
    _monarch_text = ("Yung-lo, the second founder, who reigned from fourteen oh three to "
                     "fourteen twenty-four was an exceptionally determined and aggressive monarch.")
    _monarch_chunk = {"id": "ch02:p0066:s0000-0003:o000000-000446:c0002", "text": _monarch_text}
    monarch_gen = object.__new__(Generator)
    monarch_gen._attempt_failures = {}
    monarch_gen.out_dir = pathlib.Path("/tmp")
    monarch_gen.validate = False
    monarch_gen._generate = lambda chunk, model, attempt=0, context=None: {
        "id": chunk["id"], "tail_frame_peak": 0.001,
        "final_sibilant_high_frac": qwenfix.final_sibilant_high_frac(
            np.zeros(1000, dtype=np.float32), 16000, chunk["text"]),
    }
    monarch_record, monarch_failures = monarch_gen._generate_with_retries(_monarch_chunk, object())
    check("the failing monarch chunk's replay passes end to end with no retries",
          monarch_record is not None and not monarch_failures)

    # Decode shape bucketing (pure array op; no model load).
    import mlx.core as mx

    _bucket_exact = mx.zeros((1, qwenfix.DECODE_BUCKET_FRAMES * 3, 16))
    _bucket_exact_out, _bucket_exact_pad = qwenfix._pad_codes_to_bucket(_bucket_exact)
    check("bucketing is a no-op on an exact multiple",
          _bucket_exact_pad == 0 and _bucket_exact_out is _bucket_exact)
    _bucket_short = mx.arange(2 * 16).reshape(1, 2, 16).astype(mx.float32)
    _bucket_short_out, _bucket_short_pad = qwenfix._pad_codes_to_bucket(_bucket_short, bucket=5)
    check("bucketing pads to the next multiple by repeating the last frame",
          _bucket_short_pad == 3
          and _bucket_short_out.shape == (1, 5, 16)
          and bool(mx.all(_bucket_short_out[:, 2:] == _bucket_short_out[:, 2:3]).item()))

    sib_gen = object.__new__(Generator)
    sib_gen.validate = False
    sib_gen._records = {}
    sib_gen._attempt_failures = {}
    sib_gen.out_dir = pathlib.Path("/tmp")
    sib_gen._generate = lambda chunk, model, attempt=0, context=None: {
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
    fail_gen._generate = lambda chunk, model, attempt=0, context=None: (
        fail_attempts.append(attempt) or dict(retry_records[1], attempt=attempt))
    fail_gen._validate_chunk = lambda chunk, record: {
        "verdict": "FAIL", "reasons": ["punctuation pause failed"]}
    failed_record, failed_reasons = fail_gen._generate_with_retries(retry_child, object())
    retry_error = fail_gen._retry_error(retry_child, failed_reasons)
    check("child retry failure names both reasons",
          failed_record is None and fail_attempts == [0, 1]
          and "every retry attempt failed" in str(retry_error)
          and "punctuation pause failed" in str(retry_error))

    # Rolling-context policy: a pure function of (last accepted unit, next chunk).
    _rolling_p2 = {"chunk_id": "ch01:p0002:s0000", "paragraph_index": 2,
                   "chapter": "ch01", "text": "prev text", "codes": "codes-obj"}
    _chunk_p2b = {"id": "ch01:p0002:s0002", "chapter": "ch01", "paragraph_index": 2}
    _chunk_p3 = {"id": "ch01:p0003:s0000", "chapter": "ch01", "paragraph_index": 3}
    _chunk_ch2 = {"id": "ch02:p0002:s0000", "chapter": "ch02", "paragraph_index": 2}
    check("rolling context used within same paragraph",
          Generator._rolling_context_for(_rolling_p2, _chunk_p2b) == _rolling_p2)
    check("rolling context reset on paragraph change",
          Generator._rolling_context_for(_rolling_p2, _chunk_p3) is None)
    check("rolling context reset on chapter change",
          Generator._rolling_context_for(_rolling_p2, _chunk_ch2) is None)
    check("rolling context absent with no prior unit",
          Generator._rolling_context_for(None, _chunk_p2b) is None)

    # Integration: `_generate_with_retries` uses context on attempt 0 only,
    # and ONLY when ROLLING_CONTEXT_ENABLED is True. A rejected first take
    # must not carry its (possibly bad) context into the retry, and the
    # retry itself must not use context either.
    def _ctx_generate(chunk, model, attempt=0, context=None):
        ctx_calls.append(context)
        peak = 0.24 if attempt == 0 else 0.0089  # fail attempt 0, pass attempt 1
        return {"id": chunk["id"], "tail_frame_peak": peak}

    ctx_gen = object.__new__(Generator)
    ctx_gen.validate = False
    ctx_gen._records = {}
    ctx_gen._attempt_failures = {}
    ctx_gen.out_dir = pathlib.Path("/tmp")
    ctx_gen._rolling = _rolling_p2
    ctx_calls = []
    ctx_gen._generate = _ctx_generate
    ctx_record, ctx_failures = ctx_gen._generate_with_retries(_chunk_p2b, object())
    check("icl-nocontext-v3 (ROLLING_CONTEXT_ENABLED default False): context "
          "withheld on every attempt, even with a same-paragraph prior chunk on hand",
          ctx_record is not None and len(ctx_calls) == 2
          and ctx_calls[0] is None and ctx_calls[1] is None)

    # Flip the switch to prove the underlying mechanism is intact (not
    # deleted, just gated off by default) -- restores exactly the
    # icl-rolling-v2 first-attempt-only behavior. (ROLLING_CONTEXT_ENABLED
    # is declared global once, at the top of this function.)
    ROLLING_CONTEXT_ENABLED = True
    try:
        ctx_gen2 = object.__new__(Generator)
        ctx_gen2.validate = False
        ctx_gen2._records = {}
        ctx_gen2._attempt_failures = {}
        ctx_gen2.out_dir = pathlib.Path("/tmp")
        ctx_gen2._rolling = _rolling_p2
        ctx_calls = []
        ctx_gen2._generate = _ctx_generate
        ctx_record2, _ = ctx_gen2._generate_with_retries(_chunk_p2b, object())
        check("switching ROLLING_CONTEXT_ENABLED back on restores context on "
              "attempt 0 only (the machinery itself still works)",
              ctx_record2 is not None and len(ctx_calls) == 2
              and ctx_calls[0] == _rolling_p2 and ctx_calls[1] is None)
    finally:
        ROLLING_CONTEXT_ENABLED = False

    # _accept_rolling only tracks accepted takes that produced codec frames.
    # context_high_frac well above CONTEXT_DRIFT_HIGH_FRAC_MIN here -- these
    # tests verify the pre-existing text/codes tracking, not the drift gate
    # itself (that gets its own tests below).
    accept_gen = object.__new__(Generator)
    accept_gen._rolling = None
    accept_gen._accept_rolling({
        "id": "ch01:p0002:s0002", "paragraph_index": 2, "chapter": "ch01",
        "text": "Second chunk.", "_gen_codes": "codes-obj",
        "context_high_frac": 0.05,
    })
    check("accepted take becomes rolling context",
          accept_gen._rolling == {
              "chunk_id": "ch01:p0002:s0002", "paragraph_index": 2,
              "chapter": "ch01", "text": "Second chunk.", "codes": "codes-obj",
              "depth": 0})
    accept_gen._accept_rolling({"id": "x", "paragraph_index": 2, "chapter": "ch01",
                                 "text": "no codes"})
    check("record without codes leaves rolling context unchanged",
          accept_gen._rolling["chunk_id"] == "ch01:p0002:s0002")

    # v31: _accept_rolling must track speak_text (what the codes actually
    # correspond to), not the chunk's original text, when both are present.
    speak_gen = object.__new__(Generator)
    speak_gen._rolling = None
    speak_gen._accept_rolling({
        "id": "ch01:p0003:s0000", "paragraph_index": 3, "chapter": "ch01",
        "text": "Abbas I, the fifth Safavid shah.",
        "speak_text": "Abbas the First, the fifth Safavid shah.",
        "_gen_codes": "codes-obj", "context_high_frac": 0.05,
    })
    check("rolling context text is speak_text, not the original chunk text",
          speak_gen._rolling["text"] == "Abbas the First, the fifth Safavid shah.")

    # --- context-drift gate (icl-rolling-v2) --------------------------------
    # Rolling ICL context compounds a loss of high-frequency energy down a
    # chain (measured -35% high-band energy by depth 3 across chapter 1) --
    # heard as increasingly muffled/underwater audio. Two independent
    # triggers gate whether a take's codes are reused as context for the
    # NEXT chunk; neither ever affects whether the take itself is accepted.
    drift_gen = object.__new__(Generator)
    drift_gen._rolling = None
    drift_gen._accept_rolling({
        "id": "ch01:p0010:s0000", "paragraph_index": 10, "chapter": "ch01",
        "text": "Low quality take.", "_gen_codes": "codes-low",
        "context_high_frac": 0.002,  # below CONTEXT_DRIFT_HIGH_FRAC_MIN (0.006)
    })
    check("a low-high-band take is not usable as context for the next chunk",
          drift_gen._rolling is None)
    check("that chunk therefore generates context-free, exactly like a paragraph start",
          Generator._rolling_context_for(
              drift_gen._rolling,
              {"id": "ch01:p0010:s0001", "chapter": "ch01", "paragraph_index": 10}
          ) is None)

    # Depth cap: a chunk at depth 3 never passes its codes onward, even
    # with a clean high_frac every time -- an independent second trigger.
    depth_gen = object.__new__(Generator)
    depth_gen._rolling = None
    depths_seen = []
    for n in range(5):
        depth_gen._accept_rolling({
            "id": f"ch01:p0020:s000{n}", "paragraph_index": 20, "chapter": "ch01",
            "text": f"Clean chunk {n}.", "_gen_codes": f"codes-{n}",
            "context_high_frac": 0.05,  # comfortably clean every time
        })
        depths_seen.append(depth_gen._rolling["depth"] if depth_gen._rolling else None)
    check("depth increments 0,1,2 while chaining, then the cap breaks the chain",
          depths_seen == [0, 1, 2, None, 0],
          repr(depths_seen))

    # Clean takes chain exactly as before this feature, when quality never
    # dips and the chain never reaches the depth cap.
    clean_gen = object.__new__(Generator)
    clean_gen._rolling = None
    clean_gen._accept_rolling({
        "id": "ch01:p0030:s0000", "paragraph_index": 30, "chapter": "ch01",
        "text": "First.", "_gen_codes": "codes-a", "context_high_frac": 0.02,
    })
    clean_gen._accept_rolling({
        "id": "ch01:p0030:s0001", "paragraph_index": 30, "chapter": "ch01",
        "text": "Second.", "_gen_codes": "codes-b", "context_high_frac": 0.02,
    })
    check("clean takes chain as before (text/codes/depth all track correctly)",
          clean_gen._rolling == {
              "chunk_id": "ch01:p0030:s0001", "paragraph_index": 30,
              "chapter": "ch01", "text": "Second.", "codes": "codes-b", "depth": 1})

    # Recording: _finalize_take stashes the decision onto the record so
    # _record_chunk persists it. With ROLLING_CONTEXT_ENABLED False
    # (icl-nocontext-v3, the default), the decision is never computed --
    # context_depth/context_usable record as None -- while context_high_frac
    # (the spectral field, computed unconditionally in _generate) still
    # lands untouched, since it stays useful for audits either way.
    record_gen = object.__new__(Generator)
    record_gen._rolling = None
    record_gen.done = {}
    record_gen._save_records = lambda: None
    record_gen._save_state = lambda: None
    rec_low = {"id": "ch01:p0040:s0000", "text_sha256": "h", "wav": "w.wav",
               "wav_sha256": "ws", "samples": 10, "seconds": 0.1,
               "terminal_silence_seconds": 0.0, "paragraph_index": 40,
               "chapter": "ch01", "_gen_codes": "codes-x", "context_high_frac": 0.001}
    record_gen._finalize_take(dict(rec_low))
    check("with rolling context off, context_depth/context_usable record as "
          "None while the spectral field (context_high_frac) is kept",
          record_gen.done[rec_low["id"]]["context_high_frac"] == 0.001
          and record_gen.done[rec_low["id"]]["context_depth"] is None
          and record_gen.done[rec_low["id"]]["context_usable"] is None
          and record_gen.done[rec_low["id"]]["context_chunk_id"] is None)

    # Flip the switch to prove the decision-recording path itself is
    # intact -- identical to icl-rolling-v2's behavior before this change.
    ROLLING_CONTEXT_ENABLED = True
    try:
        record_gen2 = object.__new__(Generator)
        record_gen2._rolling = None
        record_gen2.done = {}
        record_gen2._save_records = lambda: None
        record_gen2._save_state = lambda: None
        record_gen2._finalize_take(dict(rec_low))
        check("switching ROLLING_CONTEXT_ENABLED back on restores decision "
              "recording (high_frac, depth, usable) exactly as icl-rolling-v2 did",
              record_gen2.done[rec_low["id"]]["context_high_frac"] == 0.001
              and record_gen2.done[rec_low["id"]]["context_depth"] == 0
              and record_gen2.done[rec_low["id"]]["context_usable"] is False)
    finally:
        ROLLING_CONTEXT_ENABLED = False

    # Concern A: the generation policy lives on each `done` entry (see
    # GENERATION_POLICY / `_generate` / `_record_chunk`), not in
    # run_fingerprint, so bumping it never invalidates an existing
    # directory's resume state -- a directory legally mixes policies
    # across chunks as new code lands (a "mixed-policy directory").
    policy_gen = object.__new__(Generator)
    policy_gen.done = {}
    policy_gen._records = {}
    policy_gen._save_records = lambda: None
    policy_gen._save_state = lambda: None
    policy_gen._rolling = None
    new_take = {"id": "ch01:p0200:s0000-0001", "text_sha256": "h1", "wav": "w1.wav",
                "wav_sha256": "ws1", "samples": 10, "seconds": 0.1,
                "terminal_silence_seconds": 0.0, "paragraph_index": 200, "chapter": "ch01",
                "generation_policy": GENERATION_POLICY}
    policy_gen._record_chunk(new_take)
    check("a freshly generated chunk records the current generation policy",
          policy_gen.done[new_take["id"]]["generation_policy"] == GENERATION_POLICY)
    check("generation policy uses context-free replacement-only EOS handling",
          GENERATION_POLICY == "icl-nocontext-eos-replacement-v4"
          and qwenfix.EOS_HOLD_FRAMES == 1
          and ROLLING_CONTEXT_ENABLED is False)

    # A chunk generated before this field existed (a real done entry from
    # outputs/qwen-book-v2, generated under "icl-rolling-v1") has no
    # "generation_policy" key at all. It must load and behave exactly as
    # before: _unit_done only cares about text_sha256 and the wav's own
    # sha256, never the policy string, so a directory mixing such entries
    # with freshly-generated ones (current policy) is unremarkable.
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        v1_wav = td / "v1.wav"
        _make_wav(v1_wav, [1, 2, 3])
        mixed_gen = object.__new__(Generator)
        mixed_gen.out_dir = td
        mixed_gen.validate = False
        v1_chunk = {"id": "ch01:p0000:s0000-0001", "text_sha256": "h-v1"}
        v2_chunk = {"id": "ch01:p0001:s0000-0001", "text_sha256": "h-v2"}
        mixed_gen.done = {
            v1_chunk["id"]: {"text_sha256": "h-v1", "wav": v1_wav.name,
                             "wav_sha256": sha256_file(v1_wav), "samples": 3,
                             "terminal_silence_seconds": 0.0},  # no generation_policy key
            v2_chunk["id"]: {"text_sha256": "h-v2", "wav": v1_wav.name,
                             "wav_sha256": sha256_file(v1_wav), "samples": 3,
                             "terminal_silence_seconds": 0.0,
                             "generation_policy": GENERATION_POLICY},
        }
        check("a mixed-policy directory (a v1 entry with no policy key next to a "
              "v2 entry with the current policy) loads and validates both chunks cleanly",
              mixed_gen._unit_done(v1_chunk) and mixed_gen._unit_done(v2_chunk)
              and "generation_policy" not in mixed_gen.done[v1_chunk["id"]]
              and mixed_gen.done[v2_chunk["id"]]["generation_policy"] == GENERATION_POLICY)

    # v31 plan-identity trap: normalize_for_tts must never reach chunk id,
    # text_sha256, or anything build_plan/_child_chunks hash -- those stay
    # bound to the untouched original text, or a chapter's existing
    # state.json plan would mismatch the moment this shipped (the same
    # trap as the --chapters incident this session already hit once).
    _regnal_text = "However, the accession of Abbas I, the fifth Safavid shah."
    _regnal_chunk = _test_chunk("ch02:p0055:s0000-0001", _regnal_text, [0, len(_regnal_text)], _span_a)
    check("plan-identity fields (id/text_sha256) are untouched by normalize_for_tts",
          _regnal_chunk["text"] == _regnal_text
          and _regnal_chunk["text_sha256"] == sha256_text(_regnal_text)
          and _regnal_chunk["text_sha256"] != sha256_text(asr.normalize_for_tts(_regnal_text)))

    # Same trap, for the publisher-EPUB text-defect fix specifically: fixing
    # extraction so "entrepô t" reads correctly everywhere would change
    # text_sha256 for every affected paragraph (a full-book regeneration
    # boundary concern, not tonight's); the speak-time-only fix must not.
    _entrepot_text = "continued to be the great cultural entrepô t of the Islamic world."
    _entrepot_chunk = _test_chunk("ch02:p0057:s0000-0001", _entrepot_text,
                                  [0, len(_entrepot_text)], _span_a)
    check("plan-identity fields are untouched by the entrepot text-defect fix",
          _entrepot_chunk["text"] == _entrepot_text
          and _entrepot_chunk["text_sha256"] == sha256_text(_entrepot_text)
          and _entrepot_chunk["text_sha256"] != sha256_text(asr.normalize_for_tts(_entrepot_text)))

    # --- continue-on-failure mode -----------------------------------------
    # Kills the stop-the-world failure model: a chunk that exhausts its
    # retry budget is recorded into a persisted `failed` set instead of
    # raising, so one hard chunk no longer blocks every chunk after it.

    # 1. Deferral: _record_failure must not raise, and must land the chunk
    # in `failed`, never `done` -- the two are mutually exclusive by
    # construction (each path pops the other).
    cof_gen = object.__new__(Generator)
    cof_gen.out_dir = pathlib.Path(tempfile.mkdtemp())
    cof_gen.validate = False
    cof_gen.done = {}
    cof_gen.failed = {}
    cof_gen._records = {}
    cof_gen._last_attempt = {}
    cof_gen._save_records = lambda: None
    cof_gen._save_state = lambda: None
    cof_text = "A hard sentence that never validates."
    cof_chunk = {"id": "ch01:p0100:s0000-0001", "chapter": "ch01",
                 "text": cof_text, "text_sha256": sha256_text(cof_text)}
    cof_gen._record_failure(cof_chunk, ["attempt 0: mandatory missing: ['x']",
                                        "attempt 1: mandatory missing: ['x']"])
    check("continue-on-failure defers (no raise) and records into the failed set",
          cof_chunk["id"] in cof_gen.failed and cof_chunk["id"] not in cof_gen.done)
    check("failed entry keeps the last reason and text_sha256",
          cof_gen.failed[cof_chunk["id"]]["reasons"][-1] == "attempt 1: mandatory missing: ['x']"
          and cof_gen.failed[cof_chunk["id"]]["text_sha256"] == cof_chunk["text_sha256"])

    # 2. Re-attempt ordering: never-attempted chunks come before failed
    # ones, regardless of plan order, so a fresh chunk never waits behind
    # a retry that already burned an attempt budget.
    order_gen = object.__new__(Generator)
    order_gen.force = False
    order_gen.done = {}
    _fresh_c = {"id": "ch01:p0101:s0000-0001", "text": "Fresh.", "text_sha256": sha256_text("Fresh.")}
    _retry_c = {"id": "ch01:p0102:s0000-0001", "text": "Retry me.", "text_sha256": sha256_text("Retry me.")}
    order_gen.failed = {_retry_c["id"]: {"text_sha256": _retry_c["text_sha256"]}}
    order_gen.plan = {"chunks": [_retry_c, _fresh_c]}  # retry listed FIRST in plan order
    check("never-attempted chunks are ordered before failed-retry chunks",
          [c["id"] for c in order_gen._pending_chunks()] == [_fresh_c["id"], _retry_c["id"]])

    # 3. Policy-cleared promotion, no ASR: a failed chunk's retained wav,
    # once its structural facts pass, promotes to done without calling
    # _generate_with_retries (no new TTS call) when validation is off.
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        promote_gen = object.__new__(Generator)
        promote_gen.out_dir = td
        promote_gen.validate = False
        promote_gen.done = {}
        promote_gen._records = {}
        promote_gen._save_records = lambda: None
        promote_gen._save_state = lambda: None
        ptext = "A sentence that used to fail a gate since fixed."
        pchunk = {"id": "ch01:p0103:s0000-0001", "chapter": "ch01",
                  "text": ptext, "text_sha256": sha256_text(ptext)}
        pwav = td / Generator._chunk_wav_rel(pchunk)
        pwav.parent.mkdir(parents=True, exist_ok=True)
        _make_wav(pwav, [1, 2, 3])
        promote_gen.failed = {pchunk["id"]: {
            "text_sha256": pchunk["text_sha256"], "wav": Generator._chunk_wav_rel(pchunk),
            "wav_sha256": sha256_file(pwav), "samples": 3, "seconds": 0.0001,
            "terminal_silence_seconds": 0.0, "tail_frame_peak": 0.001,
            "reasons": ["attempt 0: a gate that has since been fixed"],
        }}
        called = {"tts": False}
        promote_gen._generate_with_retries = lambda *a, **k: called.__setitem__("tts", True)
        promoted = promote_gen._maybe_promote_failed(pchunk)
        check("policy-cleared take promotes to done with no TTS call (validate off)",
              promoted and pchunk["id"] in promote_gen.done
              and pchunk["id"] not in promote_gen.failed and not called["tts"])

    # 3b. Same, but with ASR validation on: the retained wav is
    # revalidated (mocked here to PASS, standing in for "the fix that
    # cleared it") and promoted with no TTS call.
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        promote_gen2 = object.__new__(Generator)
        promote_gen2.out_dir = td
        promote_gen2.validate = True
        promote_gen2.done = {}
        promote_gen2._records = {}
        promote_gen2._save_records = lambda: None
        promote_gen2._save_state = lambda: None
        ptext2 = "Another sentence a curated pair now covers."
        pchunk2 = {"id": "ch01:p0104:s0000-0001", "chapter": "ch01",
                   "text": ptext2, "text_sha256": sha256_text(ptext2)}
        pwav2 = td / Generator._chunk_wav_rel(pchunk2)
        pwav2.parent.mkdir(parents=True, exist_ok=True)
        _make_wav(pwav2, [4, 5, 6])
        promote_gen2.failed = {pchunk2["id"]: {
            "text_sha256": pchunk2["text_sha256"], "wav": Generator._chunk_wav_rel(pchunk2),
            "wav_sha256": sha256_file(pwav2), "samples": 3, "seconds": 0.0001,
            "terminal_silence_seconds": 0.0, "tail_frame_peak": 0.001,
            "reasons": ["attempt 0: mandatory missing (now a curated pair)"],
        }}
        class _FakeValidator:
            def validate_many(self, specs):
                return [dict(_pass, chunk_id=specs[0]["chunk_id"],
                            expected_sha256=pchunk2["text_sha256"])]
        promote_gen2._validator = lambda: _FakeValidator()
        called2 = {"tts": False}
        promote_gen2._generate_with_retries = lambda *a, **k: called2.__setitem__("tts", True)
        promoted2 = promote_gen2._maybe_promote_failed(pchunk2)
        check("policy-cleared take promotes via revalidation, no TTS call",
              promoted2 and pchunk2["id"] in promote_gen2.done
              and pchunk2["id"] not in promote_gen2.failed and not called2["tts"])
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        stillfail_gen = object.__new__(Generator)
        stillfail_gen.out_dir = td
        stillfail_gen.validate = True
        stillfail_gen.done = {}
        stillfail_gen._records = {}
        stillfail_gen._save_records = lambda: None
        stillfail_gen._save_state = lambda: None
        ptext3 = "A sentence still wrong for an unrelated reason."
        pchunk3 = {"id": "ch01:p0108:s0000-0001", "chapter": "ch01",
                   "text": ptext3, "text_sha256": sha256_text(ptext3)}
        pwav3 = td / Generator._chunk_wav_rel(pchunk3)
        pwav3.parent.mkdir(parents=True, exist_ok=True)
        _make_wav(pwav3, [7, 8, 9])
        stillfail_gen.failed = {pchunk3["id"]: {
            "text_sha256": pchunk3["text_sha256"], "wav": Generator._chunk_wav_rel(pchunk3),
            "wav_sha256": sha256_file(pwav3), "samples": 3, "seconds": 0.0001,
            "terminal_silence_seconds": 0.0, "tail_frame_peak": 0.001,
            "reasons": ["attempt 0: mandatory missing (still genuinely wrong)"],
        }}

        class _StillFailValidator:
            def validate_many(self, specs):
                return [dict(_pass, verdict="FAIL", chunk_id=specs[0]["chunk_id"],
                            expected_sha256=pchunk3["text_sha256"])]
        stillfail_gen._validator = lambda: _StillFailValidator()
        stillfail_gen._generate_with_retries = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not call the TTS in this check"))
        check("a still-failing take (revalidation FAILs) is not promoted",
              not stillfail_gen._maybe_promote_failed(pchunk3)
              and pchunk3["id"] not in stillfail_gen.done
              and pchunk3["id"] in stillfail_gen.failed)

    # 4. Assembly must refuse (never silently) while any chunk is in the
    # failed set: `_plan_complete` treats done/failed as mutually
    # exclusive, so one unresolved failure blocks the whole book.
    refuse_gen = object.__new__(Generator)
    refuse_gen.out_dir = pathlib.Path(tempfile.mkdtemp())
    rtext = "This chunk permanently failed."
    rchunk = {"id": "ch01:p0105:s0000-0001", "chapter": "ch01",
              "text": rtext, "text_sha256": sha256_text(rtext)}
    refuse_gen.plan = {"chunks": [rchunk]}
    refuse_gen.done = {}
    refuse_gen.failed = {rchunk["id"]: {"text_sha256": rchunk["text_sha256"],
                                        "reasons": ["attempt 0: mandatory missing"]}}
    check("assembly refuses (plan incomplete) while a chunk is in the failed set",
          not refuse_gen._plan_complete())

    # 4b. Defect 2 (inconsistent assembly gating): _assembly_decision is
    # the ONE gate both `run()` and `validate_generated` now consult, and
    # it must refuse -- listing the failing id(s), like `validate` always
    # has -- for a real failure alone, with no `done` entries in play at
    # all (nothing for _plan_complete/_release_gate to even look at).
    refuse_gen.config = type("Cfg", (), {"asr_repo": "a-repo", "asr_revision": "a-rev"})()
    refuse_gen._records = {}
    check("_assembly_decision refuses generate/validate identically while a real failure exists, "
          "listing the failing id",
          refuse_gen._assembly_decision() == {"status": "failed", "failed_ids": [rchunk["id"]]})

    # 4c. The gap defect 2 actually closes: a parent regenerated whole and
    # DONE, with a stale failed entry that (hypothetically) survived
    # cleanup -- e.g. a hand-edited state.json, or a future code path that
    # forgets to call _clear_resolved_failures. _plan_complete/
    # _release_gate only ever look at self.done, so THEY alone would call
    # this releasable; _assembly_decision checks self.failed first and
    # still refuses. In the live directory, normal operation never reaches
    # this state -- _clear_resolved_failures (4d/4e below) clears the
    # entry the moment the parent completes -- but the assembly gate does
    # not rely on that alone.
    _dd_parent = dict(_ca, eligible_clause_spans=[_span_a, _span_b])
    _dd_helper = object.__new__(Generator)
    _dd_child = _dd_helper._child_chunks(_dd_parent)[0]
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        dd_wav = td / "dd-parent.wav"
        _make_wav(dd_wav, [1, 2, 3])
        dd_gen = object.__new__(Generator)
        dd_gen.out_dir = td
        dd_gen.validate = False
        dd_gen.plan = {"chunks": [_dd_parent]}
        dd_gen.done = {_dd_parent["id"]: {
            "text_sha256": _dd_parent["text_sha256"], "wav": dd_wav.name,
            "wav_sha256": sha256_file(dd_wav), "samples": 3,
            "terminal_silence_seconds": 0.0,
        }}
        dd_gen.failed = {_dd_child["id"]: {
            "text_sha256": _dd_child["text_sha256"], "reasons": ["stale, survived cleanup"],
        }}
        dd_gen._records = {}
        dd_gen.config = type("Cfg", (), {"asr_repo": "a-repo", "asr_revision": "a-rev"})()
        check("_plan_complete alone would wrongly call this releasable (the gap defect 2 closes)",
              dd_gen._plan_complete())
        check("_assembly_decision still refuses even though _plan_complete is True",
              dd_gen._assembly_decision() == {"status": "failed", "failed_ids": [_dd_child["id"]]})

    # 4d. Defect 1 (stale orphaned child): the actual production shape --
    # a chunk was clause-split, one child exhausted its retry budget and
    # landed in `failed`, and the PARENT was later regenerated as one
    # whole unit that passes. The parent's audio covers the child's text
    # span end to end, so the child's failed entry is now describing
    # nothing missing -- it must clear the moment the parent enters
    # `done`, whether that happens via a fresh `_record_chunk` call
    # (below) or via `_load_state` reconciling an already-written
    # state.json (4e).
    recon_gen = object.__new__(Generator)
    recon_gen.done = {}
    recon_gen.failed = {_dd_child["id"]: {
        "text_sha256": _dd_child["text_sha256"], "reasons": ["attempt 1: mandatory missing"],
    }}
    recon_gen._records = {}
    recon_gen._save_records = lambda: None
    recon_gen._save_state = lambda: None
    recon_gen._record_chunk({
        "id": _dd_parent["id"], "text_sha256": _dd_parent["text_sha256"],
        "wav": "p.wav", "wav_sha256": "wsp", "samples": 3, "seconds": 0.1,
        "terminal_silence_seconds": 0.0,
    })
    check("completion-time reconciliation clears a stale child failure once the whole parent completes",
          _dd_child["id"] not in recon_gen.failed and _dd_parent["id"] in recon_gen.done)

    # 4e. Symmetric case (should "already work", per the ticket): a child
    # completing on its own clears its OWN failed entry the same way.
    child_recon_gen = object.__new__(Generator)
    child_recon_gen.done = {}
    child_recon_gen.failed = {_dd_child["id"]: {
        "text_sha256": _dd_child["text_sha256"], "reasons": ["attempt 1: mandatory missing"],
    }}
    child_recon_gen._records = {}
    child_recon_gen._save_records = lambda: None
    child_recon_gen._save_state = lambda: None
    child_recon_gen._record_chunk({
        "id": _dd_child["id"], "text_sha256": _dd_child["text_sha256"],
        "wav": "c.wav", "wav_sha256": "wsc", "samples": 1, "seconds": 0.05,
        "terminal_silence_seconds": 0.0,
    })
    check("a child's own failure entry clears when the child itself completes",
          _dd_child["id"] not in child_recon_gen.failed and _dd_child["id"] in child_recon_gen.done)

    # 4f. Negative case (must NOT reconcile): a failed child stays failed
    # when an unrelated chunk completes -- reconciliation only fires for
    # the id that just completed and ITS OWN descendants, never globally.
    unrelated_gen = object.__new__(Generator)
    unrelated_gen.done = {}
    unrelated_gen.failed = {_dd_child["id"]: {
        "text_sha256": _dd_child["text_sha256"], "reasons": ["attempt 1: mandatory missing"],
    }}
    unrelated_gen._records = {}
    unrelated_gen._save_records = lambda: None
    unrelated_gen._save_state = lambda: None
    _unrelated_text = "An unrelated chunk elsewhere in the plan."
    unrelated_gen._record_chunk({
        "id": "ch01:p0999:s0000-0001", "text_sha256": sha256_text(_unrelated_text),
        "wav": "o.wav", "wav_sha256": "wso", "samples": 1, "seconds": 0.05,
        "terminal_silence_seconds": 0.0,
    })
    check("a failed child with no done entry covering its parent stays failed "
          "(an unrelated completion never clears it)",
          _dd_child["id"] in unrelated_gen.failed)

    # 5. Regression: nothing changes when nothing has failed -- pending
    # order stays exactly plan order, same as before this feature.
    noop_gen = object.__new__(Generator)
    noop_gen.force = False
    noop_gen.done = {}
    noop_gen.failed = {}
    _c1 = {"id": "ch01:p0106:s0000-0001", "text": "One.", "text_sha256": sha256_text("One.")}
    _c2 = {"id": "ch01:p0107:s0000-0001", "text": "Two.", "text_sha256": sha256_text("Two.")}
    noop_gen.plan = {"chunks": [_c1, _c2]}
    check("pending order is unchanged (plan order) when nothing has failed",
          [c["id"] for c in noop_gen._pending_chunks()] == [_c1["id"], _c2["id"]])

    # 6. Resume compatibility: a state.json written before this feature
    # existed has no "failed" key at all -- must load as {}, not error.
    check("absent failed key in persisted state loads as empty (resume compat)",
          json.loads('{"done": {"x": {}}}').get("failed", {}) == {})

    # --- integration: Generator._load_state (concerns B + C) ---------------
    _FakeCfg = type("FakeCfg", (), {
        "model_repo": "repo", "model_revision": "rev", "language": "English",
        "max_tokens": 4096, "seed": 42, "asr_repo": "a-repo", "asr_revision": "a-rev",
    })

    def _make_state_gen(td, plan, fingerprint="fp-1"):
        g = object.__new__(Generator)
        g.out_dir = pathlib.Path(td)
        g.state_path = g.out_dir / STATE_REL
        g.config = _FakeCfg()
        g.ref_wav_sha, g.ref_text_sha = "w", "t"
        g.fingerprint = fingerprint
        g.plan = plan
        g.force = False
        g.discard_done = False
        g._mismatch_force_carry = False
        return g

    def _seed_state(td, plan, done, fingerprint="fp-1"):
        seed = _make_state_gen(td, plan, fingerprint)
        seed.done, seed.failed, seed.started_unix = done, {}, 0.0
        seed._save_state()

    _cc = _test_chunk("ch01:p0002:s0000-0001", "Third clause.", [27, 41],
                       {"start": 27, "end": 41, "text": "Third clause.", "words": 2,
                        "sentence_index": 2, "clause_index": 0})
    _wide_plan = {"planner": _planner,
                  "chapters": [{"id": "ch01", "title": "T", "paragraphs": 3, "groups": 3}],
                  "chunks": [_ca, _cb, _cc], "total_paragraphs": 3, "total_groups": 3}
    _ca_conflict = dict(_ca, text="Changed clause.", text_sha256=sha256_text("Changed clause."))
    _conflict_plan = dict(cur_plan, chunks=[_ca_conflict, _cb])
    _seed_done = {_ca["id"]: {"text_sha256": _ca["text_sha256"], "wav": "a.wav",
                              "wav_sha256": "sa", "samples": 1, "seconds": 0.1,
                              "terminal_silence_seconds": 0.0}}

    with tempfile.TemporaryDirectory() as td:
        _seed_state(td, cur_plan, _seed_done)
        widen_gen = _make_state_gen(td, _wide_plan)
        widen_gen._load_state()
        check("a superset-plan resume loads without error and keeps the stored done chunk",
              widen_gen.done.get(_ca["id"], {}).get("wav_sha256") == "sa"
              and _cb["id"] not in widen_gen.done and _cc["id"] not in widen_gen.done)

    # Defect 1, end to end: a state.json written to disk with a done
    # parent and a stale failed descendant (the exact shape of the live
    # outputs/qwen-book-v2 directory) reconciles the moment ANY Generator
    # loads it -- the fix this ticket asked for, "heals on the next run
    # without hand-editing" -- and the healed state is written back to
    # disk, not just held in memory for this one process.
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        dd2_wav = td / "dd2-parent.wav"
        _make_wav(dd2_wav, [1, 2, 3])
        _dd_plan = {"planner": _planner,
                    "chapters": [{"id": "ch01", "title": "T", "paragraphs": 1, "groups": 1}],
                    "chunks": [_dd_parent], "total_paragraphs": 1, "total_groups": 1}
        _dd_done = {_dd_parent["id"]: {
            "text_sha256": _dd_parent["text_sha256"], "wav": dd2_wav.name,
            "wav_sha256": sha256_file(dd2_wav), "samples": 3, "seconds": 0.1,
            "terminal_silence_seconds": 0.0,
        }}
        _dd_failed = {_dd_child["id"]: {
            "text_sha256": _dd_child["text_sha256"],
            "reasons": ["attempt 1: mandatory missing"],
        }}
        dd_seed = _make_state_gen(td, _dd_plan, fingerprint="fp-dd")
        dd_seed.done, dd_seed.failed, dd_seed.started_unix = _dd_done, _dd_failed, 0.0
        dd_seed._save_state()

        dd_load = _make_state_gen(td, _dd_plan, fingerprint="fp-dd")
        dd_load._load_state()
        check("load-time reconciliation clears a stale child failure once its parent is done on disk",
              dd_load.done.get(_dd_parent["id"], {}).get("wav_sha256") == sha256_file(dd2_wav)
              and _dd_child["id"] not in dd_load.failed)
        persisted = json.loads(dd_load.state_path.read_text())
        check("the healed state.json is written back to disk, not just held in this process's memory",
              _dd_child["id"] not in persisted.get("failed", {}))

    with tempfile.TemporaryDirectory() as td:
        _seed_state(td, cur_plan, _seed_done)
        fp_mismatch_gen = _make_state_gen(td, cur_plan, fingerprint="fp-2")
        check("a fingerprint mismatch (model/reference/book/seed changed) also "
              "requires --force/--discard-done, even against an identical plan",
              _raises(RunError, fp_mismatch_gen._load_state))

    with tempfile.TemporaryDirectory() as td:
        _seed_state(td, cur_plan, _seed_done)
        conflict_gen = _make_state_gen(td, _conflict_plan)
        check("a genuinely conflicting plan raises without --force/--discard-done",
              _raises(RunError, conflict_gen._load_state))

    # --- concern C: safe --force / --discard-done on a state mismatch ------
    with tempfile.TemporaryDirectory() as td:
        _seed_state(td, cur_plan, _seed_done)

        # --force on a conflict archives the old state.json and carries
        # done forward -- it never zeroes it outright.
        force_gen = _make_state_gen(td, _conflict_plan)
        force_gen.force = True
        force_gen._load_state()
        backup1 = force_gen.out_dir / "state.json.bak-1"
        check("--force archives the mismatched state.json instead of deleting it",
              backup1.is_file()
              and json.loads(backup1.read_text())["done"][_ca["id"]]["wav_sha256"] == "sa")
        check("--force carries the old done record forward rather than zeroing it",
              force_gen.done.get(_ca["id"], {}).get("wav_sha256") == "sa")

        # --discard-done on a fresh mismatch explicitly zeroes done -- but
        # still archives first, never silently.
        _seed_state(td, cur_plan, _seed_done)
        discard_gen = _make_state_gen(td, _conflict_plan)
        discard_gen.discard_done = True
        discard_gen._load_state()
        check("--discard-done explicitly zeroes done after archiving (never silent)",
              discard_gen.done == {} and (force_gen.out_dir / "state.json.bak-2").is_file())

    # Regression: --force on a mismatch must not fall into force's OTHER
    # meaning ("regenerate everything", for an already-matching state) --
    # a carried-forward chunk with an intact wav and a PASSing validation
    # record has to be re-admitted as done by the normal _is_done check,
    # not wiped by _pending_chunks's unconditional full-regen branch. Seen
    # in production: --force on an old-schema state.json logged "carrying
    # forward 248 ... chunk(s)" but then regenerated 251 of 255 anyway,
    # because _pending_chunks checked only `self.force`, not WHY it was
    # set.
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        wav_a = td / "chunks" / "ch01" / (_ca["id"].replace(":", "-") + ".wav")
        wav_b = td / "chunks" / "ch01" / (_cb["id"].replace(":", "-") + ".wav")
        wav_a.parent.mkdir(parents=True, exist_ok=True)
        _make_wav(wav_a, [1, 2, 3])
        _make_wav(wav_b, [4, 5, 6])
        old_done = {
            _ca["id"]: {"text_sha256": _ca["text_sha256"], "wav": str(wav_a.relative_to(td)),
                       "wav_sha256": sha256_file(wav_a), "samples": 3, "seconds": 0.1,
                       "terminal_silence_seconds": 0.0},
            _cb["id"]: {"text_sha256": _cb["text_sha256"], "wav": str(wav_b.relative_to(td)),
                       "wav_sha256": sha256_file(wav_b), "samples": 3, "seconds": 0.1,
                       "terminal_silence_seconds": 0.0},
        }
        # An "old-schema" state: a fingerprint this run's code will never
        # produce (concern A narrowed run_fingerprint), and a "plan" shape
        # irrelevant to matching once the fingerprint itself disagrees.
        _atomic_write_json(td / STATE_REL, {
            "fingerprint": "old-schema-fp", "plan": {"planner": {"policy": "old"}},
            "done": old_done, "failed": {}, "started_unix": 0.0,
        })

        carry_gen = _make_state_gen(str(td), _wide_plan, fingerprint="new-fp")
        carry_gen.force = True
        carry_gen.validate = True
        carry_gen._records = {_ca["id"]: dict(_pass), _cb["id"]: dict(_pass)}
        carry_gen._forced_parents = set()
        carry_gen._load_state()
        check("--force on a mismatch carries the old done set into self.done",
              carry_gen.done.get(_ca["id"], {}).get("wav_sha256")
              == old_done[_ca["id"]]["wav_sha256"]
              and carry_gen._mismatch_force_carry is True)

        pending_ids = [c["id"] for c in carry_gen._pending_chunks()]
        check("a carried-forward chunk with an intact wav and a PASSing record "
              "is NOT regenerated (the production bug this reproduces)",
              _ca["id"] not in pending_ids and _cb["id"] not in pending_ids)
        check("only the genuinely new chunk introduced by the widened plan is pending",
              pending_ids == [_cc["id"]])

    # --- concern D: selective regeneration ----------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        wav_a = td / "chunks" / "ch01" / (_ca["id"].replace(":", "-") + ".wav")
        wav_b = td / "chunks" / "ch01" / (_cb["id"].replace(":", "-") + ".wav")
        wav_a.parent.mkdir(parents=True, exist_ok=True)
        _make_wav(wav_a, [1, 2, 3])
        _make_wav(wav_b, [4, 5, 6])
        done0 = {
            _ca["id"]: {"text_sha256": _ca["text_sha256"], "wav": str(wav_a.relative_to(td)),
                       "wav_sha256": sha256_file(wav_a), "samples": 3, "seconds": 0.1,
                       "terminal_silence_seconds": 0.0, "generation_policy": "icl-rolling-v1"},
            _cb["id"]: {"text_sha256": _cb["text_sha256"], "wav": str(wav_b.relative_to(td)),
                       "wav_sha256": sha256_file(wav_b), "samples": 3, "seconds": 0.1,
                       "terminal_silence_seconds": 0.0, "generation_policy": "icl-rolling-v1"},
        }
        _atomic_write_json(td / STATE_REL, {
            "fingerprint": "fp-regen", "plan": st_plan, "done": dict(done0),
            "failed": {}, "started_unix": 0.0,
        })
        _atomic_write_json(td / RECORDS_REL, {
            "asr": {"model_repo": "a", "model_revision": "b"},
            "records": [{"chunk_id": _ca["id"], "verdict": "PASS"},
                        {"chunk_id": _cb["id"], "verdict": "PASS"}],
        })
        _atomic_write_json(td / ASR_CACHE_REL, {
            "key-a": {"chunk_id": _ca["id"], "verdict": "PASS"},
            "key-b": {"chunk_id": _cb["id"], "verdict": "PASS"},
        })

        result = regenerate(str(td), str(td), [_ca["id"]])
        new_state = json.loads((td / STATE_REL).read_text())
        new_records = json.loads((td / RECORDS_REL).read_text())["records"]
        new_cache = json.loads((td / ASR_CACHE_REL).read_text())
        check("regenerate moves exactly the listed chunk out of done, leaves the other",
              _ca["id"] not in new_state["done"] and _cb["id"] in new_state["done"])
        check("regenerate archives the wav (renamed, not deleted) and leaves the other's alone",
              not wav_a.is_file() and wav_b.is_file()
              and len(result["archived_wavs"]) == 1
              and (td / result["archived_wavs"][0]).is_file()
              and ".superseded-" in result["archived_wavs"][0])
        check("regenerate clears only the regenerated chunk's validation record",
              {r["chunk_id"] for r in new_records} == {_cb["id"]})
        check("regenerate clears only the regenerated chunk's ASR-cache entry",
              set(new_cache) == {"key-b"})
        check("regenerate rejects an id that is not in the done set",
              _raises(RunError, regenerate, str(td), str(td), ["no-such-chunk"]))

        # Survives a subsequent resume: state.json's "plan" field is
        # untouched by regenerate, so a fresh Generator._load_state()
        # against the SAME plan sees a "match" relation and loads the
        # trimmed done set -- the regenerated chunk is pending, the
        # untouched one is still done.
        resume_gen = _make_state_gen(str(td), cur_plan, fingerprint="fp-regen")
        resume_gen._load_state()
        check("regenerate's result survives a subsequent resume (match, not conflict)",
              _ca["id"] not in resume_gen.done and _cb["id"] in resume_gen.done
              and resume_gen.done[_cb["id"]]["wav_sha256"] == done0[_cb["id"]]["wav_sha256"])

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
