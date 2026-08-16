"""Persistent ASR validation for generated audiobook chunks (mlx-whisper).

Engine used by the runner's ``validate`` CLI subcommand (and independently
via ``python -m audiobook.asr``). One Whisper model per process: mlx-whisper
0.4.3 caches the loaded model in a module-level ``ModelHolder`` keyed by the
local snapshot path, so repeated ``transcribe`` calls never reload weights.

Contract (runner side, see README):
    from audiobook.asr import AsrValidator

    v = AsrValidator()  # lazy: nothing loads until the first validate_chunk
    record = v.validate_chunk(wav_path, expected_text, chunk_id="s1")
    v.validate_many([{wav, expected_text, chunk_id, mandatory?, leakage_texts?}])
    v.stats()           # measured {chunks, asr_seconds, audio_seconds, rtf, ...}
    v.eta_estimate(remaining_audio_seconds)

Records are plain JSON dicts (schema below); the optional cache file is a
JSON object keyed by
``<model_repo>|<model_revision>|<language>|<word_timestamps>|<wav_sha256>|<expected_sha256>``
and written atomically (tmp + rename), so revalidation after a crash reuses
completed chunks. Failed chunks are flagged in the record (verdict FAIL +
reasons) and
never silently regenerated or retried without the caller asking.

Record schema (all values JSON-native):
    chunk_id, wav_sha256, expected_sha256,
    asr: {model_repo, model_revision}, language,
    transcript, transcript_normalized, asr_tokens,
    confidence: {avg_logprob, no_speech_prob, compression_ratio},
    coverage: {expected_tokens, matched_tokens, fraction, missing},
    mandatory: {items, missing},
    repetition: {max_multiplicity, most_repeated, repeated_count},
    leakage: {flagged, detail},
    words: {count, first_start, last_end, max_internal_gap_s, long_gaps_over_1s},
    signal: {seconds, sample_rate, channels, subtype, rms, peak, active_ratio, sha256},
    verdict, reasons, asr_seconds, rtf, cache_hit

Module import is stdlib-only; numpy/soundfile load lazily on first audio
read, mlx_whisper lazily on first transcribe (actionable error otherwise).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter

# --- frozen ASR model (default) ---------------------------------------------
# whisper-tiny (weights.npz layout): loadable by pinned mlx-whisper 0.4.3,
# already used for the reference ASR runs. revision == cache refs/main.
DEFAULT_MODEL_REPO = "mlx-community/whisper-tiny"
DEFAULT_MODEL_REVISION = "78c52ab98ca87f570bc57ad852e15ef7060f9f76"
DEFAULT_LANGUAGE = "en"

# --- verdict thresholds ------------------------------------------------------
COVERAGE_MIN = 0.85          # fraction of expected tokens found, in order
NO_SPEECH_MAX = 0.6          # whisper's own no-speech threshold
LOGPROB_MIN = -1.0           # whisper's own logprob threshold
COMPRESSION_MAX = 2.4        # whisper's own repetition/loop threshold
REPEAT_MAX_MULTIPLICITY = 3  # same adjacent n-gram (n>=2) seen this many times
MAX_INTERNAL_GAP_S = 2.5     # internal silence inside a spoken chunk
MANDATORY_FUZZY_RATIO = 0.8  # token match floor (difflib ratio) for mandatory
LEAKAGE_OVERLAP_MIN = 0.8    # token overlap with a leakage text that flags

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}
_TENS_COMPOUND_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-"
    r"(one|two|three|four|five|six|seven|eight|nine)\b"
)

# function words >= 5 chars that are not load-bearing content
_FUNCTION_WORDS = {
    "within", "during", "around", "before", "after", "between", "through",
    "across", "against", "without", "because", "although", "however",
    "together", "another", "about", "above", "below", "under", "while",
    "since", "until", "beyond", "toward", "among", "along", "these", "those",
    "there", "their", "which", "where", "when", "what", "would", "could",
    "should", "might", "must", "being", "been", "very", "most", "more",
    "than", "that", "with", "from", "into", "upon", "then", "them", "they",
    "this", "have", "has", "had", "were", "was", "will", "just", "only",
    "also", "even", "still", "other", "first", "last", "next", "each",
    "every", "both", "some", "such", "same", "much", "many", "never", "ever",
    "here", "over", "under",
}


# --- hashing -----------------------------------------------------------------
def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --- normalization -----------------------------------------------------------
def tokenize(text: str) -> list:
    """Normalize spoken text into comparable tokens.

    Lowercases, drops punctuation, folds numbers so both sides agree:
    number words -> digits, "fourteen oh five" -> "1405" (digit runs merge),
    hyphenated compounds ("twenty-one") -> "21". Apostrophes are kept so
    possessives/contractions survive. Both expected text and transcript go
    through the same transform, so any heuristic applies symmetrically.
    """
    t = text.lower().replace("\u2019", "'")
    # hyphenated number compounds first (before punctuation stripping)
    t = _TENS_COMPOUND_RE.sub(
        lambda m: str(int(_NUM_WORDS[m.group(1)]) + int(_NUM_WORDS[m.group(2)])), t
    )
    t = re.sub(r"[^a-z0-9']+", " ", t)
    out = []
    for tok in t.split():
        if tok in _NUM_WORDS:
            out.append(_NUM_WORDS[tok])
        elif tok == "oh":
            # spoken digit "oh" ("fourteen oh five"); harmless elsewhere since
            # both sides normalize identically
            out.append("0")
        else:
            out.append(tok)
    merged = []
    run = []
    for tok in out:
        if tok.isdigit():
            run.append(tok)
        else:
            if run:
                merged.append("".join(run))
                run = []
            merged.append(tok)
    if run:
        merged.append("".join(run))
    # ponytail: digit-run merge ("14 0 5" -> "1405") can misfire on unrelated
    # adjacent number words ("nineteen eighty seven" -> "19807"); acceptable
    # for this corpus (sentences are known), revisit with a word-phrase map if
    # years-within-prose ever gate wrongly.
    return merged


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


# --- alignment metrics -------------------------------------------------------
def ordered_coverage(expected: list, asr: list) -> dict:
    """Maximum ordered token coverage of expected in ASR tokens (LCS).

    A token of `expected` is matched if the expected tokens form a
    subsequence of the ASR tokens; the best alignment is found with a
    classic LCS DP, so an expected token absent from the audio only costs
    itself, never the tokens after it. O(len(expected) * len(asr)) — trivial
    at sentence scale.
    """
    n, m = len(expected), len(asr)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ei = expected[i]
        row, nxt = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if ei == asr[j] else max(nxt[j], row[j + 1])
    matched = dp[0][0]
    missing = []
    i = j = 0
    while i < n and j < m:
        if expected[i] == asr[j]:
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            missing.append(expected[i])
            i += 1
        else:
            j += 1
    missing.extend(expected[i:])
    return {
        "expected_tokens": n,
        "matched_tokens": matched,
        "fraction": round(matched / n, 4) if n else 1.0,
        "missing": missing,
    }


def _tok_eq(a: str, b: str, ratio: float) -> bool:
    if a == b:
        return True
    # phonetic tolerance for proper nouns ("Tamerlane" ~ "Tamalane") without
    # letting short function words match anything
    if len(a) >= 4 and len(b) >= 4:
        return difflib.SequenceMatcher(None, a, b).ratio() >= ratio
    return False


def _phrase_in(asr: list, phrase: list, ratio: float) -> bool:
    n = len(phrase)
    if n == 1:
        return any(_tok_eq(t, phrase[0], ratio) for t in asr)
    for i in range(len(asr) - n + 1):
        if all(_tok_eq(asr[i + k], phrase[k], ratio) for k in range(n)):
            return True
    return False


def check_mandatory(asr: list, mandatory: list, ratio: float = MANDATORY_FUZZY_RATIO) -> dict:
    """Each mandatory item (string or token list) must appear as a contiguous,
    fuzzy-tolerant run in the ASR tokens."""
    items = []
    missing = []
    for item in mandatory:
        phrase = tokenize(item) if isinstance(item, str) else list(item)
        if not phrase:
            continue
        label = item if isinstance(item, str) else " ".join(item)
        items.append(label)
        if not _phrase_in(asr, phrase, ratio):
            missing.append(label)
    return {"items": items, "missing": missing}


def derive_mandatory(expected_tokens: list) -> list:
    """Default mandatory content: digits plus content words (len >= 6, not
    function words) from the expected text. Callers may pass an explicit
    `mandatory` list to override."""
    out = []
    seen = set()
    for tok in expected_tokens:
        if tok in seen:
            continue
        keep = tok.isdigit() or (len(tok) >= 6 and tok.isalpha() and tok not in _FUNCTION_WORDS)
        if keep:
            seen.add(tok)
            out.append(tok)
    return out


def repetition_stats(tokens: list) -> dict:
    """Repeated adjacent n-grams (n=2..4) — whisper loop/hallucination signal."""
    counts = Counter()
    for n in (2, 3, 4):
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i:i + n])] += 1
    reps = {k: v for k, v in counts.items() if v > 1}
    if not reps:
        return {"max_multiplicity": 1, "most_repeated": None, "repeated_count": 0}
    best = max(reps, key=reps.get)
    return {
        "max_multiplicity": reps[best],
        "most_repeated": " ".join(best),
        "repeated_count": len(reps),
    }


def leakage_check(asr: list, leakage_texts) -> dict:
    """Flag if ASR output heavily overlaps a leakage text (e.g. the reference
    transcript the TTS was cloned on) — the model reciting the reference
    instead of the target sentence."""
    if not leakage_texts:
        return {"flagged": False, "detail": None}
    best = None
    for text in leakage_texts:
        toks = tokenize(text)
        if not toks:
            continue
        uniq = set(toks)
        inter = sum(1 for tok in uniq if tok in set(asr))
        frac = inter / len(uniq)
        if best is None or frac > best[1]:
            best = (text, frac, inter, len(uniq))
    if best is None:
        return {"flagged": False, "detail": None}
    text, frac, inter, total = best
    flagged = frac >= LEAKAGE_OVERLAP_MIN
    return {
        "flagged": flagged,
        "detail": None if not flagged else f"token overlap {frac:.2f} ({inter}/{total}) with leakage text",
    }


# --- audio -------------------------------------------------------------------
def _resample_16k(x, sr_in: int):
    """Windowed-sinc resample to 16 kHz in-process (no ffmpeg subprocess).

    Supports the rates this pipeline produces: 24k (2/3), 32k (1/2), 48k
    (1/3); 16k passes through. Filtering happens at the upsampled rate, so
    the anti-alias cutoff is 8 kHz for every supported input.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if sr_in == 16000:
        return x
    if sr_in not in (24000, 32000, 48000):
        raise ValueError(
            f"unsupported input sample rate {sr_in} for 16 kHz resample "
            "(supported: 16000/24000/32000/48000)"
        )
    num, den = (2, 3) if sr_in == 24000 else ((1, 2) if sr_in == 32000 else (1, 3))
    rate = sr_in * num
    up = np.zeros(len(x) * num, dtype=np.float32)
    up[::num] = x
    cutoff = min(8000.0, sr_in / 2.0)
    taps = 64
    n = np.arange(-taps, taps + 1)
    h = np.sinc(2 * cutoff / rate * n) * np.hanning(2 * taps + 1)
    h /= h.sum()
    y = np.convolve(up, h, mode="same") * num  # gain for zero-stuffed upsample
    return y[::den][: int(round(len(x) * num / den))]


def _signal_facts(wav: pathlib.Path) -> dict:
    """PCM16 structural facts of the chunk WAV (same conventions as the
    generator runner's out_facts)."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(wav))
    data = sf.read(str(wav), dtype="int16")[0]
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak_int16 = int(np.max(np.abs(data))) if data.size else 0
    f32 = data.astype(np.float64) / 32768.0
    return {
        "seconds": float(data.size / info.samplerate),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "rms": float(np.sqrt(np.mean(np.square(f32)))) if data.size else 0.0,
        "peak": float(peak_int16 / 32768.0),
        "active_ratio": float(np.mean(np.abs(f32) > 0.01)) if data.size else 0.0,
        "sha256": sha256_file(wav),
    }


def _word_stats(segments: list) -> dict:
    words = [
        (float(w["start"]), float(w["end"]))
        for s in segments
        for w in (s.get("words") or [])
        if "start" in w and "end" in w
    ]
    if not words:
        return {"count": 0, "first_start": None, "last_end": None,
                "max_internal_gap_s": None, "long_gaps_over_1s": 0}
    gaps = [words[i][0] - words[i - 1][1] for i in range(1, len(words))]
    return {
        "count": len(words),
        "first_start": round(words[0][0], 3),
        "last_end": round(words[-1][1], 3),
        "max_internal_gap_s": round(max(gaps), 3) if gaps else None,
        "long_gaps_over_1s": int(sum(1 for g in gaps if g > 1.0)),
    }


def _confidence(segments: list) -> dict:
    if not segments:
        return {"avg_logprob": None, "no_speech_prob": None, "compression_ratio": None}
    n = len(segments)
    return {
        "avg_logprob": round(float(sum(s.get("avg_logprob", 0.0) for s in segments)) / n, 4),
        "no_speech_prob": round(float(sum(s.get("no_speech_prob", 0.0) for s in segments)) / n, 4),
        "compression_ratio": round(float(sum(s.get("compression_ratio", 0.0) for s in segments)) / n, 4),
    }


# --- verdict -----------------------------------------------------------------
def verdict(metrics: dict) -> tuple:
    reasons = []
    cov = metrics["coverage"]
    if cov["fraction"] < COVERAGE_MIN:
        reasons.append(
            f"coverage {cov['fraction']:.2f} < {COVERAGE_MIN} "
            f"(matched {cov['matched_tokens']}/{cov['expected_tokens']}, missing {cov['missing'][:5]})"
        )
    mand = metrics["mandatory"]
    if mand["missing"]:
        reasons.append(f"mandatory missing: {mand['missing'][:5]}")
    conf = metrics["confidence"]
    if conf.get("no_speech_prob") is not None and conf["no_speech_prob"] >= NO_SPEECH_MAX:
        reasons.append(f"no_speech_prob {conf['no_speech_prob']:.2f} >= {NO_SPEECH_MAX}")
    if conf.get("avg_logprob") is not None and conf["avg_logprob"] < LOGPROB_MIN:
        reasons.append(f"avg_logprob {conf['avg_logprob']:.2f} < {LOGPROB_MIN}")
    if conf.get("compression_ratio") is not None and conf["compression_ratio"] > COMPRESSION_MAX:
        reasons.append(f"compression_ratio {conf['compression_ratio']:.2f} > {COMPRESSION_MAX}")
    rep = metrics["repetition"]
    if rep["max_multiplicity"] >= REPEAT_MAX_MULTIPLICITY:
        reasons.append(
            f"repetition: n-gram {rep['most_repeated']!r} x{rep['max_multiplicity']}"
        )
    gap = metrics["words"]["max_internal_gap_s"]
    if gap is not None and gap > MAX_INTERNAL_GAP_S:
        reasons.append(f"max internal gap {gap:.2f}s > {MAX_INTERNAL_GAP_S}s")
    if metrics["leakage"]["flagged"]:
        reasons.append(metrics["leakage"]["detail"])
    return ("FAIL" if reasons else "PASS", reasons)


# --- cache -------------------------------------------------------------------
def _atomic_write_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


# --- engine ------------------------------------------------------------------
class AsrValidator:
    """In-process mlx-whisper validation of generated audio chunks.

    The model is loaded exactly once per process (mlx-whisper's ModelHolder
    caches by snapshot path) and reused for every chunk; there are no
    subprocesses and no per-chunk model reloads.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        revision: str = DEFAULT_MODEL_REVISION,
        language: str = DEFAULT_LANGUAGE,
        word_timestamps: bool = True,
        cache_path=None,
    ):
        self.model_repo = model_repo
        self.revision = revision
        self.language = language
        self.word_timestamps = word_timestamps
        self.cache_path = pathlib.Path(cache_path) if cache_path else None
        self._engine = None  # lazily imported mlx_whisper module
        self._snapshot_dir = None
        self._cache = None  # {key: record} loaded lazily
        self._chunks = 0
        self._asr_seconds = 0.0
        self._audio_seconds = 0.0
        self._load_seconds = None  # wall time of the first transcribe (model load)

    # -- model resolution -----------------------------------------------------
    def _resolve_snapshot(self) -> str:
        hf_home = pathlib.Path(
            os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface")
        )
        snap = (
            hf_home / "hub" / f"models--{self.model_repo.replace('/', '--')}"
            / "snapshots" / self.revision
        )
        if (snap / "config.json").is_file():
            return str(snap)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub not importable; cannot fetch ASR model weights. "
                "Install the project env with `uv sync --locked`."
            ) from e
        try:
            return str(pathlib.Path(
                snapshot_download(repo_id=self.model_repo, revision=self.revision)
            ))
        except Exception as e:  # offline, missing cache, network
            raise RuntimeError(
                f"ASR model {self.model_repo}@{self.revision} not in HF cache and "
                f"snapshot_download failed: {e}. On the first run this machine needs "
                f"network access (or HF_HUB_OFFLINE must be unset); the cache dir is "
                f"{hf_home}. The weights are also downloadable manually into that "
                f"snapshots/{self.revision} path."
            ) from e

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            import mlx_whisper  # lazy: heavy MLX import, only on first validation
        except ImportError as e:
            raise RuntimeError(
                "ASR validation requires mlx-whisper (pinned 0.4.3). Install the "
                "project env with `uv sync --locked` (or `uv pip install "
                "mlx-whisper==0.4.3`). It is imported lazily — this error only "
                "appears when validation is first used."
            ) from e
        snap = pathlib.Path(self._resolve_snapshot())
        wf = snap / "weights.safetensors"
        if not wf.is_file():
            wf = snap / "weights.npz"
        if not wf.is_file():
            raise RuntimeError(
                f"ASR snapshot {snap} has no weights.safetensors/weights.npz. "
                "mlx-whisper 0.4.3 only loads those two layouts (repos shipping "
                "model.safetensors, e.g. mlx-community/whisper-base-asr-fp16, are "
                f"unsupported). The default {DEFAULT_MODEL_REPO} uses weights.npz."
            )
        self._snapshot_dir = str(snap)
        self._engine = mlx_whisper
        return self._engine

    # -- cache ----------------------------------------------------------------
    def _load_cache(self) -> dict:
        if self._cache is None:
            self._cache = {}
            if self.cache_path and self.cache_path.is_file():
                try:
                    self._cache = json.loads(self.cache_path.read_text())
                except (json.JSONDecodeError, OSError):
                    self._cache = {}  # corrupt cache: rebuild from scratch
        return self._cache

    def _cache_key(self, wav_sha: str, exp_sha: str) -> str:
        return "|".join((
            self.model_repo,
            self.revision,
            self.language,
            "wt" if self.word_timestamps else "nowt",
            wav_sha,
            exp_sha,
        ))

    def _cache_get(self, key: str):
        return self._load_cache().get(key)

    def _cache_put(self, key: str, record: dict) -> None:
        self._load_cache()[key] = {k: v for k, v in record.items() if k != "cache_hit"}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.cache_path, self._cache)

    # -- transcription --------------------------------------------------------
    def _transcribe(self, audio16):
        engine = self._ensure_engine()
        t0 = time.perf_counter()
        result = engine.transcribe(
            audio16,
            path_or_hf_repo=self._snapshot_dir,
            verbose=None,               # silent
            temperature=0.0,            # greedy: deterministic validation
            condition_on_previous_text=False,  # chunks are independent
            word_timestamps=self.word_timestamps,
            language=self.language,
        )
        wall = time.perf_counter() - t0
        if self._load_seconds is None:
            # first call pays the one-time model load (ModelHolder cache)
            self._load_seconds = wall
        return result, wall

    # -- public API -----------------------------------------------------------
    def validate_chunk(
        self,
        wav_path,
        expected_text: str,
        chunk_id=None,
        mandatory=None,
        leakage_texts=None,
    ) -> dict:
        """Transcribe one chunk and validate it against expected_text.

        Returns a JSON-native record. Cache hits (same wav sha256 + ASR model
        revision + expected text) short-circuit transcription. Raises on
        missing files or engine errors; callers that want per-chunk failure
        records should use validate_many().
        """
        wav = pathlib.Path(wav_path)
        if not wav.is_file():
            raise FileNotFoundError(f"chunk wav not found: {wav}")
        wav_sha = sha256_file(wav)
        exp_sha = sha256_bytes(expected_text.encode("utf-8"))
        key = self._cache_key(wav_sha, exp_sha)
        cached = self._cache_get(key)
        if cached is not None and cached.get("expected_sha256") == exp_sha:
            return {**cached, "cache_hit": True}

        signal = _signal_facts(wav)
        import numpy as np

        data = np.asarray(sf_read_16k(wav), dtype=np.float32)
        result, wall = self._transcribe(_resample_16k(data, signal["sample_rate"]))

        segments = result.get("segments") or []
        transcript = (result.get("text") or "").strip()
        asr_tokens = tokenize(transcript)
        expected_tokens = tokenize(expected_text)
        mand_items = derive_mandatory(expected_tokens) if mandatory is None else mandatory
        metrics = {
            "coverage": ordered_coverage(expected_tokens, asr_tokens),
            "mandatory": check_mandatory(asr_tokens, mand_items),
            "confidence": _confidence(segments),
            "repetition": repetition_stats(asr_tokens),
            "leakage": leakage_check(asr_tokens, leakage_texts or []),
            "words": _word_stats(segments),
        }
        v, reasons = verdict(metrics)
        record = {
            "chunk_id": chunk_id,
            "wav_sha256": wav_sha,
            "expected_sha256": exp_sha,
            "asr": {"model_repo": self.model_repo, "model_revision": self.revision},
            "language": self.language,
            "transcript": transcript,
            "transcript_normalized": " ".join(asr_tokens),
            "asr_tokens": asr_tokens,
            "confidence": metrics["confidence"],
            "coverage": metrics["coverage"],
            "mandatory": metrics["mandatory"],
            "repetition": metrics["repetition"],
            "leakage": metrics["leakage"],
            "words": metrics["words"],
            "signal": signal,
            "verdict": v,
            "reasons": reasons,
            "asr_seconds": round(wall, 4),
            "rtf": round(wall / signal["seconds"], 4) if signal["seconds"] else None,
        }
        self._chunks += 1
        self._asr_seconds += wall
        self._audio_seconds += signal["seconds"]
        self._cache_put(key, record)
        return {**record, "cache_hit": False}

    def validate_many(self, specs: list, force: bool = False) -> list:
        """Validate a list of chunk specs; never raises per chunk.

        Each spec: {"wav", "expected" | "expected_text", "chunk_id"?,
        "mandatory"?, "leakage_texts"?}. Cache hits are reused unless
        force=True. Failures are recorded as verdict FAIL records with the
        error in reasons — chunks are flagged, never silently regenerated.
        """
        records = []
        for spec in specs:
            try:
                if not force:
                    wav = pathlib.Path(spec["wav"])
                    wav_sha = sha256_file(wav) if wav.is_file() else None
                    if wav_sha:
                        expected = spec.get("expected", spec.get("expected_text", ""))
                        exp_sha = sha256_bytes(expected.encode("utf-8"))
                        hit = self._cache_get(self._cache_key(wav_sha, exp_sha))
                        if hit is not None and hit.get("expected_sha256") == exp_sha:
                            records.append({**hit, "cache_hit": True})
                            continue
                records.append(self.validate_chunk(
                    spec["wav"],
                    spec.get("expected", spec.get("expected_text", "")),
                    chunk_id=spec.get("chunk_id"),
                    mandatory=spec.get("mandatory"),
                    leakage_texts=spec.get("leakage_texts"),
                ))
            except Exception as e:
                records.append({
                    "chunk_id": spec.get("chunk_id"),
                    "wav": str(spec.get("wav")),
                    "verdict": "FAIL",
                    "reasons": [f"{type(e).__name__}: {e}"],
                    "error": str(e),
                })
        return records

    def stats(self) -> dict:
        """Measured ASR cost for ETA: only transcription wall time counts
        (cache hits pay nothing)."""
        return {
            "chunks_validated": self._chunks,
            "asr_seconds": round(self._asr_seconds, 4),
            "audio_seconds": round(self._audio_seconds, 4),
            "rtf": round(self._asr_seconds / self._audio_seconds, 4)
            if self._audio_seconds else None,
            "load_seconds": round(self._load_seconds, 4) if self._load_seconds is not None else None,
            "model_repo": self.model_repo,
            "model_revision": self.revision,
        }

    def eta_estimate(self, remaining_audio_seconds: float) -> dict:
        """Projected ASR wall time for remaining audio at measured RTF."""
        s = self.stats()
        projected = remaining_audio_seconds * s["rtf"] if s["rtf"] else None
        return {
            "remaining_audio_seconds": remaining_audio_seconds,
            "projected_asr_seconds": round(projected, 1) if projected is not None else None,
            "rtf": s["rtf"],
            "note": None if s["rtf"] else "no ASR measurements yet (validate first)",
        }


def sf_read_16k(wav: pathlib.Path):
    import soundfile as sf

    return sf.read(str(wav), dtype="float32")[0]


# --- self-check (no model) ---------------------------------------------------
def selfcheck() -> int:
    """Unit-level normalization/alignment self-check. Pure stdlib except the
    numpy resampler check, which is skipped when numpy is unavailable. Never
    loads mlx-whisper or any model."""
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")

    print("audiobook.asr selfcheck")
    check("tokenize lowercases + strips punctuation",
          tokenize("The DEATH, of Tamerlane!") == ["the", "death", "of", "tamerlane"])
    check("number words -> digits", tokenize("fourteen oh five") == ["1405"])
    check("plain digits pass through", tokenize("1405") == ["1405"])
    check("hyphenated compound -> 21", tokenize("twenty-one") == ["21"])
    check("fifty -> 50", tokenize("fifty years") == ["50", "years"])
    check("apostrophes kept", tokenize("world's") == ["world's"])

    cov = ordered_coverage(["a", "b", "c"], ["x", "a", "b", "c"])
    check("coverage full", cov["fraction"] == 1.0 and cov["missing"] == [], repr(cov))
    cov = ordered_coverage(["a", "b", "c", "d"], ["a", "c", "d"])
    check("coverage missing b", cov["matched_tokens"] == 3 and cov["missing"] == ["b"], repr(cov))
    cov = ordered_coverage(["a", "b"], ["b", "a"])
    check("coverage ordered (no backtrack)", cov["matched_tokens"] == 1, repr(cov))
    cov = ordered_coverage([], ["anything"])
    check("coverage empty expected", cov["fraction"] == 1.0, repr(cov))

    check("mandatory fuzzy proper noun",
          check_mandatory(["the", "death", "of", "tamalane"], ["tamerlane"])["missing"] == [],
          repr(check_mandatory(["the", "death", "of", "tamalane"], ["tamerlane"])))
    check("mandatory absent flagged",
          check_mandatory(["the", "death"], ["tamerlane"])["missing"] == ["tamerlane"])
    check("mandatory phrase contiguous",
          check_mandatory(["genghis", "khan", "ruled"], ["genghis khan"])["missing"] == [])
    check("mandatory rejects scattered phrase",
          check_mandatory(["khan", "x", "genghis"], ["genghis khan"])["missing"] == ["genghis khan"])
    check("derive_mandatory content words",
          derive_mandatory(tokenize("The death of Tamerlane in 1405 was a turning point."))
          == ["tamerlane", "1405", "turning"],
          repr(derive_mandatory(tokenize("The death of Tamerlane in 1405 was a turning point."))))

    rep = repetition_stats(["thank", "you", "thank", "you", "thank", "you"])
    check("repetition loop detected", rep["max_multiplicity"] == 3, repr(rep))
    check("repetition clean sentence",
          repetition_stats(["the", "cat", "sat"])["repeated_count"] == 0)
    check("leakage flagged on overlap",
          leakage_check(tokenize("a b c d e"), ["a b c"])["flagged"] is True)
    check("leakage silent without texts",
          leakage_check(tokenize("a b c"), [])["flagged"] is False)

    good = {
        "coverage": {"fraction": 0.96, "matched_tokens": 24, "expected_tokens": 25, "missing": []},
        "mandatory": {"missing": []},
        "confidence": {"avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.1},
        "repetition": {"max_multiplicity": 1},
        "words": {"max_internal_gap_s": 0.4},
        "leakage": {"flagged": False},
    }
    check("verdict PASS on clean chunk", verdict(good)[0] == "PASS")
    bad = {**good, "coverage": {**good["coverage"], "fraction": 0.5}}
    check("verdict FAIL on truncated chunk", verdict(bad)[0] == "FAIL")

    def ck(**kw):
        return AsrValidator(**kw)._cache_key("W1", "E1")
    base = ck()
    variants = {
        ck(model_repo="other/model"),
        ck(revision="0" * 40),
        ck(language="fr"),
        ck(word_timestamps=False),
        AsrValidator()._cache_key("W2", "E1"),  # different wav hash
        AsrValidator()._cache_key("W1", "E2"),  # different expected hash
    }
    check("cache key separates all params",
          base not in variants and len(variants) == 6)

    try:
        import numpy as np  # noqa: F401
        sr = 24000
        t = np.arange(sr // 2) / sr
        x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        y = _resample_16k(x, sr)
        check("resample length 24k->16k", y.size == int(round(sr / 2 * 2 / 3)),
              f"{y.size} != {int(round(sr / 2 * 2 / 3))}")
        zc = int(np.sum(np.diff(np.signbit(y))))
        check("resample preserves 1 kHz", abs(zc - 1000) <= 6, f"zero crossings {zc}")
        check("16k pass-through", _resample_16k(x.astype(np.float32), 16000) is not None)
    except ImportError:
        print("  skip resample checks (numpy unavailable)")

    failed = [name for name, ok in results if not ok]
    print(f"selfcheck: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


# --- CLI ---------------------------------------------------------------------
def _load_specs(path: str) -> list:
    raw = json.loads(pathlib.Path(path).read_text())
    chunks = raw.get("chunks") if isinstance(raw, dict) else raw
    if not isinstance(chunks, list):
        raise ValueError(f"specs file {path} must be a JSON list or {{'chunks': [...]}}")
    return chunks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="audiobook.asr",
        description="Persistent mlx-whisper ASR validation for generated chunks.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selfcheck", help="run the no-model normalization/alignment self-check")

    p = sub.add_parser("validate", help="validate a single chunk")
    p.add_argument("--wav", required=True)
    p.add_argument("--expected", required=True, help="expected text for this chunk")
    p.add_argument("--chunk-id")
    p.add_argument("--cache", help="JSON cache file (keyed by model+language+timestamps+wav+expected hashes)")
    p.add_argument("--mandatory", help="comma-separated required tokens/phrases (default: derived)")
    p.add_argument("--leakage-texts", help="comma-separated texts to check for leakage")
    p.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    p.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--json", action="store_true", help="print the record as JSON")

    p = sub.add_parser("validate-batch", help="validate pending chunks from a specs file")
    p.add_argument("--specs", required=True,
                   help="JSON list or {'chunks': [...]} of {wav, expected|expected_text, chunk_id?}")
    p.add_argument("--cache")
    p.add_argument("--records-out", help="write all records to this JSON file (atomic)")
    p.add_argument("--force", action="store_true", help="ignore cache hits")
    p.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    p.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--json", action="store_true", help="print records as JSON")

    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return selfcheck()

    v = AsrValidator(
        model_repo=args.model_repo,
        revision=args.revision,
        cache_path=args.cache,
    )

    if args.cmd == "validate":
        try:
            record = v.validate_chunk(
                args.wav, args.expected, chunk_id=args.chunk_id,
                mandatory=args.mandatory.split(",") if args.mandatory else None,
                leakage_texts=args.leakage_texts.split("|") if args.leakage_texts else None,
            )
        except Exception as e:
            print(f"validate: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(record, indent=2))
        else:
            print(f"{record['chunk_id'] or record['wav_sha256'][:12]} "
                  f"{record['verdict']} coverage={record['coverage']['fraction']:.2f} "
                  f"rtf={record['rtf']} asr_s={record['asr_seconds']:.2f}")
            for reason in record["reasons"]:
                print(f"  - {reason}")
        return 0 if record["verdict"] == "PASS" else 1

    # validate-batch
    specs = _load_specs(args.specs)
    records = v.validate_many(specs, force=args.force)
    if args.records_out:
        out = pathlib.Path(args.records_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out, {
            "asr": {"model_repo": args.model_repo, "model_revision": args.revision},
            "records": records,
        })
    failed = [r for r in records if r["verdict"] != "PASS"]
    cached = sum(1 for r in records if r.get("cache_hit"))
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        print(f"chunks={len(records)} cached={cached} passed={len(records) - len(failed)} "
              f"failed={len(failed)}")
        print("asr_seconds={:.2f} audio_seconds={:.2f} rtf={}".format(
            v.stats()["asr_seconds"], v.stats()["audio_seconds"], v.stats()["rtf"]))
        for r in failed:
            print(f"  FAIL {r.get('chunk_id') or r.get('wav', '?')}: {r['reasons'][0]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
