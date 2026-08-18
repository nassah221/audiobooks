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
    mandatory: {items, missing, missing_fragile},
    terminal: {expected, matched},
    repetition: {max_multiplicity, most_repeated, repeated_count}
    leakage: {flagged, detail},
    words: {count, first_start, last_end, max_internal_gap_s, long_gaps_over_1s},
    word_timings: [{text, start, end}],
    punctuation: {boundaries, sentence_end, colon_semicolon,
                  parenthetical_comma, comma},
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

from . import epub

# --- frozen ASR model (default) ---------------------------------------------
# whisper-large-v3-turbo (weights.safetensors layout): loadable by pinned
# mlx-whisper 0.4.3. Proper-noun coverage is the deciding factor: whisper-tiny
# misread "Tamerlane" as "Tamil Nadu" and whisper-small as "Tamilaine",
# both blocking preflight; large-v3-turbo transcribes it exactly (13/13).
DEFAULT_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MODEL_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
DEFAULT_LANGUAGE = "en"
VALIDATION_POLICY = "paragraph-v10-abbreviations"

# --- verdict thresholds ------------------------------------------------------
COVERAGE_MIN = 0.85          # fraction of expected tokens found, in order
NO_SPEECH_MAX = 0.6          # whisper's own no-speech threshold
LOGPROB_MIN = -1.0           # whisper's own logprob threshold
COMPRESSION_MAX = 2.4        # whisper's own repetition/loop threshold
REPEAT_MAX_MULTIPLICITY = 3  # same adjacent n-gram (n>=2) seen this many times
MAX_INTERNAL_GAP_S = 2.5     # internal silence inside a spoken chunk
MANDATORY_FUZZY_RATIO = 0.8  # token match floor (difflib ratio) for mandatory
LEAKAGE_OVERLAP_MIN = 0.8    # token overlap with a leakage text that flags
FRAGILE_NUMBER_DIGITS_MAX = 2  # bare number word (0-99) mandatory items are
                                # ASR-fragile, not hard-mandatory (see is_fragile_mandatory)
SHORT_SENTENCE_MAX_WORDS = 2   # sentence_end boundaries closing a sentence
                                # this short use the colon_semicolon threshold

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
    "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30,
}
_ORDINAL_UNITS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
}
for _tens, _base in (("twenty", 20), ("thirty", 30)):
    for _word, _unit in _ORDINAL_UNITS.items():
        if _base + _unit <= 31:
            _ORDINAL_WORDS[f"{_tens}-{_word}"] = _base + _unit
            _ORDINAL_WORDS[f"{_tens} {_word}"] = _base + _unit
_ORDINAL_WORD_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(
        sorted(_ORDINAL_WORDS, key=len, reverse=True)
    ) + r")(?![a-z0-9])"
)
_NUMERIC_RANGE_RE = re.compile(r"(?<![a-z0-9])(\d+)-(\d+)(?![a-z0-9])")
_DECIMAL_RE = re.compile(r"(?<![a-z0-9])(\d+)\.(\d+)(?![a-z0-9])")
_NUMERIC_ORDINAL_RE = re.compile(r"(?<![a-z0-9])(\d+)(?:st|nd|rd|th)(?![a-z0-9])")
_DECADE_SUFFIXES = {
    "twenties": "20", "thirties": "30", "forties": "40", "fifties": "50",
    "sixties": "60", "seventies": "70", "eighties": "80", "nineties": "90",
    "20": "20", "30": "30", "40": "40", "50": "50", "60": "60",
    "70": "70", "80": "80", "90": "90",
}
_DECADE_PREFIXES = {str(n): str(n) for n in range(10, 20)}
_CENTURY_PREFIXES = {str(n): str(n * 100) for n in range(10, 20)}

# Bare decade word <-> digit-form equivalence, independent of any century
# prefix. "the eighteen thirties and forties" merges "eighteen thirties"
# into "1830s" (a prefix+suffix pair, see _merge_decade_tokens) but leaves
# the elliptical second decade "forties" as a standalone word -- Whisper may
# still render that one as a digit form ("40s"/"'40s"/"40's"). Both spellings
# canonicalize to the same "NNs" token so the match is equivalence-based, not
# a fragile exemption: the content is still required, just spelled either way.
_BARE_DECADE_WORDS = {k: v for k, v in _DECADE_SUFFIXES.items() if not k.isdigit()}
_BARE_DECADE_NUMERIC_RE = re.compile(r"^'?([2-9]0)'?s$")
_DECADE_TOKEN_RE = re.compile(r"^\d+0s$")  # any canonical decade token ("40s", "1830s", ...)


def _canonicalize_bare_decades(tokens: list) -> list:
    """Canonicalize a standalone decade word or digit rendering to "NNs".

    Runs after century/decade prefix merging, so an already-merged 4-digit
    token ("1830s") is untouched; only a decade left bare by ellipsis or
    written directly as digits gets normalized.
    """
    out = []
    for token in tokens:
        t = str(token)
        if t in _BARE_DECADE_WORDS:
            out.append(f"{_BARE_DECADE_WORDS[t]}s")
            continue
        m = _BARE_DECADE_NUMERIC_RE.fullmatch(t)
        out.append(f"{m.group(1)}s" if m else token)
    return out


def _merge_century_tokens(tokens: list) -> list:
    """Canonicalize spoken ``fourteen hundred`` and written ``1400``."""
    merged = []
    i = 0
    while i < len(tokens):
        if (i + 1 < len(tokens) and str(tokens[i]) in _CENTURY_PREFIXES
                and str(tokens[i + 1]) == "hundred"
                and (i + 2 == len(tokens) or not str(tokens[i + 2]).isdigit())):
            merged.append(_CENTURY_PREFIXES[str(tokens[i])])
            i += 2
            continue
        merged.append(tokens[i])
        i += 1
    return merged

_DECADE_NUMERIC_RE = re.compile(r"^(1[0-9])(\d0)(?:'s|s)$")

def _merge_decade_tokens(tokens: list) -> list:
    """Canonicalize spoken ``thirteen thirties`` and written ``1330s``."""
    merged = []
    i = 0
    while i < len(tokens):
        token = str(tokens[i])
        numeric = _DECADE_NUMERIC_RE.fullmatch(token)
        if numeric:
            merged.append(f"{numeric.group(1)}{numeric.group(2)}s")
            i += 1
            continue
        if i + 1 < len(tokens):
            prefix, suffix = token, str(tokens[i + 1])
            suffix = suffix.removesuffix("'s")
            if prefix in _DECADE_PREFIXES and suffix in _DECADE_SUFFIXES:
                merged.append(f"{prefix}{_DECADE_SUFFIXES[suffix]}s")
                i += 2
                continue
        merged.append(tokens[i])
        i += 1
    return merged

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


def tokenize(text: str) -> list:
    """Normalize spoken and written text into comparable tokens."""
    t = text.lower().replace("\u2019", "'")
    t = _DECIMAL_RE.sub(r"\1 point \2", t)
    t = _NUMERIC_RANGE_RE.sub(r"\1 to \2", t)
    t = _DECADE_NUMERIC_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}s", t)

    def ordinal(m):
        n = _ORDINAL_WORDS[m.group(1)]
        suffix = "th" if n % 10 not in (1, 2, 3) or n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}[n % 10]
        return f"{n}{suffix}"

    t = _ORDINAL_WORD_RE.sub(ordinal, t)
    t = _NUMERIC_ORDINAL_RE.sub(lambda m: m.group(1) + m.group(0)[-2:], t)
    t = _TENS_COMPOUND_RE.sub(
        lambda m: str(int(_NUM_WORDS[m.group(1)]) + int(_NUM_WORDS[m.group(2)])), t
    )
    t = re.sub(r"[^a-z0-9']+", " ", t)
    out = []
    for tok in t.split():
        if tok in _NUM_WORDS:
            out.append(_NUM_WORDS[tok])
        elif tok == "oh":
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
    return _canonicalize_bare_decades(_merge_century_tokens(_merge_decade_tokens(merged)))




PUNCTUATION_THRESHOLDS = {
    "sentence_end": 0.150,
    "colon_semicolon": 0.100,
    "parenthetical_comma": 0.075,
    "comma": None,
}


def normalized_word_timings(segments: list) -> list:
    """Return JSON-native normalized tokens with their Whisper timings."""
    out = []
    for segment in segments or []:
        for word in segment.get("words") or []:
            if "start" not in word or "end" not in word:
                continue
            start, end = float(word["start"]), float(word["end"])
            tokens = tokenize(str(word.get("word", word.get("text", ""))))
            for text in tokens:
                if text.isdigit() and out and out[-1]["text"].isdigit():
                    out[-1]["text"] += text
                    out[-1]["end"] = end
                else:
                    out.append({"text": text, "start": start, "end": end})
    return out


_INITIALS_RE = re.compile(r"(?:[A-Z]\.)+")


def _is_sentence_abbreviation(word: str) -> bool:
    """True when `word` (the text up to and including a "." candidate) is a
    known abbreviation ("i.e.", "e.g.", ...) or an initials run ("U.S."),
    not a genuine sentence end. Reuses epub.sentence_spans's own
    abbreviation set (`epub._ABBREVIATIONS`) and initials pattern -- the
    planner's sentence splitter and this ASR boundary scanner must agree on
    what counts as a sentence, or the scanner flags a pause the source text
    never asked for (e.g. inside "(i.e." there is no sentence break, so
    requiring one fails the take by construction)."""
    stripped = word.lstrip("([{\"'‘“")
    return stripped.lower() in epub._ABBREVIATIONS or bool(_INITIALS_RE.fullmatch(stripped))


def _source_punctuation_boundaries(source: str) -> list:
    """Find punctuation boundaries and their preceding normalized token.

    Each `sentence_end` boundary also records `sentence_word_count`: the
    number of tokens in the sentence it closes (since the previous
    sentence_end boundary, or the start of the source). A one- or two-word
    sentence -- an enumerated-list marker like "one." -- is read with a
    shorter pause than ordinary prose, so `punctuation_metrics` relaxes its
    threshold when this count is low (see SHORT_SENTENCE_MAX_WORDS).

    A "." only starts a `sentence_end` candidate when it ends its
    whitespace-delimited token (mirroring epub.sentence_spans's `(?=\\s|$)`
    lookahead) and that token is not a known abbreviation -- otherwise an
    abbreviation like "i.e." would spuriously split into two boundaries at
    its own internal periods.
    """
    boundaries = []
    prev_sentence_end_index = -1
    for chunk_match in re.finditer(r"\S+", source):
        chunk = chunk_match.group(0)
        chunk_tokens = tokenize(chunk)
        if not chunk_tokens:
            continue
        for punctuation in re.finditer(r"[.?!:;,]+", chunk):
            mark = punctuation.group(0)
            before = chunk[:punctuation.start()]
            preceding = tokenize(source[:chunk_match.start()] + before)
            if not preceding:
                continue
            if mark[0] == ".":
                pos = punctuation.start()
                if pos and pos + 1 < len(chunk) and chunk[pos - 1].isdigit() and chunk[pos + 1].isdigit():
                    continue
                if punctuation.end() != len(chunk):
                    continue  # not sentence-final in this token (e.g. "i." inside "i.e.")
                if _is_sentence_abbreviation(chunk[:punctuation.end()]):
                    continue
                kind = "sentence_end"
            elif mark[0] in ":;":
                kind = "colon_semicolon"
            elif mark[0] == "," and before.endswith(")"):
                kind = "parenthetical_comma"
            elif mark[0] == ",":
                kind = "comma"
            else:
                continue
            token_index = len(preceding) - 1
            boundary = {
                "kind": kind,
                "punctuation": mark[0],
                "expected_token_index": token_index,
            }
            if kind == "sentence_end":
                boundary["sentence_word_count"] = token_index - prev_sentence_end_index
                prev_sentence_end_index = token_index
            boundaries.append(boundary)
    return boundaries


def _fuzzy_alignment(expected: list, asr: list, ratio: float = MANDATORY_FUZZY_RATIO) -> dict:
    """Map expected tokens to ASR positions with ordered fuzzy LCS."""
    n, m = len(expected), len(asr)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = (dp[i + 1][j + 1] + 1 if _tok_eq(expected[i], asr[j], ratio)
                        else max(dp[i + 1][j], dp[i][j + 1]))
    mapping = {}
    i = j = 0
    while i < n and j < m:
        if _tok_eq(expected[i], asr[j], ratio) and dp[i][j] == dp[i + 1][j + 1] + 1:
            mapping[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return mapping


def punctuation_metrics(source: str, word_timings: list) -> dict:
    """Measure source punctuation pauses against ordered ASR word timings."""
    expected = tokenize(source)
    asr = [str(word["text"]) for word in word_timings or []]
    mapping = _fuzzy_alignment(expected, asr)
    boundaries = []
    groups = {kind: [] for kind in PUNCTUATION_THRESHOLDS}
    for boundary in _source_punctuation_boundaries(source):
        kind = boundary["kind"]
        idx = boundary["expected_token_index"]
        before_asr = mapping.get(idx)
        after_asr = mapping.get(idx + 1)
        aligned = before_asr is not None and after_asr is not None
        gap = None
        if aligned:
            gap = round(float(word_timings[after_asr]["start"]) - float(word_timings[before_asr]["end"]), 4)
        threshold = PUNCTUATION_THRESHOLDS[kind]
        if (kind == "sentence_end"
                and boundary.get("sentence_word_count", 0) <= SHORT_SENTENCE_MAX_WORDS):
            # a one- or two-word sentence (e.g. a list marker "one.") is a
            # natural short pause, not full prose -- use the colon/semicolon
            # tier instead of the standard sentence_end minimum.
            threshold = PUNCTUATION_THRESHOLDS["colon_semicolon"]
        passed = None if not aligned or threshold is None else gap >= threshold
        measured = {
            **boundary,
            "asr_token_index": before_asr,
            "next_asr_token_index": after_asr,
            "aligned": aligned,
            "gap_s": gap,
            "threshold_s": threshold,
            "passed": passed,
        }
        boundaries.append(measured)
        groups[kind].append(measured)

    def summary(items):
        return {
            "checked": len(items),
            "aligned": sum(1 for item in items if item["aligned"]),
            "passed": sum(1 for item in items if item["passed"] is True),
            "failed": sum(1 for item in items if item["passed"] is False),
            "unaligned": sum(1 for item in items if not item["aligned"]),
            "gaps_s": [item["gap_s"] for item in items if item["gap_s"] is not None],
        }

    return {
        "boundaries": boundaries,
        **{kind: summary(items) for kind, items in groups.items()},
    }
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


def is_fragile_mandatory(phrase: list) -> bool:
    """A mandatory item is ASR-fragile when it is a single bare number word/
    digit (0-99) after canonicalization, e.g. an enumerated-list marker
    ("1." from "one."). Whisper often mishears an isolated number word --
    "won", merged into the next word -- so, like a proper name the take
    otherwise renders correctly, its absence alone must not fail the take.
    Multi-digit numbers (years, page counts) stay hard-mandatory: whisper
    renders those reliably (see the century-equivalence handling above)."""
    return (len(phrase) == 1 and phrase[0].isdigit()
            and len(phrase[0]) <= FRAGILE_NUMBER_DIGITS_MAX)


def check_mandatory(asr: list, mandatory: list, ratio: float = MANDATORY_FUZZY_RATIO) -> dict:
    """Each mandatory item (string or token list) must appear as a contiguous,
    fuzzy-tolerant run in the ASR tokens. Items flagged `is_fragile_mandatory`
    (bare number-word markers) are reported in `missing_fragile` instead of
    `missing`: they still count toward `coverage` (computed separately over
    every expected token), but their absence alone never fails the verdict."""
    items = []
    missing = []
    missing_fragile = []
    for item in mandatory:
        raw_phrase = item if isinstance(item, str) else " ".join(map(str, item))
        phrase = tokenize(raw_phrase)
        if not phrase:
            continue
        label = item if isinstance(item, str) else " ".join(map(str, item))
        items.append(label)
        if not _phrase_in(asr, phrase, ratio):
            (missing_fragile if is_fragile_mandatory(phrase) else missing).append(label)
    return {"items": items, "missing": missing, "missing_fragile": missing_fragile}


def terminal_suffix(expected: list, asr: list, size: int = 5) -> dict:
    """Require the final expected phrase near the end of the transcript."""
    tail = expected[-size:]
    window = asr[-(len(tail) + 2):]
    return {
        "expected": " ".join(tail),
        "matched": not tail or _phrase_in(window, tail, MANDATORY_FUZZY_RATIO),
    }


def derive_mandatory(expected_tokens: list) -> list:
    """Default mandatory content: digits plus content words (len >= 6, not
    function words) from the expected text, plus canonical decade tokens
    ("40s", "1830s", ...) -- alnum, so neither the digit nor the alpha branch
    would otherwise catch them, but a decade is real content just like a
    year. Callers may pass an explicit `mandatory` list to override.
    `check_mandatory` (not this function) decides which of these items are
    hard-mandatory vs. ASR-fragile bare number words -- both are returned
    here so fragile items still count toward the returned candidate set."""
    out = []
    seen = set()
    for tok in expected_tokens:
        if tok in seen:
            continue
        keep = (tok.isdigit() or _DECADE_TOKEN_RE.fullmatch(tok)
                or (len(tok) >= 6 and tok.isalpha() and tok not in _FUNCTION_WORDS))
        if keep:
            seen.add(tok)
            out.append(tok)
    return out


def repetition_stats(tokens: list, expected=()) -> dict:
    """Report transcript multiplicity for n-grams beyond expected counts."""
    counts = Counter()
    expected_counts = Counter()
    for n in (2, 3, 4):
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i:i + n])] += 1
        for i in range(len(expected) - n + 1):
            expected_counts[tuple(expected[i:i + n])] += 1
    repeated = {k: v for k, v in counts.items()
                if v > 1 and v > expected_counts[k]}
    if not repeated:
        return {"max_multiplicity": 0, "most_repeated": None, "repeated_count": 0}
    best = max(repeated, key=repeated.get)
    return {
        "max_multiplicity": repeated[best],
        "most_repeated": " ".join(best),
        "repeated_count": len(repeated),
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
    timings = normalized_word_timings(segments)
    words = [(float(w["start"]), float(w["end"])) for w in timings]
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
    terminal = metrics["terminal"]
    if not terminal["matched"]:
        reasons.append(f"terminal phrase missing: {terminal['expected']!r}")
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
    punctuation = metrics.get("punctuation", {})
    for kind in ("sentence_end", "colon_semicolon", "parenthetical_comma"):
        for boundary in punctuation.get("boundaries", []):
            if boundary.get("kind") == kind and boundary.get("passed") is False:
                reasons.append(
                    f"{kind} pause {boundary['gap_s']:.3f}s < {boundary['threshold_s']:.3f}s"
                )
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
            VALIDATION_POLICY,
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
        if (cached is not None and cached.get("expected_sha256") == exp_sha
                and isinstance(cached.get("terminal"), dict)
                and isinstance(cached.get("punctuation"), dict)
                and isinstance(cached.get("word_timings"), list)):
            return {**cached, "cache_hit": True}

        signal = _signal_facts(wav)
        import numpy as np

        data = np.asarray(sf_read_16k(wav), dtype=np.float32)
        result, wall = self._transcribe(_resample_16k(data, signal["sample_rate"]))

        segments = result.get("segments") or []
        transcript = (result.get("text") or "").strip()
        asr_tokens = tokenize(transcript)
        expected_tokens = tokenize(expected_text)
        word_timings = normalized_word_timings(segments)
        mand_items = derive_mandatory(expected_tokens) if mandatory is None else mandatory
        metrics = {
            "coverage": ordered_coverage(expected_tokens, asr_tokens),
            "mandatory": check_mandatory(asr_tokens, mand_items),
            "terminal": terminal_suffix(expected_tokens, asr_tokens),
            "confidence": _confidence(segments),
            "repetition": repetition_stats(asr_tokens, expected_tokens),
            "leakage": leakage_check(asr_tokens, leakage_texts or []),
            "words": _word_stats(segments),
            "punctuation": punctuation_metrics(expected_text, word_timings),
        }
        v, reasons = verdict(metrics)
        record = {
            "chunk_id": chunk_id,
            "wav_sha256": wav_sha,
            "expected_sha256": exp_sha,
            "asr": {"model_repo": self.model_repo, "model_revision": self.revision},
            "validation_policy": VALIDATION_POLICY,
            "language": self.language,
            "transcript": transcript,
            "transcript_normalized": " ".join(asr_tokens),
            "asr_tokens": asr_tokens,
            "confidence": metrics["confidence"],
            "coverage": metrics["coverage"],
            "terminal": metrics["terminal"],
            "words": metrics["words"],
            "word_timings": word_timings,
            "punctuation": metrics["punctuation"],
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
                        if (hit is not None and hit.get("expected_sha256") == exp_sha
                                and isinstance(hit.get("terminal"), dict)
                                and isinstance(hit.get("punctuation"), dict)
                                and isinstance(hit.get("word_timings"), list)):
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
                    "validation_error": True,
                    "terminal": {"expected": None, "matched": False},
                    "word_timings": [],
                    "punctuation": None,
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

    check("spoken century equals written year",
          tokenize("fourteen hundred") == tokenize("1400") == ["1400"])
    check("century equivalence reaches coverage",
          ordered_coverage(tokenize("in fourteen hundred"),
                           tokenize("in 1400"))["missing"] == [])
    check("century equivalence reaches mandatory",
          check_mandatory(tokenize("in 1400"), ["fourteen hundred"])["missing"] == [])
    check("century equivalence reaches terminal",
          terminal_suffix(tokenize("before fourteen hundred"),
                          tokenize("before 1400"))["matched"])
    check("different century remains distinct",
          tokenize("fourteen hundred") != tokenize("fifteen hundred"))
    check("century count continuation remains distinct",
          tokenize("fourteen hundred one") != tokenize("1400"))
    check("non-century prefix remains distinct",
          tokenize("twenty hundred") != tokenize("2000"))
    check("plural hundred remains distinct",
          tokenize("fourteen hundreds") != tokenize("1400"))
    check("tokenize lowercases + strips punctuation",
          tokenize("The DEATH, of Tamerlane!") == ["the", "death", "of", "tamerlane"])
    check("number words -> digits", tokenize("fourteen oh five") == ["1405"])
    check("plain digits pass through", tokenize("1405") == ["1405"])
    check("hyphenated compound -> 21", tokenize("twenty-one") == ["21"])
    check("fifty -> 50", tokenize("fifty years") == ["50", "years"])

    check("spoken and written ordinal equivalent",
          tokenize("fifteenth") == tokenize("15th") and
          tokenize("twentieth") == tokenize("20th"))
    check("spoken decade equals written decade",
          tokenize("thirteen thirties") == tokenize("1330s") == ["1330s"])
    check("decade apostrophe variant equals written decade",
          tokenize("thirteen-thirties") == tokenize("1330's") == ["1330s"])
    check("nearby decade remains distinct",
          tokenize("thirteen forties") != tokenize("1330s"))
    check("decade equivalence reaches coverage",
          ordered_coverage(tokenize("born in thirteen thirties"),
                           tokenize("born in 1330s"))["missing"] == [])
    check("decade equivalence reaches mandatory",
          check_mandatory(tokenize("born in 1330s"), ["thirteen thirties"])["missing"] == [])
    check("decade equivalence reaches terminal",
          terminal_suffix(tokenize("born in thirteen thirties"),
                          tokenize("born in 1330s"))["matched"])

    # Bare decade word <-> digit-form equivalence (no century prefix), e.g.
    # an elliptical list "the eighteen thirties and forties" where the
    # second decade is never paired with its own prefix. This is a real
    # matching equivalence (like century/decade above): the mandatory item
    # still has to be present, just spelled either way -- not a fragile
    # exemption.
    check("bare decade word equals its digit form",
          tokenize("forties") == tokenize("40s") == ["40s"])
    check("bare decade apostrophe variants equal the digit form",
          tokenize("'40s") == tokenize("40's") == tokenize("40s") == ["40s"])
    check("nearby bare decade remains distinct",
          tokenize("thirties") != tokenize("40s"))
    check("elliptical decade list keeps the merged prefix decade distinct",
          tokenize("the eighteen thirties and forties") ==
          ["the", "1830s", "and", "40s"],
          repr(tokenize("the eighteen thirties and forties")))
    check("bare decade word reaches mandatory as its digit form",
          check_mandatory(tokenize("the 40s were turbulent"), ["forties"])["missing"] == [])
    check("bare decade digit form reaches mandatory as the word",
          check_mandatory(tokenize("the forties were turbulent"), ["40s"])["missing"] == [])
    check("unrelated missing content word still hard-fails",
          check_mandatory(tokenize("nothing relevant here"),
                          ["catastrophe"])["missing"] == ["catastrophe"])
    decade_expected = tokenize("the eighteen thirties and forties were turbulent")
    decade_asr = tokenize("the eighteen thirties and 40s were turbulent")
    check("elliptical decade paragraph is fully mandatory-covered despite the digit rendering",
          check_mandatory(decade_asr, derive_mandatory(decade_expected))["missing"] == [],
          repr(check_mandatory(decade_asr, derive_mandatory(decade_expected))))

    check("ordinal range through thirty-first",
          tokenize("twenty-first") == tokenize("21st") and
          tokenize("thirty-first") == tokenize("31st"))
    check("numeric range stays ordered",
          tokenize("thirteen thirty-two to fourteen oh six") == tokenize("1332-1406"))
    check("decimal spoken form equivalent",
          tokenize("1.3") == tokenize("one point three"))
    check("hyphenated words stay separate",
          tokenize("well-known state-of-the-art") == ["well", "known", "state", "of", "the", "art"])
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

    # bare number-word markers (enumerated-list items like "one.") are
    # ASR-fragile: Whisper often mishears an isolated number word, so their
    # absence alone must not fail the take, but they are reported (and still
    # count toward the ordered `coverage` check computed separately).
    check("bare number word is fragile", is_fragile_mandatory(tokenize("one")))
    check("multi-digit number word is not fragile (e.g. a year)",
          not is_fragile_mandatory(tokenize("fourteen oh five")))
    check("multi-token phrase is not fragile",
          not is_fragile_mandatory(tokenize("genghis khan")))
    list_marker_mand = check_mandatory(tokenize("won the appearance of a single global market"),
                                       derive_mandatory(tokenize(
                                           "one. the appearance of a single global market")))
    check("missing bare number marker is not hard-mandatory",
          list_marker_mand["missing"] == [], repr(list_marker_mand))
    check("missing bare number marker is recorded as fragile",
          list_marker_mand["missing_fragile"] == ["1"], repr(list_marker_mand))
    list_marker_exp = tokenize("one. the appearance of a single global market")
    list_marker_asr = tokenize("won the appearance of a single global market")
    list_marker_verdict = verdict({
        "coverage": ordered_coverage(list_marker_exp, list_marker_asr),
        "mandatory": check_mandatory(list_marker_asr, derive_mandatory(list_marker_exp)),
        "terminal": terminal_suffix(list_marker_exp, list_marker_asr),
        "confidence": {}, "repetition": {"max_multiplicity": 0},
        "words": {"max_internal_gap_s": 0}, "leakage": {"flagged": False},
        "punctuation": {"boundaries": []},
    })
    check("list-marker paragraph passes despite missing bare number word",
          list_marker_verdict[0] == "PASS", repr(list_marker_verdict))

    check("terminal suffix present",
          terminal_suffix(tokenize("the history attempts to explain"),
                          tokenize("the history attempts to explain"))["matched"])
    check("terminal suffix missing",
          not terminal_suffix(tokenize("the history attempts to explain"),
                              tokenize("explain appears earlier but history attempts to"))["matched"])
    check("derive_mandatory content words",
          derive_mandatory(tokenize("The death of Tamerlane in 1405 was a turning point."))
          == ["tamerlane", "1405", "turning"],
          repr(derive_mandatory(tokenize("The death of Tamerlane in 1405 was a turning point."))))

    check("leakage flagged on overlap",
          leakage_check(tokenize("a b c d e"), ["a b c"])["flagged"] is True)
    check("leakage silent without texts",
          leakage_check(tokenize("a b c"), [])["flagged"] is False)
    timing_segments = [{"words": [
        {"word": "Fourteen", "start": 0.0, "end": 0.2},
        {"word": "oh", "start": 0.21, "end": 0.3},
        {"word": "five:", "start": 0.4, "end": 0.5},
        {"word": "Tamerlane", "start": 0.61, "end": 0.8},
        {"word": "returns", "start": 0.95, "end": 1.1},
    ]}]
    timings = normalized_word_timings(timing_segments)
    check("normalized timing entries are JSON-native",
          timings == [
              {"text": "1405", "start": 0.0, "end": 0.5},
              {"text": "tamerlane", "start": 0.61, "end": 0.8},
              {"text": "returns", "start": 0.95, "end": 1.1},
          ] and json.loads(json.dumps(timings)) == timings)
    source = "Alpha ends. Beta follows: Gamma (aside), delta, quiet."
    words = [
        {"text": text, "start": i * 0.3, "end": i * 0.3 + 0.1}
        for i, text in enumerate(tokenize(source))
    ]
    punctuation = punctuation_metrics(source, words)
    check("sentence punctuation pause passes",
          punctuation["sentence_end"]["passed"] == 1 and
          punctuation["sentence_end"]["unaligned"] == 1)
    check("colon punctuation pause passes",
          punctuation["colon_semicolon"]["passed"] == 1)
    check("parenthetical comma pause passes",
          punctuation["parenthetical_comma"]["passed"] == 1)
    check("ordinary comma is advisory",
          punctuation["comma"]["checked"] == 1 and punctuation["comma"]["failed"] == 0)

    # An enumerated-list marker ("one.") is a one-word sentence read with a
    # shorter, natural pause -- its sentence_end boundary uses the
    # colon_semicolon (100ms) tier. An ordinary multi-word sentence keeps
    # the standard 150ms minimum.
    list_marker_source = "one. the appearance of a single global market."
    list_marker_words = [
        {"text": text, "start": i * 0.3, "end": i * 0.3 + 0.1}
        for i, text in enumerate(tokenize(list_marker_source))
    ]
    list_marker_punct = punctuation_metrics(list_marker_source, list_marker_words)
    sentence_boundaries = [b for b in list_marker_punct["boundaries"] if b["kind"] == "sentence_end"]
    check("one-word sentence boundary uses the 100ms threshold",
          sentence_boundaries[0]["sentence_word_count"] == 1 and
          sentence_boundaries[0]["threshold_s"] == PUNCTUATION_THRESHOLDS["colon_semicolon"],
          repr(sentence_boundaries[0]))
    check("normal multi-word sentence boundary keeps the 150ms threshold",
          sentence_boundaries[1]["sentence_word_count"] > SHORT_SENTENCE_MAX_WORDS and
          sentence_boundaries[1]["threshold_s"] == PUNCTUATION_THRESHOLDS["sentence_end"],
          repr(sentence_boundaries[1]))

    # An abbreviation like "i.e." must not spawn a false sentence_end at its
    # own internal period -- the planner's sentence splitter (epub.py) never
    # treats it as a sentence break, so the ASR boundary scanner must agree
    # (see epub._ABBREVIATIONS, the shared source of truth).
    abbrev_source = ("This is modernization (i.e. the replication of "
                      "structure). Both attitudes had it.")
    abbrev_boundaries = [b for b in _source_punctuation_boundaries(abbrev_source)
                         if b["kind"] == "sentence_end"]
    check("abbreviation period produces no sentence_end boundary",
          len(abbrev_boundaries) == 2, repr(abbrev_boundaries))
    abbrev_tokens = tokenize(abbrev_source)
    check("abbreviation is not among the sentence_end boundary words",
          all(abbrev_tokens[b["expected_token_index"]] not in ("i", "e")
              for b in abbrev_boundaries),
          repr([abbrev_tokens[b["expected_token_index"]] for b in abbrev_boundaries]))
    check("a real sentence end after an ordinary word still boundaries",
          abbrev_tokens[abbrev_boundaries[0]["expected_token_index"]] == "structure" and
          abbrev_tokens[abbrev_boundaries[1]["expected_token_index"]] == "it",
          repr([abbrev_tokens[b["expected_token_index"]] for b in abbrev_boundaries]))
    initials_source = "He worked for the U.S. government."
    initials_boundaries = [b for b in _source_punctuation_boundaries(initials_source)
                           if b["kind"] == "sentence_end"]
    check("initials abbreviation (U.S.) produces exactly one real sentence end",
          len(initials_boundaries) == 1, repr(initials_boundaries))

    short_words = words[:2] + words[3:]
    diagnostic = punctuation_metrics(source, short_words)
    check("unaligned punctuation stays diagnostic",
          diagnostic["sentence_end"]["unaligned"] > 0 and
          diagnostic["sentence_end"]["failed"] == 0)
    check("short sentence pause fails verdict",
          verdict({"coverage": {"fraction": 1, "matched_tokens": 1, "expected_tokens": 1, "missing": []},
                   "mandatory": {"missing": []}, "terminal": {"matched": True, "expected": ""},
                   "confidence": {}, "repetition": {"max_multiplicity": 0},
                   "words": {"max_internal_gap_s": 0}, "leakage": {"flagged": False},
                   "punctuation": {"boundaries": [{"kind": "sentence_end", "gap_s": 0.1,
                                                     "threshold_s": 0.15, "passed": False}]}})[0] == "FAIL")

    good = {
        "coverage": {"fraction": 0.96, "matched_tokens": 24, "expected_tokens": 25, "missing": []},
        "mandatory": {"missing": []},
        "terminal": {"expected": "attempts to explain", "matched": True},
        "confidence": {"avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.1},
        "repetition": {"max_multiplicity": 1},
        "words": {"max_internal_gap_s": 0.4},
        "leakage": {"flagged": False},
    }
    rep = repetition_stats(tokenize("thank you thank you thank you"), tokenize("thank you"))
    check("repetition x3 vs expected once fails",
          rep["max_multiplicity"] == 3 and
          verdict({**good, "repetition": rep})[0] == "FAIL", repr(rep))
    check("expected repetition accepted",
          repetition_stats(tokenize("thank you thank you thank you"),
                           tokenize("thank you thank you thank you"))["max_multiplicity"] == 0)
    schema_terminal = terminal_suffix(tokenize("alpha omega"), tokenize("alpha omega"))
    schema_punctuation = punctuation_metrics("Alpha ends. Beta follows.", [
        {"text": text, "start": i * 0.4, "end": i * 0.4 + 0.1}
        for i, text in enumerate(tokenize("Alpha ends Beta follows"))
    ])
    schema_record = {"terminal": schema_terminal, "word_timings": timings,
                     "punctuation": schema_punctuation, "verdict": "PASS"}
    serialized = json.loads(json.dumps(schema_record))
    check("serialized record preserves terminal matched state",
          serialized["terminal"] == {"expected": "alpha omega", "matched": True})
    missing_terminal = terminal_suffix(tokenize("alpha omega"), tokenize("alpha"))
    check("serialized record preserves terminal missing state",
          json.loads(json.dumps({"terminal": missing_terminal}))["terminal"]["matched"] is False)
    check("serialized record preserves exact punctuation schema",
          set(serialized["punctuation"]) == {"boundaries", "sentence_end", "colon_semicolon",
                                               "parenthetical_comma", "comma"} and
          set(serialized["punctuation"]["boundaries"][0]) == {
              "kind", "punctuation", "expected_token_index", "sentence_word_count",
              "asr_token_index", "next_asr_token_index", "aligned", "gap_s",
              "threshold_s", "passed"})
    check("record fields remain JSON-native", serialized == schema_record)
    check("repetition clean sentence",
          repetition_stats(["the", "cat", "sat"])["repeated_count"] == 0)
    check("verdict PASS on clean chunk", verdict(good)[0] == "PASS")
    bad = {**good, "coverage": {**good["coverage"], "fraction": 0.5}}
    check("verdict FAIL on truncated chunk", verdict(bad)[0] == "FAIL")
    bad = {**good, "terminal": {"expected": "attempts to explain", "matched": False}}
    check("verdict FAIL on missing terminal phrase", verdict(bad)[0] == "FAIL")

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
