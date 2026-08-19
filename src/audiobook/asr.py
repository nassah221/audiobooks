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
    mandatory: {items, missing, missing_fragile, phonetic_matches, hyphen_matches,
                vowel_matches, transliteration_matches},
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
import unicodedata
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
VALIDATION_POLICY = "paragraph-v33-asr-fragile-lexicon"

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
# "N hundred" multiplies by 100 for two distinct, equally idiomatic English
# uses: a century/year reference ("fourteen hundred" = 1400, teens 10-19)
# and a plain quantity ("three hundred years" = 300, single digits 1-9).
# A non-teen two-digit prefix ("twenty hundred") is neither -- English says
# "two thousand" instead -- so it stays out of this table and un-merged.
_CENTURY_PREFIXES = {str(n): str(n * 100) for n in range(1, 20)}

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
_HYPHENATED_WORD_RE = re.compile(r"\b[a-zA-Z]+-[a-zA-Z]+(?:'s)?\b")


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
    """Canonicalize spoken ``fourteen hundred``/``1400`` and spoken
    ``three hundred``/``300`` -- Whisper renders both a century and a plain
    hundred-quantity as digits, so the mandatory/coverage checks need them
    to agree regardless of which the source text spelled out."""
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


def _merge_hundreds_decade_tokens(tokens: list) -> list:
    """Canonicalize spoken ``four hundreds``/written ``400s``: a
    century-block decade reference (the 5th-century era, 400-499), the
    plural sibling of the plain ``four hundred``/``400`` merge above.
    epub.expand_numbers renders "400s" as "four hundreds" when speaking the
    book, so Whisper's preferred digit rendering ("the 400s") needs the
    same token as the source text's spelled-out plural."""
    merged = []
    i = 0
    while i < len(tokens):
        if (i + 1 < len(tokens) and str(tokens[i]) in _CENTURY_PREFIXES
                and str(tokens[i + 1]) == "hundreds"
                and (i + 2 == len(tokens) or not str(tokens[i + 2]).isdigit())):
            merged.append(f"{_CENTURY_PREFIXES[str(tokens[i])]}s")
            i += 2
            continue
        merged.append(tokens[i])
        i += 1
    return merged


def _merge_thousand_tokens(tokens: list) -> list:
    """Canonicalize spoken ``one thousand``/written ``1000``: any digit
    token (a number word, or an already-merged hundred value, e.g. ``six
    hundred`` -> ``600``) followed by "thousand" multiplies by 1000, the
    same pattern as the century/hundred merges above. Unlike "N hundred"
    (where a non-teen two-digit prefix like "twenty hundred" is not
    idiomatic), "N thousand" is unambiguous English for any N, so every
    digit prefix qualifies.

    A directly following hundred-block ("two thousand five hundred" = 2500,
    already merged to "2", "thousand", "500" by the century pass above)
    adds on too -- both parts are round scale units, so unlike a bare
    remainder ("fourteen hundred one"), no "and" is needed to disambiguate.
    Any other trailing digit (not a clean hundred-block) still blocks the
    merge, the same guard as before, e.g. "one thousand one" stays distinct
    from "1000" or "1001"."""
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and str(tokens[i]).isdigit() and str(tokens[i + 1]) == "thousand":
            nxt = tokens[i + 2] if i + 2 < len(tokens) else None
            if nxt is None:
                merged.append(str(int(tokens[i]) * 1000))
                i += 2
                continue
            if str(nxt).isdigit() and 0 < int(nxt) < 1000 and int(nxt) % 100 == 0:
                merged.append(str(int(tokens[i]) * 1000 + int(nxt)))
                i += 3
                continue
            if str(nxt).isdigit():
                merged.append(tokens[i])
                i += 1
                continue
            merged.append(str(int(tokens[i]) * 1000))
            i += 2
            continue
        merged.append(tokens[i])
        i += 1
    return merged


def _merge_century_remainder_tokens(tokens: list) -> list:
    """Canonicalize the classic English year idiom "eight hundred and
    forty-three" (843) / "nineteen hundred and eighty-four" (1984): an
    explicit "and" plus a 1-99 remainder adds onto an already-merged
    hundred/century/thousand value (any multiple of 100). The "and" is the
    disambiguating signal -- without it ("fourteen hundred one") this is
    not idiomatic English for a single number, so _merge_century_tokens's
    own guard already leaves that case unmerged before this pass runs."""
    merged = []
    i = 0
    while i < len(tokens):
        if (i + 2 < len(tokens) and str(tokens[i]).isdigit()
                and int(tokens[i]) > 0 and int(tokens[i]) % 100 == 0
                and str(tokens[i + 1]) == "and"
                and str(tokens[i + 2]).isdigit() and 1 <= int(tokens[i + 2]) <= 99
                and (i + 3 == len(tokens) or not str(tokens[i + 3]).isdigit())):
            merged.append(str(int(tokens[i]) + int(tokens[i + 2])))
            i += 3
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

# American "-ize" <-> British "-ise" spelling family: sound-preserving, so
# canonicalizing one fixed direction (here, -ize forms -> -ise forms) makes
# "colonized"/"colonised" the same token without any fuzzy tolerance. This
# is the live-gate promotion of the same equivalence adjudicate.py's
# lenient_coverage already applies post-hoc. Anchored to known suffixes
# (word-final) rather than a blanket "z"->"s", to limit collateral hits on
# unrelated words that happen to contain "iz" (still not zero -- "size",
# "prize" end in one of these suffixes too -- but since the same function
# processes source and transcript alike, a coincidental hit is consistent
# on both sides and costs nothing at comparison time).
_IZE_SUFFIX_RE = re.compile(r"iz(e|es|ed|ing|ation|ations|er|ers|able|ability)\b")


def _canonicalize_spelling(word: str) -> str:
    return _IZE_SUFFIX_RE.sub(lambda m: "is" + m.group(1), word)


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


def _tokenize_pre_merge(text: str) -> list:
    """Shared preprocessing for tokenize()'s merge-chain orderings:
    lowercase, accent-fold, numeral/ordinal normalization, per-word number
    conversion, and digit-run merging. Returns the token list just before
    the final century/decade/thousand merge passes, which is where the two
    readings in `tokenize_variants` diverge."""
    t = text.lower().replace("\u2019", "'")
    # Fold accented Latin letters to their base ASCII form (e/e/i/i/a/a/..
    # for e,\u00e9,i,\u00ed,a,\u00e2,..) before the punctuation-stripping
    # regex below: an accent is not punctuation, but the regex's ASCII-only
    # allowlist would otherwise delete the letter entirely -- "Rio" (source
    # "R\u00edo") became "r"+"o", two garbage tokens, when Whisper (which
    # renders these names in plain ASCII) said "Rio". NFKD decomposition
    # separates a base letter from its combining accent mark (category Mn),
    # which is then dropped; a symmetric fold, so an accented ASR token
    # (Whisper occasionally emits one) agrees with a plain-ASCII source
    # token, and vice versa, regardless of which side has the accent.
    t = "".join(ch for ch in unicodedata.normalize("NFKD", t)
                if not unicodedata.combining(ch))
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
            out.append(_canonicalize_spelling(tok))
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
    return merged


def tokenize(text: str) -> list:
    """Normalize spoken and written text into comparable tokens.

    "N hundred and M thousand" reads as two separate numbers joined by
    "and" ("between five hundred and one thousand" = 500, 1000) --
    `tokenize_variants` also tries the reverse merge order for the
    alternate "one compound number" reading ("one hundred and fifty
    thousand" = 150000), since both are genuine, ambiguous-in-text-alone
    parses of the same words.
    """
    merged = _tokenize_pre_merge(text)
    return _canonicalize_bare_decades(
        _merge_century_remainder_tokens(
            _merge_thousand_tokens(
                _merge_hundreds_decade_tokens(_merge_century_tokens(_merge_decade_tokens(merged))))))


def _tokenize_compound_thousand(text: str) -> list:
    """Alternate reading for the "<hundred-multiple> and <remainder>
    thousand" shape: combine the hundred-and-remainder into one number
    BEFORE multiplying by thousand ("one hundred and fifty thousand" =
    150000) -- the reverse merge order from `tokenize`'s own "two numbers
    joined by 'and'" reading. Both are faithful parses of the same source
    words; see `tokenize_variants`, which callers should use instead of
    this directly."""
    merged = _tokenize_pre_merge(text)
    return _canonicalize_bare_decades(
        _merge_thousand_tokens(
            _merge_century_remainder_tokens(
                _merge_hundreds_decade_tokens(_merge_century_tokens(_merge_decade_tokens(merged))))))


def tokenize_variants(text: str) -> list:
    """All tokenizations of `text` under readings the source words support.

    Normally exactly one (`tokenize`'s own result). A "<hundred-multiple>
    and <remainder> thousand" shape is genuinely ambiguous in the text
    alone: "one hundred and fifty thousand" is one compound number
    (150,000), while "between five hundred and one thousand" is two
    numbers (500, 1000) joined by the same word "and" -- identical token
    shape, opposite grouping. It is NOT ambiguous in the audio: the TTS
    spoke one specific reading and Whisper transcribed what it heard, so
    whichever variant matches the transcript proves those words were
    spoken. Callers (the lexical gates) should accept a match under EITHER
    variant rather than guess the grouping at tokenize time."""
    default = tokenize(text)
    compound = _tokenize_compound_thousand(text)
    if compound != default:
        return [default, compound]
    return [default]


# --- speak-time text normalization (v31) -------------------------------------
# A bare capital roman numeral immediately after a name ("Abbas I,") is a
# real TTS mispronunciation hazard, not an ASR/gate equivalence problem --
# traced from ch02:p0055 ("Abbas I, the fifth Safavid shah"): the TTS
# rendered a short, unrelated syllable ("V") in both draws where "the
# First" needs roughly 6-7x longer, confirmed by word-timestamp duration
# (0.26s where "the First, the fifth" needs ~1.5-2s of speech) and by
# re-transcribing an isolated clip with no surrounding sentence context
# (still heard the same short "V" with high confidence, not the longer
# phrase Whisper's own language model would need to normalize away).
#
# normalize_for_tts rewrites this ONE hazard -- "Name/War/Part + roman
# numeral" -- into the word actually meant to be spoken, before the text
# reaches the TTS. It is not a general text normalizer: nothing else about
# the sentence changes.
_ROMAN_TO_ORDINAL = {
    "I": "First", "II": "Second", "III": "Third", "IV": "Fourth", "V": "Fifth",
    "VI": "Sixth", "VII": "Seventh", "VIII": "Eighth", "IX": "Ninth", "X": "Tenth",
}
_ROMAN_TO_CARDINAL = {
    "I": "One", "II": "Two", "III": "Three", "IV": "Four", "V": "Five",
    "VI": "Six", "VII": "Seven", "VIII": "Eight", "IX": "Nine", "X": "Ten",
}
# After one of these words, a roman numeral is a count, not a regnal
# ordinal: "World War II" = "World War Two", not "...the Second"; "Part
# III" = "Part Three".
_CARDINAL_CONTEXT_WORDS = frozenset({"War", "Part", "Chapter", "Volume", "Book", "Act"})

# Bare "I" pronoun-collision guard, resolved empirically rather than by
# guessing: grepping the WHOLE book (all 11 sections) for <CapitalizedWord>
# I <next-token> turns up exactly 3 hits total. Two are the regnal hazard
# this function exists to fix ("...the urgency with which Elizabeth I
# constructed the Anglican via media..." in ch02; "...the imperial legacy
# of Abbas I had been summarily dissolved" in ch03). One is a genuine
# first-person pronoun: "Thus I have used Constantinople and not
# Istanbul..." (the "names" front-matter section) -- "Thus" is a sentence
# adverb, not a name, immediately followed by its own subject "I". This is
# the ONLY connective word the grep actually found in this position, so
# it is the only one excluded; nothing else is guessed at.
_PRONOUN_CONTEXT_WORDS = frozenset({"Thus"})

_REGNAL_NUMERAL_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*)\s+(I|II|III|IV|V|VI|VII|VIII|IX|X)\b([.,;:]?)"
)


def normalize_for_tts(text: str) -> str:
    """Speak-time-only substitution for the "Name/War/Part + roman numeral"
    hazard (see module comment above). NEVER changes plan identity, chunk
    ids, or text_sha256 -- callers apply this only at the point text is
    handed to the TTS, and symmetrically as the ASR comparison's expected
    text (that is what was actually spoken); the chunk's own `text` field
    stays the untouched original everywhere else (plan hashing,
    `derive_mandatory`, state.json).

    Two readings of "Word + roman numeral":
    - Regnal ordinal (default): "Abbas I" -> "Abbas the First", "Suleiman
      II" -> "Suleiman the Second".
    - Cardinal count, when the preceding word is in the fixed exclusion
      list (_CARDINAL_CONTEXT_WORDS): "World War II" -> "World War Two",
      "Part III" -> "Part Three" -- English reads these as counts, not
      regnal ordinals.

    "I" specifically fires when followed by punctuation (comma/period/
    semicolon/colon) OR a lowercase word ("Elizabeth I constructed" ->
    "Elizabeth the First constructed"), UNLESS the preceding word is in
    _PRONOUN_CONTEXT_WORDS -- a bare "I" collides with the first-person
    pronoun, and a whole-book grep found exactly one word where that
    collision actually occurs ("Thus I have used..."), so that is the only
    exclusion; nothing else is guessed at. II-X have no such collision and
    fire on the pattern alone.
    """
    def repl(m):
        word, numeral, punct = m.group(1), m.group(2), m.group(3)
        if numeral == "I" and not punct and word in _PRONOUN_CONTEXT_WORDS:
            return m.group(0)
        cardinal = word in _CARDINAL_CONTEXT_WORDS
        table = _ROMAN_TO_CARDINAL if cardinal else _ROMAN_TO_ORDINAL
        prefix = "" if cardinal else "the "
        return f"{word} {prefix}{table[numeral]}{punct}"

    return _REGNAL_NUMERAL_RE.sub(repl, text)


# Roman-numeral <-> spoken-ordinal equivalence: normalize_for_tts rewrites
# "Abbas I," to "Abbas the First," before synthesis, but Whisper's own
# transcription may renormalize spoken "the First" straight back to the
# roman-numeral shorthand "I" (or the digit-ordinal "1st"), the same way it
# renormalizes any other spoken ordinal. Scoped to exactly the I-X set
# normalize_for_tts rewrites; matched and substituted as a whole phrase via
# the same _apply_phrase_pairs machinery as _TRANSLITERATION_PHRASE_PAIRS,
# so "the 1st" only ever equates to "i"/"1st" here, never a bare "first"
# floating anywhere else in the text.
_REGNAL_NUMERAL_PHRASE_PAIRS = tuple(
    (("the", digit_ordinal), (roman.lower(),), (digit_ordinal,))
    for roman, digit_ordinal in (
        ("I", "1st"), ("II", "2nd"), ("III", "3rd"), ("IV", "4th"), ("V", "5th"),
        ("VI", "6th"), ("VII", "7th"), ("VIII", "8th"), ("IX", "9th"), ("X", "10th"),
    )
)



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
_COMPLEMENTIZER_THAT_RE = re.compile(r"^\s*that\b")
_SERIAL_CONJUNCTION_RE = re.compile(r"^\s*(?:and|or)\b")


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
            if kind == "colon_semicolon" and mark[0] == ":":
                # A colon introducing a complement clause ("held good: that
                # European depictions...") reads straight through, unlike a
                # colon before a list or an independent clause -- the
                # 100ms rule was written for those, not this construction.
                # Lowercase "that" is the signal: a capitalized "That" would
                # start a new independent clause instead.
                after = source[chunk_match.start() + punctuation.end():]
                boundary["colon_complementizer_that"] = bool(_COMPLEMENTIZER_THAT_RE.match(after))
            if kind == "colon_semicolon" and mark[0] == ";":
                # A semicolon closing a serial list's final item ("a space;
                # a community; and a programme") reads straight through --
                # the conjunction itself marks the final item, so read-
                # through is the natural rendering. A non-final semicolon in
                # the same list, or one followed by anything but a lowercase
                # "and"/"or", stays gated at 100ms.
                after = source[chunk_match.start() + punctuation.end():]
                boundary["semicolon_serial_conjunction"] = bool(_SERIAL_CONJUNCTION_RE.match(after))
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
        exemption_reason = None
        if kind == "colon_semicolon" and boundary.get("colon_complementizer_that"):
            # "held good: that European depictions..." reads straight
            # through the colon -- advisory only, like a plain comma.
            threshold = None
            exemption_reason = "colon_complementizer_that"
        if kind == "colon_semicolon" and boundary.get("semicolon_serial_conjunction"):
            # "a space; a community; and a programme" reads straight through
            # the final semicolon -- the conjunction marks the last item.
            threshold = None
            exemption_reason = "semicolon_serial_conjunction"
        passed = None if not aligned or threshold is None else gap >= threshold
        measured = {
            **boundary,
            "asr_token_index": before_asr,
            "next_asr_token_index": after_asr,
            "aligned": aligned,
            "gap_s": gap,
            "threshold_s": threshold,
            "passed": passed,
            "exemption_reason": exemption_reason,
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


def _apply_phonetic_matches(tokens: list, phonetic_matches: dict) -> list:
    """Substitute a source token the mandatory gate already demoted via
    v12's phonetic-match machinery (`_find_phonetic_match`) with its
    recorded transcript counterpart. This is not a new fuzzy comparison --
    the match was already made and recorded in `phonetic_matches`
    (mandatory.phonetic_matches); this just stops coverage/terminal from
    contradicting that conclusion with a second, stricter verdict on the
    same evidence. A token with no recorded match is left untouched, so
    this can only turn a would-be miss into a match, never the reverse."""
    if not phonetic_matches:
        return tokens
    return [phonetic_matches.get(tok, tok) for tok in tokens]


# --- alignment metrics -------------------------------------------------------
def ordered_coverage(expected: list, asr: list, phonetic_matches: dict = None) -> dict:
    """Maximum ordered token coverage of expected in ASR tokens (LCS).

    A token of `expected` is matched if the expected tokens form a
    subsequence of the ASR tokens; the best alignment is found with a
    classic LCS DP, so an expected token absent from the audio only costs
    itself, never the tokens after it. O(len(expected) * len(asr)) — trivial
    at sentence scale.

    `expected` first has any recorded phonetic matches substituted in
    (`_apply_phonetic_matches`), then any curated phrase-pair variant
    actually present in `asr` (`_apply_phrase_pairs`), then both lists are
    reconciled across hyphen/compound-word boundaries
    (`_reconcile_concat_boundaries`): a token-boundary difference like
    "waste lands" / "wastelands" is orthography-only, so it must not cost
    a match here any more than it does in the mandatory-word gate.
    """
    expected = _apply_phonetic_matches(expected, phonetic_matches)
    expected, phrase_pair_matches = _apply_phrase_pairs(expected, asr)
    expected, asr = _reconcile_concat_boundaries(expected, asr)
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
        "phrase_pair_matches": phrase_pair_matches,
    }


_VOWELS = set("aeiouy")


def _is_vowel_swap(a: str, b: str) -> bool:
    """True when `a` and `b` are the same length, both >= 4 characters, and
    differ in exactly one position by a vowel-for-vowel substitution
    (a/e/i/o/u/y) -- a loanword transliteration variant Whisper prefers
    ("amir" -> "emir"), the same vocabulary-snapping phenomenon as v12's
    phonetic matching but one letter gentler. A consonant difference, or
    any insertion/deletion (a length difference), does not count -- only
    an exact single vowel-for-vowel substitution."""
    if len(a) != len(b) or len(a) < 4:
        return False
    diff = None
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        if diff is not None:
            return False
        diff = (ca, cb)
    return diff is not None and diff[0] in _VOWELS and diff[1] in _VOWELS


# Curated transliteration-equivalent pairs: names and loanwords with a
# well-established alternate English spelling that is NOT a single
# vowel-for-vowel substitution (_is_vowel_swap already covers those, e.g.
# amir/emir) -- so a general rule can't reach them, and a fuzzy ratio would
# either miss them or blur genuinely different words. Each pair must be
# justified as two spellings of the SAME referent (verified against the
# source text and, for names, an outside reference), never a mangled form
# (that is v12's phonetic-match job) and never two ordinary English words
# that happen to look alike (no form/farm-type pairs).
#
# Extend this set following the standing self-extension lane: add a pair
# without stopping the line only when (1) the only miss in an otherwise
# content-correct transcript is the candidate token, (2) both forms are
# verifiably established spellings of the same loanword/proper noun (a
# recognized transliteration alternation, e.g. k/q, ee/i, oo/u, dj/j,
# gh/g, single/double consonant), and (3) neither form is an ordinary
# English content word with its own distinct meaning. Anything short of
# all three still stops the line for a decision, the same as before this
# lane existed.
#
# Seeded from paragraphs that actually hit this in ch01: Whisper renders
# "Koranic" (source spelling, used throughout the book) as "Quranic" --
# same holy book, k/q is a standard Arabic-transliteration alternation.
# "Genghis"/"Chinggis" (source: "the Mongol empire of Genghis (Chinggis
# Khan)") names the same 13th-century Mongol ruler; the book itself
# glosses one against the other. Checked and deliberately NOT added:
# Muhammad/Mohammed (this book uses "Mohammed" for a distinct 18th-century
# Qajar shah, "Agha Mohammed" -- not the Prophet -- so the two spellings
# are not interchangeable here); khalifa/caliph(ate) (person vs.
# institution, not a spelling variant); Tamerlane/Timur (an epithet vs. a
# given name, not a transliteration pair).
#
# ch02: "Cracow" (source spelling, the traditional English exonym) /
# "Krakow" (transcript, the modern standard English spelling) name the
# same Polish city -- consonant substitution (c/k twice), below the
# general fuzzy-ratio floor (0.667 < 0.8) and not a vowel swap, so neither
# existing mechanism reaches it; both draws rendered it identically, the
# only miss in an otherwise content-correct transcript.
_TRANSLITERATION_PAIRS = frozenset({
    frozenset({"koran", "quran"}),
    frozenset({"koranic", "quranic"}),
    frozenset({"genghis", "chinggis"}),
    frozenset({"cracow", "krakow"}),
})


def _is_transliteration_pair(a: str, b: str) -> bool:
    return frozenset({a, b}) in _TRANSLITERATION_PAIRS


def _tok_eq(a: str, b: str, ratio: float) -> bool:
    if a == b:
        return True
    if _is_transliteration_pair(a, b):
        return True
    # phonetic tolerance for proper nouns ("Tamerlane" ~ "Tamalane") without
    # letting short function words match anything
    if len(a) >= 4 and len(b) >= 4:
        if _is_vowel_swap(a, b):
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= ratio
    return False


def _find_vowel_match(word: str, asr: list) -> str:
    """First ASR token that is a vowel-swap transliteration variant of
    `word` (see `_is_vowel_swap`), or None."""
    for candidate in asr:
        if _is_vowel_swap(word, candidate):
            return candidate
    return None


def _find_transliteration_match(word: str, asr: list) -> str:
    """First ASR token that is a curated transliteration variant of `word`
    (see `_TRANSLITERATION_PAIRS`), or None."""
    for candidate in asr:
        if _is_transliteration_pair(word, candidate):
            return candidate
    return None


# Curated PHRASE-level transliteration pairs: a fixed multi-word sequence
# naming one entity/title, where one short function word inside it (an
# article/preposition, <= 3 chars) has an alternate rendering. Matched and
# substituted as a whole, positionally-aligned sequence -- never a bare
# word pair -- so the equivalence exists only inside this exact phrase; a
# "da"/"de" swap anywhere else in the text is still a real mismatch (the
# v23 guard this structure exists specifically to respect).
#
# Self-extension lane (phrase form): add a phrase pair without stopping
# only when ALL hold: (1) both attempts are otherwise content-correct,
# the only miss is one short (<= 3 char) function word inside the phrase;
# (2) the variance is a single vowel or single-character difference in
# that function word; (3) the phrase names one fixed entity or title, not
# free text; (4) commit the addition on its own with the evidence.
#
# Seeded from ch02: "Estado da India" (source spelling, the Portuguese
# crown's colonial administration in Asia -- one fixed historical entity,
# recurring across the colonial chapters) transcribed as "Estado de
# India" -- Whisper reducing the Portuguese /dɐ/ in "da" to "de", a vowel
# distinction in a foreign monosyllable embedded in an English sentence.
_TRANSLITERATION_PHRASE_PAIRS = (
    (("estado", "da", "india"), ("estado", "de", "india")),
)


def _apply_phrase_pairs(expected: list, asr: list) -> tuple:
    """Rewrite `expected` so a curated phrase-pair variant (see
    `_TRANSLITERATION_PHRASE_PAIRS` and `_REGNAL_NUMERAL_PHRASE_PAIRS`)
    actually present in `asr` replaces the source's own spelling of that
    phrase. Returns (rewritten_expected, fired) where `fired` maps the
    source phrase to the transcript phrase it was aligned to, for the
    morning audit.

    A single forward pass over the ORIGINAL `expected` list, never
    re-reading its own output: some groups pair variants of different
    lengths where one is a token-subset of another (the regnal-numeral
    group has both "the 1st" and bare "1st"), so a naive repeated
    whole-list rewrite can match a span it just wrote and re-substitute it
    forever. Reading only from the untouched original and advancing past
    whatever was matched makes that impossible -- the scan position is
    monotonic and bounded by len(expected).
    """
    all_pairs = _TRANSLITERATION_PHRASE_PAIRS + _REGNAL_NUMERAL_PHRASE_PAIRS
    out = []
    fired = {}
    i = 0
    n_exp = len(expected)
    while i < n_exp:
        matched = False
        for variants in all_pairs:
            for variant in variants:
                n = len(variant)
                if tuple(expected[i:i + n]) != variant:
                    continue
                for alt in variants:
                    if alt == variant:
                        continue
                    m = len(alt)
                    if any(tuple(asr[j:j + m]) == alt for j in range(len(asr) - m + 1)):
                        out.extend(alt)
                        fired[" ".join(variant)] = " ".join(alt)
                        i += n
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if not matched:
            out.append(expected[i])
            i += 1
    return out, fired


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


# --- phonetic-match demotion --------------------------------------------------
# Whisper strongly prefers real words in its training vocabulary: a correctly
# spoken RARE word ("decentre") predictably transcribes as its nearest common
# neighbor ("dissenter", "desanter", "disantre" -- all independent draws of
# the same word landed within one vowel of it). That is an ASR vocabulary
# limitation, not evidence the word was missing from the audio, so a mandatory
# word with a phonetically-adjacent transcript candidate demotes the same way
# a bare number word does: reported, not hard-failed.
PHONETIC_LENGTH_TOLERANCE = 0.4  # raw word lengths must be within this fraction
_PHONETIC_C_BEFORE_EI_RE = re.compile(r"c(?=[ei])")
_PHONETIC_DOUBLE_RE = re.compile(r"(.)\1+")
_PHONETIC_VOWEL_RE = re.compile(r"[aeiouy]")


def _phonetic_skeleton(word: str) -> str:
    """Collapse a word to a coarse consonant skeleton so common digraph and
    vowel-spelling differences ("decentre" / "dissenter") fall out: lowercase,
    map digraphs towards their sound (ph->f, gh (silent) -> dropped, ck->k,
    q->k, c->s before e/i else k, x->ks), collapse doubled letters, then
    drop vowels except a leading one (which carries the word's opening
    sound).

    q->k (v32): "qullars" (a Persian/Ottoman administrative loanword,
    transliterated with a q) is /k/, the same phoneme as "colors"'s c --
    q was left as a distinct letter, so "qullars" (klrs after this fix)
    and Whisper's "colors" (klrs already, since c->k) never matched before
    this: q and k denote the same sound in English, so this is filling a
    gap in an existing rule, not a new tolerance."""
    w = word.lower()
    w = w.replace("ph", "f")
    w = w.replace("gh", "")
    w = w.replace("ck", "k")
    w = w.replace("q", "k")
    w = _PHONETIC_C_BEFORE_EI_RE.sub("s", w)
    w = w.replace("c", "k")
    w = w.replace("x", "ks")
    w = _PHONETIC_DOUBLE_RE.sub(r"\1", w)
    if w and w[0] in "aeiouy":
        return w[0] + _PHONETIC_VOWEL_RE.sub("", w[1:])
    return _PHONETIC_VOWEL_RE.sub("", w)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _phonetic_match(word: str, candidate: str) -> bool:
    """True when `candidate` is a plausible ASR mishearing of `word`: same
    (or off-by-one, for long-enough skeletons) phonetic skeleton, the same
    onset sound, and comparable raw length. All three must hold -- this is
    the precision guard against demoting a genuine content miss.

    The onset check compares the SKELETON's first letter, not the raw
    word's: a digraph fold (ph->f, q->k, ...) can change a word's onset
    letter without changing its onset sound ("qullars" raw-starts with q,
    sounds like k, same as "colors") -- comparing raw first letters would
    reject that pair even though _phonetic_skeleton already normalizes
    both to "klrs"."""
    if not word or not candidate or word == candidate:
        return False
    w_skel, c_skel = _phonetic_skeleton(word), _phonetic_skeleton(candidate)
    if not w_skel or not c_skel or w_skel[0] != c_skel[0]:
        return False
    if w_skel == c_skel:
        pass
    elif len(w_skel) >= 4 and len(c_skel) >= 4 and _levenshtein(w_skel, c_skel) <= 1:
        pass
    else:
        return False
    longer, shorter = max(len(word), len(candidate)), min(len(word), len(candidate))
    return shorter / longer >= (1 - PHONETIC_LENGTH_TOLERANCE)


def _find_phonetic_match(word: str, asr: list) -> str:
    """First ASR token that is a plausible mishearing of `word`, or None."""
    for candidate in asr:
        if _phonetic_match(word, candidate):
            return candidate
    return None


# Curated ASR-fragile lexicon (v33): words the book has empirically proven
# Whisper cannot spell, same family as v8's bare-digit demotion but for
# specific rare vocabulary rather than a structural token class. Matching
# (_find_lexicon_match) is more permissive than the general phonetic-match
# tolerance (_find_phonetic_match): same skeleton onset, skeleton edit
# distance <= 2, not exact-or-<=1-with-a-length-floor -- safe only because
# the lexicon is small, curated, and evidence-gated per word, never a
# blanket loosening of the general mechanism.
#
# Self-extension lane, ALL conditions required: (1) the word is a rare
# loanword/proper term >= 6 chars; (2) at least 2 DISTINCT garbles of it
# have been observed across independent draws with the surrounding
# context otherwise exact; (3) each addition is its own tiny commit
# citing the garbles; (4) every fire lands in the validation record
# (`lexicon_matches`) for morning audit. Anything short of all four stops
# the line for a decision, same as before this lane existed.
#
# Seeded from ch02:p0055 -- "qullars" (a Persian/Ottoman administrative
# loanword) garbled three distinct ways across independent draws, context
# exact every time: "colors" (v32's q->k skeleton fold already catches
# this one exactly, skeleton "klrs" both), "Kulas" (missing the r sound:
# skeleton "kls", only 3 characters, below _find_phonetic_match's >= 4-char
# tolerance floor), "Gula Mani" (a different occurrence's garble, split
# into two words).
_ASR_FRAGILE_LEXICON = frozenset({"qullars"})


def _find_lexicon_match(word: str, asr: list) -> str:
    """First ASR token sharing `word`'s skeleton onset with skeleton edit
    distance <= 2, or None. Only ever called for a word in
    `_ASR_FRAGILE_LEXICON` -- see that set's comment for why this wider
    tolerance is safe there but not as a general rule."""
    w_skel = _phonetic_skeleton(word)
    if not w_skel:
        return None
    for candidate in asr:
        c_skel = _phonetic_skeleton(candidate)
        if c_skel and c_skel[0] == w_skel[0] and _levenshtein(w_skel, c_skel) <= 2:
            return candidate
    return None


def _concat_match(phrase: list, asr: list) -> str:
    """Exact hyphen-boundary match for a mandatory item, in either
    direction: adjacent ASR tokens concatenating to equal a single-token
    mandatory word ("southeastward" <- "south" + "eastward"), or a
    multi-token mandatory phrase (from a hyphenated source word tokenize()
    split) concatenating to equal a single ASR token (the reverse). A
    hyphen changes spelling, never sound, so this is exact string equality
    on adjacent tokens only -- a real content match, not a fuzzy or fragile
    one. Returns the matched ASR span (space-joined) or None."""
    if len(phrase) == 1:
        word = phrase[0]
        for i in range(len(asr)):
            acc = ""
            for j in range(i, len(asr)):
                acc += asr[j]
                if acc == word:
                    return " ".join(asr[i:j + 1])
                if len(acc) >= len(word):
                    break
        return None
    target = "".join(phrase)
    for tok in asr:
        if tok == target:
            return tok
    return None


_CONCAT_RECONCILE_MAX_SPAN = 3


def _concat_reconcile(a: list, b: list) -> list:
    """Rewrite `a` so a token-boundary difference from a hyphen or compound-
    word split against `b` doesn't cost a coverage/terminal match: wherever
    a run of 2-3 adjacent tokens in `a` concatenates to exactly equal some
    token in `b`, collapse the run into that single token. Exact string
    equality only -- orthography-only and sound-preserving, the same
    equivalence `_concat_match` already applies to the mandatory-word gate
    (v17/v20), extended here to coverage and terminal matching so all three
    lexical gates agree on what counts as the same word."""
    b_set = set(b)
    out = []
    i, n = 0, len(a)
    while i < n:
        merged = None
        for span in range(min(_CONCAT_RECONCILE_MAX_SPAN, n - i), 1, -1):
            candidate = "".join(a[i:i + span])
            if candidate in b_set:
                merged = candidate
                break
        if merged is not None:
            out.append(merged)
            i += span
        else:
            out.append(a[i])
            i += 1
    return out


def _reconcile_concat_boundaries(expected: list, asr: list) -> tuple:
    """Reconcile both directions: merge `expected` runs that concatenate to
    an `asr` token, and merge `asr` runs that concatenate to an `expected`
    token, each against the other's original (pre-reconciliation) list."""
    return _concat_reconcile(expected, asr), _concat_reconcile(asr, expected)


def check_mandatory(asr: list, mandatory: list, ratio: float = MANDATORY_FUZZY_RATIO) -> dict:
    """Each mandatory item (string or token list) must appear as a contiguous,
    fuzzy-tolerant run in the ASR tokens, or match across a hyphen boundary
    (`_concat_match`) -- the latter counts as a full, non-fragile match since
    a hyphen never changes pronunciation. An item flagged `is_fragile_mandatory`
    (a bare number-word marker) or matched by `_find_phonetic_match` (a rare
    word Whisper snapped to its nearest common neighbor, e.g. "decentre" ->
    "dissenter") is reported in `missing_fragile` instead of `missing`: it
    still counts toward `coverage` (computed separately over every expected
    token), but its absence alone never fails the verdict. Phonetic matches
    and hyphen matches are additionally recorded (`phonetic_matches`,
    `hyphen_matches`: item -> matched ASR span) so a later review can see
    every such acceptance. A single-token item that already passed via the
    shared `_tok_eq` comparator's vowel-swap tolerance (a loanword
    transliteration variant, e.g. "amir" -> "emir") is also noted in
    `vowel_matches`, and one that passed via a curated transliteration pair
    (`_TRANSLITERATION_PAIRS`, e.g. "koranic" -> "quranic") in
    `transliteration_matches`, for the same audit, even though neither was
    ever at risk of hard-failing here."""
    items = []
    missing = []
    missing_fragile = []
    phonetic_matches = {}
    hyphen_matches = {}
    vowel_matches = {}
    transliteration_matches = {}
    lexicon_matches = {}
    for item in mandatory:
        raw_phrase = item if isinstance(item, str) else " ".join(map(str, item))
        phrase = tokenize(raw_phrase)
        if not phrase:
            continue
        label = item if isinstance(item, str) else " ".join(map(str, item))
        items.append(label)
        if _phrase_in(asr, phrase, ratio):
            if len(phrase) == 1 and phrase[0].isalpha():
                vmatch = _find_vowel_match(phrase[0], asr)
                if vmatch:
                    vowel_matches[label] = vmatch
                tmatch = _find_transliteration_match(phrase[0], asr)
                if tmatch:
                    transliteration_matches[label] = tmatch
            continue
        concat = _concat_match(phrase, asr)
        if concat:
            hyphen_matches[label] = concat
            continue
        if is_fragile_mandatory(phrase):
            missing_fragile.append(label)
            continue
        if len(phrase) == 1 and phrase[0].isalpha():
            match = _find_phonetic_match(phrase[0], asr)
            if match:
                missing_fragile.append(label)
                phonetic_matches[label] = match
                continue
            if phrase[0] in _ASR_FRAGILE_LEXICON:
                lmatch = _find_lexicon_match(phrase[0], asr)
                if lmatch:
                    missing_fragile.append(label)
                    # Also recorded into phonetic_matches: v25's coverage/
                    # terminal propagation reads that field, and a curated
                    # lexicon hit deserves the same positional match there
                    # as any other phonetic acceptance -- lexicon_matches
                    # is the separate, audit-clear record of which specific
                    # fires came from the curated list.
                    phonetic_matches[label] = lmatch
                    lexicon_matches[label] = lmatch
                    continue
        missing.append(label)
    return {"items": items, "missing": missing, "missing_fragile": missing_fragile,
            "phonetic_matches": phonetic_matches, "hyphen_matches": hyphen_matches,
            "vowel_matches": vowel_matches, "transliteration_matches": transliteration_matches,
            "lexicon_matches": lexicon_matches}


def terminal_suffix(expected: list, asr: list, size: int = 5, phonetic_matches: dict = None) -> dict:
    """Require the final expected phrase near the end of the transcript.

    Phonetic matches already recorded by the mandatory gate are substituted
    in first (`_apply_phonetic_matches`), then any curated phrase-pair
    variant (`_apply_phrase_pairs`), then both are reconciled across
    hyphen/compound-word boundaries, same as `ordered_coverage`, so a tail
    like "waste lands were colonized" matches a transcript ending
    "wastelands were colonised" -- the reported `expected` phrase reflects
    the reconciled tail actually checked.
    """
    expected = _apply_phonetic_matches(expected, phonetic_matches)
    expected, phrase_pair_matches = _apply_phrase_pairs(expected, asr)
    r_expected, r_asr = _reconcile_concat_boundaries(expected, asr)
    tail = r_expected[-size:]
    window = r_asr[-(len(tail) + 2):]
    return {
        "expected": " ".join(tail),
        "matched": not tail or _phrase_in(window, tail, MANDATORY_FUZZY_RATIO),
        "phrase_pair_matches": phrase_pair_matches,
    }


def derive_mandatory(expected_tokens: list, expected_text: str = None) -> list:
    """Default mandatory content: digits plus content words (len >= 6, not
    function words) from the expected text, plus canonical decade tokens
    ("40s", "1830s", ...) -- alnum, so neither the digit nor the alpha branch
    would otherwise catch them, but a decade is real content just like a
    year. Callers may pass an explicit `mandatory` list to override.
    `check_mandatory` (not this function) decides which of these items are
    hard-mandatory vs. ASR-fragile bare number words -- both are returned
    here so fragile items still count toward the returned candidate set.

    When `expected_text` is given, a two-part hyphenated compound word
    ("strong-points") whose combined letters qualify as content (>= 6) is
    added as ONE phrase item (its tokenize()'d halves), replacing its
    individual halves in the per-token derivation below. tokenize() splits
    a hyphen into separate tokens, so without this, "strong" and "points"
    would each be checked alone and neither matches Whisper's solid
    rendering ("strongpoints") -- only a phrase-level mandatory item lets
    check_mandatory's hyphen-boundary equivalence (_concat_match) see them
    as one unit and match the concatenation."""
    hyphen_phrases = []
    excluded = set()
    if expected_text:
        for m in _HYPHENATED_WORD_RE.finditer(expected_text):
            parts = tokenize(m.group(0))
            if len(parts) >= 2 and sum(len(p) for p in parts) >= 6:
                hyphen_phrases.append(" ".join(parts))
                excluded.update(parts)
    out = []
    seen = set()
    for tok in expected_tokens:
        if tok in seen or tok in excluded:
            continue
        keep = (tok.isdigit() or _DECADE_TOKEN_RE.fullmatch(tok)
                or (len(tok) >= 6 and tok.isalpha() and tok not in _FUNCTION_WORDS))
        if keep:
            seen.add(tok)
            out.append(tok)
    for phrase in hyphen_phrases:
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
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
        word_timings = normalized_word_timings(segments)
        # tokenize_variants: normally one reading, but a "<hundred-multiple>
        # and <remainder> thousand" shape is genuinely ambiguous in the text
        # alone (one compound number vs. two numbers joined by "and" -- see
        # tokenize_variants's docstring). Try the default ("separate")
        # reading first, then the alternate ("compound") one only if it
        # exists; keep whichever passes, or the default's result if neither
        # does, so a real failure is still reported against the primary
        # reading.
        expected_variants = tokenize_variants(expected_text)
        variant_labels = ("separate", "compound")
        metrics = v = reasons = number_parse = None
        for i, expected_tokens in enumerate(expected_variants):
            mand_items = derive_mandatory(expected_tokens, expected_text) if mandatory is None else mandatory
            mandatory_result = check_mandatory(asr_tokens, mand_items)
            phonetic_matches = mandatory_result.get("phonetic_matches")
            candidate_metrics = {
                "coverage": ordered_coverage(expected_tokens, asr_tokens, phonetic_matches),
                "mandatory": mandatory_result,
                "terminal": terminal_suffix(expected_tokens, asr_tokens, phonetic_matches=phonetic_matches),
                "confidence": _confidence(segments),
                "repetition": repetition_stats(asr_tokens, expected_tokens),
                "leakage": leakage_check(asr_tokens, leakage_texts or []),
                "words": _word_stats(segments),
                "punctuation": punctuation_metrics(expected_text, word_timings),
            }
            candidate_v, candidate_reasons = verdict(candidate_metrics)
            if metrics is None:
                metrics, v, reasons, number_parse = (
                    candidate_metrics, candidate_v, candidate_reasons, variant_labels[i])
            if candidate_v == "PASS":
                metrics, v, reasons, number_parse = (
                    candidate_metrics, candidate_v, candidate_reasons, variant_labels[i])
                break
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
            "mandatory": metrics["mandatory"],
            "terminal": metrics["terminal"],
            "words": metrics["words"],
            "word_timings": word_timings,
            "punctuation": metrics["punctuation"],
            "number_parse": number_parse,
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

    # The classic English year idiom "N hundred and M" (843, 1984, ...): the
    # explicit "and" is the disambiguating signal that separates this from
    # the "fourteen hundred one" case above (no "and" -- not idiomatic for
    # a single number, so it stays unmerged).
    check("hundred-and-remainder equals the written year",
          tokenize("eight hundred and forty-three") == tokenize("843") == ["843"])
    check("century-and-remainder equals the written year",
          tokenize("nineteen hundred and eighty-four") == tokenize("1984") == ["1984"])
    check("hundred-and-remainder reaches coverage",
          ordered_coverage(tokenize("fallen apart by eight hundred and forty-three"),
                           tokenize("fallen apart by 843"))["missing"] == [])
    check("hundred-and-remainder reaches mandatory",
          check_mandatory(tokenize("fallen apart by 843"),
                          ["eight hundred and forty-three"])["missing"] == [])
    check("without 'and' the remainder stays unmerged (not idiomatic)",
          tokenize("eight hundred forty-three") != tokenize("843"))
    check("non-century prefix remains distinct",
          tokenize("twenty hundred") != tokenize("2000"))
    check("plural hundred remains distinct",
          tokenize("fourteen hundreds") != tokenize("1400"))

    # "N hundred" as a plain quantity ("three hundred years" = 300 years),
    # not a century reference, for single-digit prefixes one-nine.
    check("spoken hundred-quantity equals written digits",
          tokenize("three hundred years") == tokenize("300 years"))
    check("hundred-quantity equivalence reaches coverage",
          ordered_coverage(tokenize("nearly three hundred years"),
                           tokenize("nearly 300 years"))["missing"] == [])
    check("hundred-quantity equivalence reaches mandatory",
          check_mandatory(tokenize("nearly 300 years"), ["three hundred"])["missing"] == [])
    check("hundred-quantity count continuation remains distinct",
          tokenize("three hundred one") != tokenize("300"))
    check("different hundred-quantity remains distinct",
          tokenize("three hundred") != tokenize("four hundred"))

    # "N hundreds" (plural) is a century-block decade reference ("the four
    # hundreds" = the 400s era), the plural sibling of the plain N-hundred
    # quantity above -- epub.expand_numbers renders written "400s" as
    # spoken "four hundreds", so Whisper's digit rendering needs to match.
    check("spoken century-block equals written digits",
          tokenize("four hundreds") == tokenize("400s"))
    check("century-block equivalence reaches coverage",
          ordered_coverage(tokenize("by the four hundreds"),
                           tokenize("by the 400s"))["missing"] == [])
    check("century-block equivalence reaches mandatory",
          check_mandatory(tokenize("by the 400s"), ["four hundreds"])["missing"] == [])
    check("plural hundred still distinct from the bare year",
          tokenize("fourteen hundreds") != tokenize("1400") and
          tokenize("fourteen hundreds") == tokenize("1400s"))
    check("different century-block remains distinct",
          tokenize("four hundreds") != tokenize("nine hundreds"))

    # "N thousand" multiplies by 1000 for any digit prefix -- unlike "N
    # hundred", there is no non-idiomatic two-digit case to exclude ("twenty
    # thousand" is ordinary English), and a merged hundred value chains in
    # ("six hundred thousand" -> "600" -> "600000").
    check("spoken thousand equals written digits",
          tokenize("one thousand") == tokenize("1000"))
    check("thousand equivalence reaches coverage",
          ordered_coverage(tokenize("between five hundred and one thousand"),
                           tokenize("between 500 and 1000"))["missing"] == [])
    check("thousand equivalence reaches mandatory",
          check_mandatory(tokenize("between 500 and 1000"), ["one thousand"])["missing"] == [])
    check("thousand count continuation remains distinct",
          tokenize("one thousand one") != tokenize("1000"))
    check("hundred-thousand compound chains through both merges",
          tokenize("six hundred thousand") == tokenize("600000") == ["600000"])
    check("different thousand remains distinct",
          tokenize("one thousand") != tokenize("two thousand"))

    # "N thousand M hundred" (2500, 12300, ...): both parts are round scale
    # units, so unlike a bare hundred-remainder ("eight hundred forty-
    # three"), no "and" is needed to disambiguate -- this is the standard
    # unmarked way to speak such a number.
    check("thousand-hundred compound equals written digits",
          tokenize("two thousand five hundred miles") == tokenize("2500 miles"))
    check("larger thousand-hundred compound equals written digits",
          tokenize("twelve thousand three hundred delegates") ==
          tokenize("12300 delegates"))
    check("thousand-hundred equivalence reaches mandatory",
          check_mandatory(tokenize("some 2500 miles west"),
                          ["two thousand five hundred"])["missing"] == [])
    check("thousand plus a non-hundred remainder stays unmerged",
          tokenize("two thousand fifty") != tokenize("2050"))

    # v29: "<hundred-multiple> and <remainder> thousand" is genuinely
    # ambiguous in the text alone -- "one hundred and fifty thousand" is
    # one compound number (150,000), "between five hundred and one
    # thousand" is two numbers (500, 1000) joined by the same word "and".
    # Not ambiguous in the audio, though: whichever variant matches the
    # transcript proves those exact words were spoken, so the lexical
    # gates try both and accept either.
    check("unambiguous number produces exactly one variant",
          len(tokenize_variants("two thousand miles")) == 1)
    check("ordinary sentence produces exactly one variant",
          len(tokenize_variants("The death of Tamerlane in 1405.")) == 1)
    compound_variants = tokenize_variants("some one hundred and fifty thousand white Spanish.")
    check("ambiguous shape produces exactly two variants",
          len(compound_variants) == 2, repr(compound_variants))
    compound_asr = tokenize("some 150,000 white Spanish.")
    compound_passed = False
    for exp in compound_variants:
        mand = derive_mandatory(exp, "some one hundred and fifty thousand white Spanish.")
        mres = check_mandatory(compound_asr, mand)
        if (mres["missing"] == [] and
                ordered_coverage(exp, compound_asr, mres["phonetic_matches"])["missing"] == [] and
                terminal_suffix(exp, compound_asr, phonetic_matches=mres["phonetic_matches"])["matched"]):
            compound_passed = True
    check("compound reading (150,000) passes under some variant", compound_passed)
    range_variants = tokenize_variants("between five hundred and one thousand.")
    range_asr = tokenize("between 500 and 1000.")
    range_passed = False
    for exp in range_variants:
        mand = derive_mandatory(exp, "between five hundred and one thousand.")
        mres = check_mandatory(range_asr, mand)
        if (mres["missing"] == [] and
                ordered_coverage(exp, range_asr, mres["phonetic_matches"])["missing"] == [] and
                terminal_suffix(exp, range_asr, phonetic_matches=mres["phonetic_matches"])["matched"]):
            range_passed = True
    check("separate reading (500 and 1000) passes under some variant", range_passed)
    wrong_asr = tokenize("some 80,000 white Spanish.")
    wrong_passed = False
    for exp in compound_variants:
        mand = derive_mandatory(exp, "some one hundred and fifty thousand white Spanish.")
        mres = check_mandatory(wrong_asr, mand)
        if (mres["missing"] == [] and
                ordered_coverage(exp, wrong_asr, mres["phonetic_matches"])["missing"] == [] and
                terminal_suffix(exp, wrong_asr, phonetic_matches=mres["phonetic_matches"])["matched"]):
            wrong_passed = True
    check("a genuinely different number fails under every variant", not wrong_passed)

    # v28: an accented Latin letter is a letter, not punctuation -- fold it
    # to its base ASCII form (NFKD decompose, drop combining marks) instead
    # of deleting it. "Río" was becoming "r"+"o" (two garbage tokens)
    # against Whisper's plain-ASCII "Rio", failing terminal by construction.
    check("accented proper noun folds to plain ASCII",
          tokenize("Río de la Plata") == ["rio", "de", "la", "plata"])
    check("circumflex folds to plain ASCII",
          tokenize("the coup de grâce") == ["the", "coup", "de", "grace"])
    check("tilde folds to plain ASCII", tokenize("São Paulo") == ["sao", "paulo"])
    check("cedilla folds to plain ASCII", tokenize("François") == ["francois"])
    check("acute accent folds to plain ASCII", tokenize("café") == ["cafe"])
    check("umlaut folds to plain ASCII", tokenize("Müller") == ["muller"])
    check("accent folding is symmetric (accented transcript, plain source)",
          tokenize("Rio de la Plata") == tokenize("Río de la Plata"))
    check("punctuation is still stripped after accent folding",
          tokenize("Café, naïve—résumé!") == ["cafe", "naive", "resume"])
    check("digits and apostrophes still handled as before",
          tokenize("world's fourteen oh five") == ["world's", "1405"])
    check("plain ASCII text is byte-identical to the pre-fix output",
          tokenize("The DEATH, of Tamerlane in 1405!") ==
          ["the", "death", "of", "tamerlane", "in", "1405"])
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

    # Phonetic-match demotion: Whisper snaps a rare word to its nearest
    # common neighbor ("decentre" -> "dissenter"/"desanter"/"disantre", all
    # within one vowel of the source). A mandatory word with a
    # phonetically-adjacent ASR candidate demotes to missing_fragile, same
    # family as the bare-number-word demotion above -- reported, not
    # hard-failed. A genuinely absent or phonetically unrelated word stays
    # a hard miss (the precision guard).
    check("decentre matches its real mishearing dissenter",
          _phonetic_match("decentre", "dissenter"))
    check("decentre matches its real mishearing desanter",
          _phonetic_match("decentre", "desanter"))
    check("decentre matches its real mishearing disantre",
          _phonetic_match("decentre", "disantre"))
    check("decentre does not match an unrelated word (demolish)",
          not _phonetic_match("decentre", "demolish"))
    check("europe does not match an unrelated word (erode)",
          not _phonetic_match("europe", "erode"))
    decentre_asr = tokenize(
        "the saidian critique was part of a great sea change a conscious "
        "attempt to dissenter europe or even to provincialize it")
    decentre_mand = ["saidian", "critique", "change", "conscious", "attempt",
                     "decentre", "europe", "provincialize"]
    decentre_check = check_mandatory(decentre_asr, decentre_mand)
    check("decentre/dissenter is not hard-mandatory",
          decentre_check["missing"] == [], repr(decentre_check))
    check("decentre/dissenter is recorded as fragile with its phonetic match",
          decentre_check["missing_fragile"] == ["decentre"] and
          decentre_check["phonetic_matches"] == {"decentre": "dissenter"},
          repr(decentre_check))
    check("a mandatory word entirely absent (no candidate) stays hard-mandatory",
          check_mandatory(tokenize("nothing at all related here"),
                          ["decentre"])["missing"] == ["decentre"])
    check("a phonetically unrelated substitution stays hard-mandatory",
          check_mandatory(tokenize("a conscious attempt to demolish europe"),
                          ["decentre"])["missing"] == ["decentre"])
    check("europe swapped for the unrelated erode stays hard-mandatory",
          check_mandatory(tokenize("the attempt to decentre erode entirely"),
                          ["europe"])["missing"] == ["europe"])

    # v32: q and k are the same phoneme ("qullars" is /'kulɑrz/), but the
    # skeleton normalizer left q as a distinct letter -- "qullars" (klrs
    # after this fix) and Whisper's mishearing "colors" (klrs already,
    # since c->k) never matched before this. Filling a gap in an existing
    # rule, not a new tolerance: v12's demotion now fires with no
    # rule-loosening, same as any other phonetic-match case.
    check("qullars matches its real mishearing colors (q/k skeleton fold)",
          _phonetic_match("qullars", "colors"))
    check("qullars does not match the unrelated-onset mishearing kulas",
          not _phonetic_match("qullars", "kulas"))
    check("existing decentre/dissenter phonetic match is unaffected by the q/k fold",
          _phonetic_match("decentre", "dissenter") and
          not _phonetic_match("decentre", "demolish") and
          not _phonetic_match("europe", "erode"))
    qullars_asr = tokenize(
        "he recruited an army and bureaucracy of colors, or golemani.")
    qullars_mand = ["recruited", "bureaucracy", "qullars", "gholamani"]
    qullars_check = check_mandatory(qullars_asr, qullars_mand)
    check("qullars/colors is not hard-mandatory and is recorded for the morning audit",
          qullars_check["missing"] == [] and
          qullars_check["phonetic_matches"].get("qullars") == "colors",
          repr(qullars_check))

    # v33: curated ASR-fragile lexicon. "colors" already passes via the
    # general v12 mechanism (exact skeleton match after v32's q->k fold);
    # "Kulas" needs the lexicon specifically, since its skeleton ("kls",
    # missing the r sound) is only 3 characters -- below
    # _find_phonetic_match's >= 4-char tolerance floor.
    kulas_expected = tokenize(
        "more than half the Safavid provinces were ruled by qullars.")
    kulas_asr = tokenize(
        "more than half the Safavid provinces were ruled by Kulas.")
    kulas_mand = derive_mandatory(kulas_expected,
                                  "more than half the Safavid provinces were ruled by qullars.")
    kulas_check = check_mandatory(kulas_asr, kulas_mand)
    check("qullars/Kulas passes via the curated lexicon (v12 alone can't reach it)",
          kulas_check["missing"] == [] and
          kulas_check["missing_fragile"] == ["qullars"] and
          kulas_check["lexicon_matches"] == {"qullars": "kulas"} and
          kulas_check["phonetic_matches"].get("qullars") == "kulas",
          repr(kulas_check))
    kulas_cov = ordered_coverage(kulas_expected, kulas_asr, kulas_check["phonetic_matches"])
    kulas_term = terminal_suffix(kulas_expected, kulas_asr,
                                 phonetic_matches=kulas_check["phonetic_matches"])
    check("qullars/Kulas passes coverage and the terminal-phrase gate",
          kulas_cov["missing"] == [] and kulas_term["matched"],
          repr((kulas_cov, kulas_term)))
    check("qullars still hard-fails against an unrelated word in its slot",
          check_mandatory(tokenize("provinces were ruled by governors."),
                          ["qullars"])["missing"] == ["qullars"])
    check("a non-lexicon rare word is unaffected by the wider lexicon tolerance",
          check_mandatory(tokenize("a conscious attempt to demolish europe"),
                          ["decentre"])["missing"] == ["decentre"])

    # v25: coverage/terminal must not contradict a phonetic match the
    # mandatory gate already made and recorded -- no new fuzzy comparison,
    # just not re-litigating the same evidence with a stricter verdict.
    # "Cannalore" (fifteen oh five) and Goa (fifteen ten)." is short enough
    # that one garbled rare name alone drops coverage under 0.85 and blanks
    # the terminal phrase, even though the mandatory gate already accepted it.
    cannalore_expected = tokenize("Cannalore (fifteen oh five) and Goa (fifteen ten).")
    for cannalore_transcript in ("Canelor 1505 and Goa 1510.", "Canelaw 1505 and Goa 1510."):
        cannalore_asr = tokenize(cannalore_transcript)
        cannalore_mand = derive_mandatory(cannalore_expected,
                                          "Cannalore (fifteen oh five) and Goa (fifteen ten).")
        cannalore_result = check_mandatory(cannalore_asr, cannalore_mand)
        cannalore_pm = cannalore_result["phonetic_matches"]
        cannalore_cov = ordered_coverage(cannalore_expected, cannalore_asr, cannalore_pm)
        cannalore_term = terminal_suffix(cannalore_expected, cannalore_asr,
                                         phonetic_matches=cannalore_pm)
        check(f"Cannalore/{cannalore_asr[0]} passes coverage once phonetically matched",
              cannalore_cov["missing"] == [] and cannalore_cov["fraction"] == 1.0,
              repr(cannalore_cov))
        check(f"Cannalore/{cannalore_asr[0]} passes terminal once phonetically matched",
              cannalore_term["matched"], repr(cannalore_term))
    check("a rare word entirely absent (no phonetic candidate) still fails coverage",
          ordered_coverage(tokenize("Cannalore and Goa."),
                           tokenize("and Goa."))["missing"] == ["cannalore"])
    check("a phonetically unrelated wrong word still fails terminal",
          not terminal_suffix(tokenize("Cannalore and Goa."),
                              tokenize("Demolished and Goa."),
                              phonetic_matches={})["matched"])
    long_expected = tokenize(
        "The abrupt abandonment of maritime ventures in the fourteen twenties "
        "signalled part of a much broader problem for the declining empire.")
    long_asr = tokenize(
        "The abrupt abandonment of maritime ventures in the 1420s "
        "signalled part of a much broader problem for the declining empire.")
    check("coverage arithmetic on an ordinary long sentence is unchanged",
          ordered_coverage(long_expected, long_asr, {})["fraction"] == 1.0)

    # v26: curated PHRASE pairs (never a bare word pair) -- a fixed
    # multi-word sequence naming one entity, where one short (<= 3 char)
    # function word inside it has an alternate rendering. The equivalence
    # exists only inside the exact phrase, so a bare "da"/"de" swap
    # anywhere else stays a real mismatch (the v23 guard this respects).
    estado_expected = tokenize(
        "Golden Goa had become the capital of their Estado da India.")
    estado_asr = tokenize(
        "Golden Goa had become the capital of their Estado de India.")
    estado_term = terminal_suffix(estado_expected, estado_asr)
    check("Estado da India / Estado de India passes the terminal-phrase gate",
          estado_term["matched"] and
          estado_term["phrase_pair_matches"] == {"estado da india": "estado de india"},
          repr(estado_term))
    estado_cov = ordered_coverage(estado_expected, estado_asr)
    check("Estado da India / Estado de India passes coverage",
          estado_cov["missing"] == [] and estado_cov["fraction"] == 1.0 and
          estado_cov["phrase_pair_matches"] == {"estado da india": "estado de india"},
          repr(estado_cov))
    check("a bare 'da'/'de' swap outside the curated phrase stays a real mismatch",
          not terminal_suffix(tokenize("he gave da book to her"),
                              tokenize("he gave de book to her"))["matched"])

    # Hyphen-boundary equivalence: a hyphen changes spelling, never sound, so
    # a mandatory word split across adjacent ASR tokens (or vice versa) is a
    # full, exact match -- not a fragile demotion, since there is no
    # precision trade-off the way there is for phonetic matching.
    south_hyphen = check_mandatory(tokenize("the centre retreated south eastward to byzantium"),
                                   ["southeastward"])
    check("southeastward matches adjacent 'south eastward'",
          south_hyphen["missing"] == [] and south_hyphen["missing_fragile"] == [] and
          south_hyphen["hyphen_matches"] == {"southeastward": "south eastward"},
          repr(south_hyphen))
    north_hyphen = check_mandatory(tokenize("pressing in from north eastern limits"),
                                   ["northeastern"])
    check("northeastern matches adjacent 'north eastern'",
          north_hyphen["missing"] == [] and
          north_hyphen["hyphen_matches"] == {"northeastern": "north eastern"},
          repr(north_hyphen))
    reverse_hyphen = check_mandatory(tokenize("retreated southeastward to byzantium"),
                                     ["south-eastward"])
    check("hyphenated source word matches solid transcript word (reverse direction)",
          reverse_hyphen["missing"] == [] and
          reverse_hyphen["hyphen_matches"] == {"south-eastward": "southeastward"},
          repr(reverse_hyphen))
    check("non-adjacent tokens do not concatenate-match",
          check_mandatory(tokenize("south of the region moved eastward"),
                          ["southeastward"])["missing"] == ["southeastward"])
    check("partial concatenation does not match",
          check_mandatory(tokenize("moved south east"),
                          ["southeastward"])["missing"] == ["southeastward"])
    check("unrelated words stay hard failures under concatenation matching",
          check_mandatory(tokenize("a conscious attempt to demolish europe"),
                          ["southeastward"])["missing"] == ["southeastward"])

    # derive_mandatory itself must turn a hyphenated source compound into
    # ONE phrase item, not two independent atomic tokens -- otherwise each
    # half is checked alone and neither matches Whisper's solid rendering
    # ("strongpoints"), even though check_mandatory's hyphen equivalence
    # (_concat_match) exists: two atomic single-token items never combine
    # into the multi-token phrase that equivalence needs.
    strongpoints_text = ("their fortified strong-points, became the "
                          "building blocks for a new round of state-making.")
    strongpoints_expected = tokenize(strongpoints_text)
    strongpoints_mandatory = derive_mandatory(strongpoints_expected, strongpoints_text)
    check("hyphenated compound becomes one phrase item, not two atomic tokens",
          "strong points" in strongpoints_mandatory and
          "strong" not in strongpoints_mandatory and "points" not in strongpoints_mandatory,
          repr(strongpoints_mandatory))
    strongpoints_check = check_mandatory(
        tokenize("their fortified strongpoints, became the building blocks "
                 "for a new round of state-making."),
        strongpoints_mandatory)
    check("derive_mandatory hyphen phrase reaches a full mandatory pass",
          strongpoints_check["missing"] == [] and
          strongpoints_check["hyphen_matches"].get("strong points") == "strongpoints",
          repr(strongpoints_check))
    check("derive_mandatory without expected_text keeps the old atomic behavior",
          derive_mandatory(tokenize("The death of Tamerlane in 1405 was a turning point."))
          == ["tamerlane", "1405", "turning"])

    # A possessive on the hyphenated compound's second half ("Cheng-ho's")
    # must be captured as part of the phrase, not dropped at the hyphen --
    # tokenize() already renders "Cheng-ho's" and "Cheng Ho's" identically
    # (["cheng", "ho's"]), so excluding the "'s" here would manufacture a
    # mismatch against a transcript that never differed in the first place.
    chengho_text = "The abandonment of Cheng-ho's maritime ventures signalled the problem."
    chengho_expected = tokenize(chengho_text)
    chengho_mandatory = derive_mandatory(chengho_expected, chengho_text)
    check("possessive hyphenated compound keeps its 's in the phrase",
          "cheng ho's" in chengho_mandatory, repr(chengho_mandatory))
    chengho_check = check_mandatory(
        tokenize("The abandonment of Cheng Ho's maritime ventures signalled the problem."),
        chengho_mandatory)
    check("possessive hyphenated compound matches the transcript's spaced form",
          chengho_check["missing"] == [], repr(chengho_check))

    # v21: a single canonicalization (ise/ize spelling + compound/adjacency
    # concatenation) applied to both source and transcript token streams,
    # now reaching coverage and terminal too -- not just the mandatory gate.
    wastelands_expected = tokenize("Waste lands were colonized.")
    wastelands_asr = tokenize("Wastelands were colonised.")
    check("spelling canonicalization alone: colonized == colonised",
          tokenize("colonized") == tokenize("colonised") == ["colonised"])
    check("compound canonicalization alone: waste lands == wastelands",
          _concat_reconcile(["waste", "lands"], ["wastelands"]) == ["wastelands"])
    check("both equivalences combined pass coverage",
          ordered_coverage(wastelands_expected, wastelands_asr)["missing"] == [] and
          ordered_coverage(wastelands_expected, wastelands_asr)["fraction"] == 1.0,
          repr(ordered_coverage(wastelands_expected, wastelands_asr)))
    check("both equivalences combined pass terminal",
          terminal_suffix(wastelands_expected, wastelands_asr)["matched"])
    check("a genuinely absent word still fails coverage on a short sentence",
          ordered_coverage(wastelands_expected, tokenize("something completely different"))
          ["fraction"] < 0.85)
    check("a wrong word still fails coverage (colonized vs demolished)",
          "colonised" in ordered_coverage(
              tokenize("waste lands were colonized"),
              tokenize("waste lands were demolished"))["missing"])
    check("a wrong word still fails terminal (colonized vs demolished)",
          not terminal_suffix(tokenize("waste lands were colonized"),
                              tokenize("waste lands were demolished"))["matched"])
    check("non-adjacent concatenation still rejected in coverage",
          _concat_reconcile(["waste", "utterly", "lands"], ["wastelands"])
          == ["waste", "utterly", "lands"])

    # v22: a single vowel-for-vowel substitution (same length, both >= 4
    # chars) is a loanword transliteration variant Whisper prefers
    # ("amir" -> "emir"), not a genuine content difference -- the same
    # vocabulary-snapping phenomenon as v12's phonetic matching, folded
    # into the shared _tok_eq comparator so every gate that uses it
    # (mandatory, terminal, alignment) inherits it consistently.
    check("amir matches its transliteration variant emir",
          _is_vowel_swap("amir", "emir"))
    check("a consonant difference does not count as a vowel swap",
          not _is_vowel_swap("amir", "abir"))
    check("an insertion (length difference) does not count as a vowel swap",
          not _is_vowel_swap("amir", "ameer"))
    check("three-letter pairs are excluded regardless of vowel difference",
          not _is_vowel_swap("cat", "cot"))
    check("a genuinely different real-word pair stays unequal",
          not _is_vowel_swap("amir", "world") and
          not _tok_eq("amir", "world", MANDATORY_FUZZY_RATIO))
    amir_terminal = terminal_suffix(
        tokenize("they owed total loyalty to the amir or ruler."),
        tokenize("they owed total loyalty to the emir or ruler."))
    check("amir/emir passes the terminal-phrase gate",
          amir_terminal["matched"], repr(amir_terminal))
    amir_mandatory = check_mandatory(tokenize("the emir ruled wisely"), ["amir"])
    check("amir/emir is recorded in vowel_matches for the morning audit",
          amir_mandatory["missing"] == [] and
          amir_mandatory["vowel_matches"] == {"amir": "emir"},
          repr(amir_mandatory))

    # v23: a curated pair list for transliteration variants that aren't a
    # single vowel swap ("koranic"/"quranic" differ by a consonant AND a
    # vowel: k/q, o/u) -- v22's general rule can't reach these, so each
    # pair is individually justified as two spellings of the same referent.
    check("koran matches its transliteration variant quran",
          _is_transliteration_pair("koran", "quran"))
    check("koranic matches its transliteration variant quranic",
          _is_transliteration_pair("koranic", "quranic"))
    check("genghis matches its transliteration variant chinggis",
          _is_transliteration_pair("genghis", "chinggis"))
    check("cracow matches its transliteration variant krakow",
          _is_transliteration_pair("cracow", "krakow"))
    cracow_mandatory = check_mandatory(
        tokenize("the first book was printed in Krakow in 1423."),
        derive_mandatory(tokenize("the first book was printed in Cracow in fourteen twenty-three."),
                         "the first book was printed in Cracow in fourteen twenty-three."))
    check("cracow/krakow is not hard-mandatory and is recorded for the morning audit",
          cracow_mandatory["missing"] == [] and
          cracow_mandatory["transliteration_matches"].get("cracow") == "krakow",
          repr(cracow_mandatory))
    check("an unrelated word is not a curated transliteration pair",
          not _is_transliteration_pair("koran", "random") and
          not _tok_eq("koran", "random", MANDATORY_FUZZY_RATIO))
    koranic_terminal = terminal_suffix(
        tokenize("their ultimate loyalty was to Koranic law which they interpreted."),
        tokenize("their ultimate loyalty was to Quranic law which they interpreted."))
    check("koranic/quranic passes the terminal-phrase gate",
          koranic_terminal["matched"], repr(koranic_terminal))
    koranic_mandatory = check_mandatory(tokenize("their loyalty was to Quranic law"),
                                        ["koranic"])
    check("koranic/quranic is recorded in transliteration_matches for the morning audit",
          koranic_mandatory["missing"] == [] and
          koranic_mandatory["transliteration_matches"] == {"koranic": "quranic"},
          repr(koranic_mandatory))

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
    circa_source = ("Until c. eighteen hundred it looked as if a variety of "
                     "factors would prevent a similar pattern in other parts "
                     "of the world.")
    circa_boundaries = [b for b in _source_punctuation_boundaries(circa_source)
                        if b["kind"] == "sentence_end"]
    check("circa abbreviation (c.) produces exactly one real sentence end",
          len(circa_boundaries) == 1, repr(circa_boundaries))

    # A colon introducing a complement clause ("held good: that European
    # depictions...") reads straight through -- exempt from the 100ms
    # colon/semicolon minimum, advisory only, like a plain comma. A colon
    # before a list, or one followed by a capitalized "That" (a new
    # independent clause), keeps the normal 100ms requirement.
    complementizer_source = ("But the broader point held good: that European "
                              "depictions of other parts of the world needed "
                              "very careful decoding.")
    complementizer_words = [
        {"text": t, "start": i * 0.3, "end": i * 0.3 + 0.05}
        for i, t in enumerate(tokenize(complementizer_source))
    ]
    complementizer_punct = punctuation_metrics(complementizer_source, complementizer_words)
    colon_boundary = next(b for b in complementizer_punct["boundaries"]
                          if b["kind"] == "colon_semicolon")
    check("colon before a lowercase 'that' complement clause is advisory only",
          colon_boundary["colon_complementizer_that"] is True and
          colon_boundary["threshold_s"] is None and colon_boundary["passed"] is None,
          repr(colon_boundary))
    list_colon_source = "He listed the ingredients: flour, sugar, and salt."
    list_colon_words = [
        {"text": t, "start": i * 0.3, "end": i * 0.3 + 0.05}
        for i, t in enumerate(tokenize(list_colon_source))
    ]
    list_colon_punct = punctuation_metrics(list_colon_source, list_colon_words)
    list_colon_boundary = next(b for b in list_colon_punct["boundaries"]
                               if b["kind"] == "colon_semicolon")
    check("colon before a list keeps the 100ms threshold",
          list_colon_boundary["colon_complementizer_that"] is False and
          list_colon_boundary["threshold_s"] == PUNCTUATION_THRESHOLDS["colon_semicolon"],
          repr(list_colon_boundary))
    capitalized_that_source = "She insisted: That decision was final."
    capitalized_that_words = [
        {"text": t, "start": i * 0.3, "end": i * 0.3 + 0.05}
        for i, t in enumerate(tokenize(capitalized_that_source))
    ]
    capitalized_that_punct = punctuation_metrics(capitalized_that_source, capitalized_that_words)
    capitalized_that_boundary = next(b for b in capitalized_that_punct["boundaries"]
                                     if b["kind"] == "colon_semicolon")
    check("colon before capitalized 'That' (independent clause) keeps the 100ms threshold",
          capitalized_that_boundary["colon_complementizer_that"] is False and
          capitalized_that_boundary["threshold_s"] == PUNCTUATION_THRESHOLDS["colon_semicolon"],
          repr(capitalized_that_boundary))

    # A semicolon closing a serial list's final item ("a space; a
    # community; and a programme") reads straight through -- the
    # conjunction itself marks the last item. Only lowercase "and"/"or"
    # immediately after the semicolon qualifies; anything else (a noun
    # phrase, "but", an independent clause) keeps the normal 100ms gate,
    # including a non-final semicolon in the very same list.
    def _semicolon_boundary(text):
        words = [{"text": t, "start": i * 0.3, "end": i * 0.3 + 0.05}
                 for i, t in enumerate(tokenize(text))]
        boundaries = [b for b in punctuation_metrics(text, words)["boundaries"]
                     if b["kind"] == "colon_semicolon"]
        return boundaries[-1]

    serial_and = _semicolon_boundary(
        "a geographical space; a socio-political community; and a cultural programme.")
    check("semicolon before final 'and' item is advisory only",
          serial_and["semicolon_serial_conjunction"] is True and
          serial_and["threshold_s"] is None and
          serial_and["exemption_reason"] == "semicolon_serial_conjunction",
          repr(serial_and))
    serial_or = _semicolon_boundary("the treaty covered trade; or the state would collapse.")
    check("semicolon before final 'or' item is advisory only",
          serial_or["semicolon_serial_conjunction"] is True and serial_or["threshold_s"] is None,
          repr(serial_or))
    serial_list_source = "a geographical space; a socio-political community; and a cultural programme."
    serial_list_words = [{"text": t, "start": i * 0.3, "end": i * 0.3 + 0.05}
                         for i, t in enumerate(tokenize(serial_list_source))]
    serial_list_semicolons = [b for b in punctuation_metrics(serial_list_source, serial_list_words)["boundaries"]
                              if b["kind"] == "colon_semicolon"]
    non_final_semicolon = serial_list_semicolons[0]
    check("non-final semicolon in the same list keeps the 100ms threshold",
          non_final_semicolon["semicolon_serial_conjunction"] is False and
          non_final_semicolon["threshold_s"] == PUNCTUATION_THRESHOLDS["colon_semicolon"],
          repr(non_final_semicolon))
    check("semicolon before a noun phrase stays hard-gated",
          _semicolon_boundary("community; the programme was launched.")["threshold_s"]
          == PUNCTUATION_THRESHOLDS["colon_semicolon"])
    check("semicolon before 'but' stays hard-gated",
          _semicolon_boundary("a geographical space; but the state intervened.")["threshold_s"]
          == PUNCTUATION_THRESHOLDS["colon_semicolon"])

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
          serialized["terminal"] ==
          {"expected": "alpha omega", "matched": True, "phrase_pair_matches": {}})
    missing_terminal = terminal_suffix(tokenize("alpha omega"), tokenize("alpha"))
    check("serialized record preserves terminal missing state",
          json.loads(json.dumps({"terminal": missing_terminal}))["terminal"]["matched"] is False)
    check("serialized record preserves exact punctuation schema",
          set(serialized["punctuation"]) == {"boundaries", "sentence_end", "colon_semicolon",
                                               "parenthetical_comma", "comma"} and
          set(serialized["punctuation"]["boundaries"][0]) == {
              "kind", "punctuation", "expected_token_index", "sentence_word_count",
              "asr_token_index", "next_asr_token_index", "aligned", "gap_s",
              "threshold_s", "passed", "exemption_reason"})
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

    # v31: speak-time text normalization for the "Name + roman numeral"
    # mispronunciation hazard (see normalize_for_tts's docstring -- traced
    # from ch02:p0055, "Abbas I," rendered as a short unrelated syllable).
    abbas_source = ("However, the accession of Abbas I, the fifth Safavid "
                    "shah, in fifteen eighty-seven marked the onset of a "
                    "political revolution.")
    abbas_speak = normalize_for_tts(abbas_source)
    check("Abbas I becomes Abbas the First in speak_text",
          "Abbas the First," in abbas_speak and "Abbas I," not in abbas_speak,
          repr(abbas_speak))
    for abbas_transcript in (
        "However, the accession of Abbas the First, the fifth Safavid "
        "Shah, in 1587, marked the onset of a political revolution.",
        "However, the accession of Abbas I, the fifth Safavid Shah, in "
        "1587, marked the onset of a political revolution.",
    ):
        abbas_exp = tokenize(abbas_speak)
        abbas_asr = tokenize(abbas_transcript)
        abbas_mand = derive_mandatory(abbas_exp, abbas_speak)
        abbas_mres = check_mandatory(abbas_asr, abbas_mand)
        abbas_cov = ordered_coverage(abbas_exp, abbas_asr, abbas_mres["phonetic_matches"])
        abbas_term = terminal_suffix(abbas_exp, abbas_asr,
                                     phonetic_matches=abbas_mres["phonetic_matches"])
        check(f"Abbas sentence validates end-to-end against {abbas_transcript[-30:]!r}",
              abbas_mres["missing"] == [] and abbas_cov["missing"] == [] and
              abbas_term["matched"],
              repr((abbas_mres["missing"], abbas_cov["missing"], abbas_term)))
    check("Suleiman II becomes Suleiman the Second",
          normalize_for_tts("Suleiman II succeeded his father.") ==
          "Suleiman the Second succeeded his father.")
    check("World War II becomes World War Two (cardinal exclusion list)",
          normalize_for_tts("World War II reshaped the continent.") ==
          "World War Two reshaped the continent.")
    check("Part III becomes Part Three (cardinal exclusion list)",
          normalize_for_tts("Part III covers the aftermath.") ==
          "Part Three covers the aftermath.")
    # Bare "I" not followed by punctuation now fires too (Elizabeth I
    # constructed -> Elizabeth the First constructed), resolved against a
    # whole-book grep rather than a guess -- see _PRONOUN_CONTEXT_WORDS.
    check("Elizabeth I constructed becomes Elizabeth the First constructed",
          normalize_for_tts(
              "the urgency with which Elizabeth I constructed the Anglican "
              "via media in England.") ==
          "the urgency with which Elizabeth the First constructed the "
          "Anglican via media in England.")
    check("Abbas I had been becomes Abbas the First had been (ch03 occurrence)",
          normalize_for_tts("the imperial legacy of Abbas I had been summarily dissolved.") ==
          "the imperial legacy of Abbas the First had been summarily dissolved.")
    check("the one genuine pronoun context the grep found stays unchanged",
          normalize_for_tts(
              "Thus I have used Constantinople and not Istanbul throughout.") ==
          "Thus I have used Constantinople and not Istanbul throughout.")
    check("genuine pronoun context, synthesized (sentence-initial 'I argue')",
          normalize_for_tts("Thus I argue that this changed everything.") ==
          "Thus I argue that this changed everything.")
    check("pronoun guard: '...and I said' unchanged (lowercase 'and' never matches)",
          normalize_for_tts("...and I said nothing at all.") ==
          "...and I said nothing at all.")
    check("plain text with no numerals is byte-identical",
          normalize_for_tts("The death of Tamerlane in 1405.") ==
          "The death of Tamerlane in 1405.")
    check("bare 'I.' after a name still fires (punctuation present)",
          normalize_for_tts("Meet Agent I.") == "Meet Agent the First.")

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
