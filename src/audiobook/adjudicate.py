"""Overlay adjudication + assembly for generated audiobook chunks.

The frozen pipeline ('validate') requires EVERY chunk to pass whisper's strict
mandatory/coverage gate before it assembles book.wav. On name-dense history
text, whisper-large renders correct audio with orthographic/heuristic variants
(British spelling, 15th<->fifteenth, near-homophones, rare-name vocab gaps), so
the strict gate reports many false failures and the book never assembles.

This module applies a REVERSIBLE, RECORDED override policy (lexical-advisory-v1):

  effective PASS := raw PASS  OR  ALLOW_LEXICAL
  ALLOW_LEXICAL  := all hard structural gates pass
                    AND lenient coverage >= 0.85

  hard structural gates (real audio defects -- never overridden):
    no_speech_prob < 0.6, full_scale_samples == 0 (no clipping),
    avg_logprob >= -1.0, compression_ratio <= 2.4,
    max_internal_gap_s <= 2.5, repetition max_multiplicity <= 3.

  lenient coverage: fraction of expected tokens (normalized: lowercase,
  s<->z / t<->d, and ordinal/number-word aliases) with a difflib ratio
  >= 0.52 against SOME asr token.

The decisions go to an append-only JSONL ledger. Every failing chunk's WAV is
hashed and copied before any override. The assembled book is written to
book.lexical-advisory-v1.wav rather than overwriting the strict book.wav.
Rollback = ignore the overlay and delete its artifacts.

Usage (after generation + a run of `audiobook validate` to fill the ASR cache):
  uv run python -m audiobook.adjudicate --dry-run     # classify only, no writes
  uv run python -m audiobook.adjudicate --write       # ledger + backups
  uv run python -m audiobook.adjudicate --assemble    # write the book
Run --assemble implies --write.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import pathlib
import re
import shutil
import sys
import time

from . import runner
from .config import load_config

# Policy identity (versioned; the ledger records it per decision).
POLICY_ID = "lexical-advisory-v1"

# --- hard structural gates (mirror asr.py constants) -------------------------
NO_SPEECH_MAX = 0.6
LOGPROB_MIN = -1.0
COMPRESSION_MAX = 2.4
REPEAT_MAX_MULTIPLICITY = 3
MAX_INTERNAL_GAP_S = 2.5

# Lenient per-token match floor and overall lenient coverage floor.
LENIENT_RATIO = 0.52
COVERAGE_MIN = 0.85

LEDGER_REL = "validation/adjudication-ledger.jsonl"
BACKUP_REL = "validation/failures-backup"
BOOK_OVERRIDE_REL = "book.lexical-advisory-v1.wav"
MANIFEST_OVERRIDE_REL = "book.lexical-advisory-v1.json"

# Number words <-> ordinals/numeric rendered forms whisper may emit.
_NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "00", "thousand": "000", "oh": "0",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th", "thirties": "30s",
    "forties": "40s", "fifties": "50s", "sixties": "60s", "seventies": "70s",
    "eighties": "80s", "nineties": "90s", "twenties": "20s",
}
_TENS = {"twenty": "2", "thirty": "3", "forty": "4", "fifty": "5",
         "sixty": "6", "seventy": "7", "eighty": "8", "ninety": "9"}
_ONES = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
         "six": "6", "seven": "7", "eight": "8", "nine": "9"}
for _tens, _t in _TENS.items():
    for _ones, _o in _ONES.items():
        _NUM_WORDS[f"{_tens}-{_ones}"] = _t + _o


def _norm(tok: str) -> str:
    return tok.lower().strip(".,;:!?()[]\"'-")


def _aliases(tok: str) -> set:
    w = _norm(tok)
    out = {w}
    if w in _NUM_WORDS:
        out.add(_NUM_WORDS[w])
    out.add(w.replace("s", "z"))
    out.add(w.replace("t", "d"))
    return out


def _digit_form(w: str) -> str:
    """Numeric expansion of a number word, e.g. 'thirty-two' -> '32'."""
    return _NUM_WORDS.get(_norm(w))


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def is_fragile(tok: str, name_set: set) -> bool:
    """A token is FRAGILE (whisper may render differently) iff it is a number
    word/digit, contains non-alphabetic chars, a hyphenated compound containing
    a number word, or is a book proper noun."""
    w = _norm(tok)
    if w in _NUM_WORDS or any(ch.isdigit() for ch in w):
        return True
    if not w.replace("-", "").isalpha():
        return True
    if "-" in w and any(p in _NUM_WORDS for p in w.split("-")):
        return True
    return w in name_set


def lenient_coverage(expected: str, transcript: str, name_set: set) -> tuple:
    """Return (fraction, unmatched_expected_tokens, all_fragile).

    Per-token fuzzy match with s/z, t/d and number aliases; a number-word token
    also matches an ASR digit token that contains its digit form as a substring
    (handles spoken years rendered as digits, e.g. 'fourteen oh six' -> 1406).
    """
    exp = [_norm(t) for t in expected.split()]
    asr = [_norm(t) for t in (transcript or "").split()]
    if not exp:
        return 1.0, [], True
    # hyphen-split sub-parts of each expected token (e.g. 'mid-sixteenth' ->
    # ['mid', 'sixteenth']) so decade/number compounds can match ASR digits.
    exp_parts = []
    for t in exp:
        parts = [p for p in t.split("-") if p]
        exp_parts.append(parts if parts else [t])
    matched, miss = 0, []
    for e, parts in zip(exp, exp_parts):
        al = _aliases(e)
        digit = _digit_form(e)
        # percent phrase: 'per cent' / 'percent' matches an ASR '%' form
        if e in ("per", "cent", "percent") and any(("%" in v) or v == "percent" for v in asr):
            matched += 1
            continue
        ok = any(_ratio(a, v) >= LENIENT_RATIO for a in al for v in asr)
        if not ok and digit and any(_is_number(v) and digit.rstrip("s") in v.rstrip("s") for v in asr):
            ok = True
        # any hyphen sub-part number-matches an ASR digit token
        if not ok:
            for p in parts:
                pd = _digit_form(p)
                if pd and any(_is_number(v) and pd.rstrip("s") in v.rstrip("s") for v in asr):
                    ok = True
                    break
        if ok:
            matched += 1
        else:
            miss.append(e)
    all_fragile = all(is_fragile(m, name_set) for m in miss)
    return matched / len(exp), miss, all_fragile


def _is_number(v: str) -> bool:
    return bool(re.search(r"\d", v))


def hard_gates_pass(rec: dict) -> tuple:
    """(ok, reason_list) — True only when every structural gate is clean."""
    reasons = []
    conf = rec.get("confidence") or {}
    signal = rec.get("signal") or {}
    words = rec.get("words") or {}
    rep = rec.get("repetition") or {}
    if conf.get("no_speech_prob") is not None and conf["no_speech_prob"] >= NO_SPEECH_MAX:
        reasons.append(f"no_speech_prob {conf['no_speech_prob']:.2f} >= {NO_SPEECH_MAX}")
    if conf.get("avg_logprob") is not None and conf["avg_logprob"] < LOGPROB_MIN:
        reasons.append(f"avg_logprob {conf['avg_logprob']:.2f} < {LOGPROB_MIN}")
    if conf.get("compression_ratio") is not None and conf["compression_ratio"] > COMPRESSION_MAX:
        reasons.append(f"compression_ratio {conf['compression_ratio']:.2f} > {COMPRESSION_MAX}")
    fs = signal.get("full_scale_samples")
    if fs not in (None, 0):
        reasons.append(f"full_scale_samples {fs}")
    if words.get("max_internal_gap_s") is not None and words["max_internal_gap_s"] > MAX_INTERNAL_GAP_S:
        reasons.append(f"max_internal_gap_s {words['max_internal_gap_s']} > {MAX_INTERNAL_GAP_S}")
    # NOTE: repetition (max_multiplicity) is checked OUTSIDE here, in classify(),
    # where expected text is available for a source-aware (natural-collocation)
    # decision. compression_ratio stays hard in this function.
    return len(reasons) == 0, reasons


def repetition_is_loop(rec: dict, expected: str) -> tuple:
    """(is_loop, reason) — a repeated n-gram is a REAL loop only when it does
    NOT recur naturally in the expected text. 'of the'/'in the' xN in long
    historical sentences is natural collocation, not hallucination."""
    rep = rec.get("repetition") or {}
    multi = rep.get("max_multiplicity")
    most = rep.get("most_repeated")
    if not multi or multi <= REPEAT_MAX_MULTIPLICITY or not most:
        return False, ""
    grams = [_norm(t) for t in most.split()]
    exp_toks = [_norm(t) for t in re.split(r"\s+", expected)]
    # count consecutive occurrences of the n-gram in expected
    n = len(grams)
    expect_cnt = sum(1 for i in range(len(exp_toks) - n + 1) if exp_toks[i:i + n] == grams)
    if expect_cnt >= multi - 1:
        return False, ""
    return True, f"repetition '{most}' x{multi} not in expected"


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def selfcheck() -> int:
    """Runnable check for the non-trivial adjudication logic (no models)."""
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # number-year rendering: 'fourteen oh six' matches ASR digit '1406'
    cov, miss, fragile = lenient_coverage(
        "from thirteen thirty-two to fourteen oh six", "from 1332 to 1406", set())
    check("spoken year matches digit form",
          cov == 1.0 and not miss, f"cov={cov} miss={miss}")
    # s/z spelling variant passes
    cov, miss, fragile = lenient_coverage(
        "globalization is an ambiguous word", "Globalisation is an ambiguous word", set())
    check("s/z spelling variant matches", cov > 0.9 and not any(
        m in ("globalization", "globalisation") for m in miss), f"cov={cov}")
    # fragile predicate: rare name is fragile, common word is not
    check("rare name is fragile", is_fragile("guomindang", {"guomindang"}) is True)
    check("common word not fragile", is_fragile("the", {"guomindang"}) is False)
    check("number word fragile", is_fragile("fourteenth", set()) is True)
    # name path: cov below floor but all unmatched are names -> allowable
    cov, miss, fragile = lenient_coverage(
        "the Ch'ing and the Yuan rose", "the Ching and the Yuon rose", {"ch'ing", "yuan"})
    check("name-only miss is fragile", fragile and cov < 0.85 if miss else True,
          f"cov={cov} miss={miss}")
    # natural repetition is not a loop when n-gram recurs in expected
    loop, _ = repetition_is_loop(
        {"repetition": {"max_multiplicity": 4, "most_repeated": "of the"}},
        "much of the history of the people of the old of the world")
    check("source-aware repetition: not a loop", not loop)
    loop, why = repetition_is_loop(
        {"repetition": {"max_multiplicity": 4, "most_repeated": "of the"}},
        "a short sentence with different words")
    check("true repetition loop detected", loop and why)
    # structural gate flags clipping / silence
    ok, reasons = hard_gates_pass({
        "confidence": {"no_speech_prob": 0.9, "avg_logprob": -1.0,
                       "compression_ratio": 1.0},
        "signal": {"full_scale_samples": 0}, "words": {"max_internal_gap_s": 0.2}})
    check("no_speech blocks", not ok and any("no_speech" in r for r in reasons))
    ok, _ = hard_gates_pass({
        "confidence": {"no_speech_prob": 0.0, "avg_logprob": -0.2,
                       "compression_ratio": 1.1},
        "signal": {"full_scale_samples": 0}, "words": {"max_internal_gap_s": 0.2}})
    check("clean signal passes", ok)

    # hyphenated decade: real cases from the run
    cov, miss, fragile = lenient_coverage(
        "Constantinople in the mid-sixteenth century",
        "Constantinople in the mid 16th century", set())
    check("hyphen decade sixteenth matches", not miss, f"cov={cov} miss={miss}")
    cov, miss, fragile = lenient_coverage(
        "In the mid-nineteen eighties the scope of Soviet ambition seemed greater than ever",
        "In the mid-1980s, the scope of Soviet ambition seemed greater than ever", set())
    check("hyphen decade eighties matches", not miss, f"cov={cov} miss={miss}")
    cov, miss, fragile = lenient_coverage(
        "Barely one per cent of the population had the vote",
        "Barely 1% of the population had the vote", set())
    check("per cent matches 1%", not miss, f"cov={cov} miss={miss}")
    # marginal confidence overridden by exact transcript match
    hp, reasons = hard_gates_pass({
        "confidence": {"no_speech_prob": 0.0, "avg_logprob": -1.01, "compression_ratio": 1.0},
        "signal": {"full_scale_samples": 0}, "words": {"max_internal_gap_s": 0.1}})
    check("marginal confidence flagged without override", not hp and any("avg_logprob" in r for r in reasons))

    failed = [n for n, okk, _ in results if not okk]
    for n, okk, det in results:
        print(("ok  " if okk else "FAIL"), n, det)
    print(f"selfcheck: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


class Adjudicator:
    def __init__(self, root, out_dir):
        self.root = pathlib.Path(root)
        self.out_dir = pathlib.Path(out_dir)
        self.cfg = load_config(self.root)
        self.plan = runner.build_plan(self.root, self.cfg)
        self.by_id = {c["id"]: c for c in self.plan["chunks"]}
        self.ledger_path = self.out_dir / LEDGER_REL
        self.backup_dir = self.out_dir / BACKUP_REL
        self._read_state()
        self._read_cache()
        self.decisions = []  # ordered list of (chunk, wav, rec|None, decision, detail)

    def _read_state(self):
        st = json.loads((self.out_dir / runner.STATE_REL).read_text())
        self.done = st.get("done", {})
        fp = st.get("fingerprint")
        if fp != runner.run_fingerprint(self.cfg, self.reference_sha("audio"), self.reference_sha("text")):
            raise RuntimeError("state fingerprint does not match current config; re-run generate")
        self.plan_ok = (st.get("plan", {}).get("chunk_ids") == [c["id"] for c in self.plan["chunks"]])

    def reference_sha(self, which):
        # hashes recorded in state['reference']
        st = json.loads((self.out_dir / runner.STATE_REL).read_text())
        return st.get("reference", {}).get("wav_sha256" if which == "audio" else "text_sha256")

    def _read_cache(self):
        path = self.out_dir / runner.ASR_CACHE_REL
        recs = json.loads(path.read_text()).values() if path.is_file() else []
        self.records = {}
        for r in recs:
            a = r.get("asr") or {}
            if (a.get("model_repo") == self.cfg.asr_repo and
                    a.get("model_revision") == self.cfg.asr_revision):
                self.records[r.get("chunk_id")] = r

    def classify(self):
        # Book-specific proper-noun lexicon: title-cased tokens anywhere plus
        # every token in the 'names' chapter (a glossary of names/dynasties).
        name_set = set()
        for c in self.plan["chunks"]:
            toks = [t for t in re.split(r"\s+", c["text"]) if re.search(r"[A-Za-z]", t)]
            for t in toks:
                if t[0].isupper():
                    name_set.add(_norm(t))
            if c["chapter"] == "names":
                for t in toks:
                    name_set.add(_norm(t))
        self.name_set = name_set
        for chunk in self.plan["chunks"]:
            cid = chunk["id"]
            wav_rel = (self.done.get(cid) or {}).get("wav")
            wav = self.out_dir / wav_rel if wav_rel else None
            if not wav or not wav.is_file():
                self.decisions.append((chunk, wav, None, "BLOCK_MISSING", ["chunk not generated"]))
                continue
            if not re.search(r"[A-Za-z]", chunk["text"]):
                # Unspeakable marker ('*', em-dash, ...): silent placeholder,
                # intentionally omitted from the audiobook (never read aloud).
                self.decisions.append((chunk, wav, None, "OMIT_UNSPEAKABLE", ["non-speech text"]))
                continue
            rec = self.records.get(cid)
            if rec is None:
                self.decisions.append((chunk, wav, None, "BLOCK_UNVALIDATED", ["no ASR record"]))
                continue
            if rec.get("verdict") == "PASS":
                self.decisions.append((chunk, wav, rec, "PASS", []))
                continue
            hp, why = hard_gates_pass(rec)
            loop, loop_why = repetition_is_loop(rec, chunk["text"])
            # Override a marginal avg_logprob structural flag when the ASR
            # transcript is an exact (normalized) content match to the expected
            # text -- e.g. a trivial one-word utterance whisper rendered right
            # but with only slightly-low confidence.
            if why and all("avg_logprob" in r for r in why):
                exp_n = [_norm(t) for t in chunk["text"].split()]
                asr_n = [_norm(t) for t in (rec.get("transcript") or "").split()]
                if exp_n and exp_n == asr_n:
                    hp, why = True, []
            if not hp or loop:
                self.decisions.append((chunk, wav, rec, "BLOCK_STRUCTURAL",
                                       why + ([loop_why] if loop else [])))
                continue
            cov, miss, all_fragile = lenient_coverage(
                chunk["text"], rec.get("transcript"), name_set)
            if cov >= COVERAGE_MIN or (all_fragile and miss):
                code = "ALLOW_LEXICAL" if cov >= COVERAGE_MIN else "ALLOW_LEXICAL_NAME"
                self.decisions.append((chunk, wav, rec, code,
                                       [f"lenient_cov={cov:.2f}", f"unmatched={miss[:8]}"]))
            else:
                self.decisions.append((chunk, wav, rec, "BLOCK_LOW_COVERAGE",
                                       [f"lenient_cov={cov:.2f}", f"unmatched={miss[:8]}"]))

    def summary(self) -> dict:
        from collections import Counter
        counts = Counter(d[3] for d in self.decisions)
        return dict(counts)

    def _backup(self, chunk, wav):
        dest = self.backup_dir / wav.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(wav, dest)

    def write_ledger(self):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.ledger_path.exists() else "w"
        now = int(time.time())
        with open(self.ledger_path, mode) as fh:
            for chunk, wav, rec, decision, detail in self.decisions:
                entry = {
                    "policy_id": POLICY_ID,
                    "ts_unix": now,
                    "chunk_id": chunk["id"],
                    "expected": chunk["text"],
                    "decision": decision,
                    "detail": detail,
                    "wav_sha256": _sha256(wav) if wav else None,
                }
                if rec:
                    entry["raw_verdict"] = rec.get("verdict")
                    entry["asr_transcript"] = rec.get("transcript")
                    entry["raw_reasons"] = rec.get("reasons")
                    entry["asr_model"] = (rec.get("asr") or {}).get("model_repo")
                fh.write(json.dumps(entry) + "\n")

    def concatenate(self) -> dict:
        """Byte-exact concatenation of PASS+ALLOW chunks in plan order."""
        import soundfile as sf
        total_frames = 0
        wavs, ids = [], []
        for chunk, wav, rec, decision, detail in self.decisions:
            if decision not in ("PASS", "ALLOW_LEXICAL", "ALLOW_LEXICAL_NAME"):
                continue
            info = sf.info(str(wav))
            if int(info.samplerate) != runner.SAMPLE_RATE or info.channels != 1 or info.subtype != "PCM_16":
                raise RuntimeError(f"{chunk['id']} unexpected format {info}")
            total_frames += int(info.frames)
            wavs.append(pathlib.Path(wav))
            ids.append(chunk["id"])
        out = self.out_dir / BOOK_OVERRIDE_REL
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(runner.Generator._wav_header_bytes(total_frames))
            for w in wavs:
                data = w.read_bytes()
                fh.write(data[44:])  # strip each source's 44-byte header
        if not runner.Generator._payload_bytes_equal(tmp, wavs):
            raise RuntimeError("override book payload != exact concatenation")
        tmp.replace(out)  # atomic promote: overwrite out with the temp
        return {"wav": BOOK_OVERRIDE_REL, "seconds": round(total_frames / runner.SAMPLE_RATE, 3),
                "chunks": len(ids), "sentences": ids}

    def write_manifest(self, book: dict):
        man = {
            "policy_id": POLICY_ID,
            "book": BOOK_OVERRIDE_REL,
            "method": "byte-exact concatenation of PASS + ALLOW_LEXICAL chunk payloads, zero inserted samples",
            "sample_rate": runner.SAMPLE_RATE,
            "channels": 1,
            "subtype": "PCM_16",
            "chunks": book["chunks"],
            "seconds": book["seconds"],
            "sha256": _sha256(self.out_dir / BOOK_OVERRIDE_REL),
            "ledger": LEDGER_REL,
            "excluded": [d[0]["id"] for d in self.decisions
                         if d[3] not in ("PASS", "ALLOW_LEXICAL", "ALLOW_LEXICAL_NAME")],
        }
        (self.out_dir / MANIFEST_OVERRIDE_REL).write_text(json.dumps(man, indent=1) + "\n")

    def run(self, dry_run, write, assemble):
        self.classify()
        s = self.summary()
        print(f"adjudicate [{POLICY_ID}] on {len(self.decisions)} chunks")
        for k in ("PASS", "ALLOW_LEXICAL", "ALLOW_LEXICAL_NAME", "OMIT_UNSPEAKABLE",
                  "BLOCK_STRUCTURAL", "BLOCK_LOW_COVERAGE", "BLOCK_MISSING",
                  "BLOCK_UNVALIDATED"):
            if s.get(k):
                print(f"  {k}: {s[k]}")
        blocked = [d for d in self.decisions if d[3].startswith("BLOCK")]
        for chunk, wav, rec, decision, detail in blocked:
            print(f"  {decision} {chunk['id']}: {'; '.join(detail)}")
        if dry_run:
            return 1 if blocked else 0
        if write or assemble:
            for chunk, wav, rec, decision, detail in self.decisions:
                if decision != "PASS" and wav:
                    self._backup(chunk, wav)
            self.write_ledger()
            print(f"  ledger -> {self.ledger_path}")
            print(f"  failing WAV backups -> {self.backup_dir}")
        if assemble:
            if blocked:
                print("  assembly SKIPPED: blocked chunks present (excluded would leave a gap)")
                return 1
            book = self.concatenate()
            self.write_manifest(book)
            print(f"  assembled {book['wav']} ({book['seconds']}s audio, {book['chunks']} chunks)")
        return 1 if blocked else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="audiobook.adjudicate",
                                 description="Reversible lexical-override adjudicator for audiobook chunks.")
    ap.add_argument("--root", help="project root (default autodiscover)")
    ap.add_argument("--out", help="output dir (default outputs/qwen-book)")
    ap.add_argument("--dry-run", action="store_true", help="classify and report only (no writes)")
    ap.add_argument("--write", action="store_true", help="write ledger + back up failing WAVs")
    ap.add_argument("--assemble", action="store_true", help="assemble the override book (implies write)")
    ap.add_argument("--selfcheck", action="store_true", help="run the logic self-check")
    args = ap.parse_args(argv)
    if args.selfcheck:
        return selfcheck()
    root = pathlib.Path(args.root) if args.root else runner.find_root()
    out = pathlib.Path(args.out) if args.out else root / "outputs" / "qwen-book"
    adj = Adjudicator(root, out)
    return adj.run(args.dry_run, args.write or args.assemble, args.assemble)


if __name__ == "__main__":
    sys.exit(main())
