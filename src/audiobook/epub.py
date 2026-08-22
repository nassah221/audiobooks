"""EPUB extraction for the frozen After Tamerlane run (pure stdlib).

Reads the EPUB spine (zipfile + xml.etree), selects the Preface, the note
on names and places, and chapters 1-9 (continuation files like ``ch01a`` are
calibre splits of the same chapter), and excludes notes, further reading,
index, and all other front matter. Footnote links and sup markers are
stripped without touching prose digits, page-number cross references are
removed, continuation files that calibre split mid-paragraph are merged
back, and typography and numbers are normalized with the explicit rules
below. Paragraph boundaries from the EPUB are preserved for generation.

Normalization rules (explicit, frozen):
  N1  Double quotation marks (U+201C/U+201D, ASCII ") are removed.
  N2  U+2019 / ASCII ' are kept as apostrophes only between word
      characters ("Tamerlane's"); in quotation position they are removed
      ("'global'" -> "global", matching the frozen pilot curation).
  N3  A spaced en/em dash (" - ", " -- ") becomes ", " (frozen comma style).
  N4  Whitespace runs collapse to a single space.
  N5  Digit runs expand to words:
      - years 1000-2099 read in pairs: "1405" -> "fourteen oh five",
        "1970" -> "nineteen seventy", "1900" -> "nineteen hundred",
        "1005" -> "ten oh five";
      - year ranges "1332-1406", "1945-89", "1482-4" read A "to" B with the
        elided end rebuilt from A's century ("nineteen forty-five to
        nineteen eighty-nine");
      - decades "1330s" pluralize the year reading ("thirteen thirties");
        3-digit decades pluralize the plain reading ("400s" -> "four
        hundreds");
      - decimals "1.3" read "one point three";
      - other integers read as ordinary words ("600,000" -> "six hundred
        thousand", British "and").
"""
import hashlib
import argparse
import posixpath
import re
import pathlib
import sys
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree

__all__ = [
    "SELECTION",
    "read_spine",
    "extract_chapters",
    "chapter_paragraphs",
    "expand_numbers",
    "split_sentences",
    "sentence_spans",
    "clause_boundaries",
    "clause_spans",
    "group_sentences",
    "sentence_chunks",
]


# href basename -> canonical chapter id (spine order: preface, names, ch01..ch09)
SELECTION = {
    "frontm_split_008.html": "preface",  # Preface
    "frontm_split_009.html": "names",  # A Note on Names and Places
}

CHAPTER_RE = re.compile(r"ch(\d{2})([a-z]*)\.html")

SENTENCE_GROUP_POLICY = "paragraph-sentences-clauses"
SENTENCE_GROUP_VERSION = 2
SENTENCE_GROUP_LIMITS = {"target_words": 70, "max_words": 85}
BLOCK_TAGS = {"p", "div", "li", "td", "th", "blockquote", "dd", "dt"}
SKIP_TAGS = {"script", "style", "sup"}  # sup: footnote markers only (verified all-numeric)
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
     "param", "source", "track", "wbr"}
)




# ---------------------------------------------------------------------------
# Spine


def _find_opf(z: zipfile.ZipFile) -> str:
    container = z.read("META-INF/container.xml").decode("utf-8")
    root = ElementTree.fromstring(container)
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    return root.find(".//c:rootfile", ns).get("full-path")


def _classify(href: str):
    """Return canonical chapter id for a spine href, or None if excluded."""
    base = posixpath.basename(href)
    if base in SELECTION:
        return SELECTION[base]
    m = CHAPTER_RE.fullmatch(base)
    if m and 1 <= int(m.group(1)) <= 9:
        return f"ch{m.group(1)}"
    return None  # titlepage, cover, frontm 000-007, notes, fur, index, ...


def read_spine(epub_path):
    """Return [(chapter_id, zip_member)] in spine order for selected chapters.

    Raises ValueError with a clear message when the EPUB is not the frozen
    book (missing container/OPF, unselected chapter file absent, or a
    chapter present in the spine that we did not select).
    """
    with zipfile.ZipFile(epub_path) as z:
        try:
            opf = _find_opf(z)
            root = ElementTree.fromstring(z.read(opf))
        except KeyError as exc:
            raise ValueError(f"{epub_path.name}: EPUB container/OPF missing ({exc})") from exc
        ns = {"opf": "http://www.idpf.org/2007/opf"}
        manifest = {item.get("id"): item.get("href") for item in root.findall("opf:manifest/opf:item", ns)}
        spine_refs = [r.get("idref") for r in root.findall("opf:spine/opf:itemref", ns)]
        opf_dir = posixpath.dirname(opf)
        members = set(z.namelist())
    selected = []
    seen = set()
    for idref in spine_refs:
        href = manifest.get(idref)
        if not href:
            continue
        member = posixpath.join(opf_dir, href) if not href.startswith("/") else href[1:]
        chapter = _classify(href)
        if chapter is None:
            continue
        if member not in members:
            raise ValueError(f"spine file missing from archive: {member}")
        selected.append((chapter, member))
        seen.add(chapter)
    present = {c for c, _ in selected}
    for c in ("preface", "names") + tuple(f"ch{i:02d}" for i in range(1, 10)):
        if c not in present:
            raise ValueError(
                f"book is not the frozen After Tamerlane edition: chapter {c!r} "
                "missing from spine (or not the expected calibre split)"
            )
    return selected


# ---------------------------------------------------------------------------
# HTML -> paragraphs


class _TextExtractor(HTMLParser):
    """Collects block paragraphs from one chapter file, dropping footnote
    sup markers, page-number references, headings (captured separately),
    and any script/style content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._stack = []  # ("skip"|"heading", tag)
        self._para = []
        self._paras = []
        self._heading = False
        self._heading_buf = []
        self._first_heading = None

    def _flush_para(self):
        txt = re.sub(r"\s+", " ", "".join(self._para)).strip()
        self._para = []
        if txt:
            self._paras.append(txt)

    def _skipping(self) -> bool:
        return any(entry is not None and entry[0] == "skip" for entry in self._stack)

    def _finish_heading(self):
        txt = re.sub(r"\s+", " ", "".join(self._heading_buf)).strip()
        self._heading_buf = []
        self._heading = False
        if txt and self._first_heading is None:
            self._first_heading = txt

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in SKIP_TAGS:
            self._stack.append(("skip", tag))
        elif tag == "a":
            href = attrs.get("href", "")
            if href.startswith("notes.html#") or "#page_" in href:
                self._stack.append(("skip", tag))
            else:
                self._stack.append(None)
        elif tag in HEADING_TAGS:
            self._flush_para()
            self._stack.append(("heading", tag))
            self._heading = True
        elif tag in BLOCK_TAGS:
            self._flush_para()
            self._stack.append(None)
        elif tag == "br":
            if self._heading:
                self._heading_buf.append(" ")
            elif not self._skipping():
                self._para.append(" ")
        elif tag not in VOID_TAGS:
            # Inline/other tags (em, small, span, ...) track depth for nesting.
            # Void/self-closing tags are transparent: they never enter the stack.
            self._stack.append(None)

    def handle_startendtag(self, tag, attrs):
        # Self-closing <br/>, <img/>, <link/>, ... must not emit a matching
        # end tag — the default swaps to handle_endtag, which would pop the
        # enclosing heading (or paragraph) off the stack and leave it dangling.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if not self._stack or tag in VOID_TAGS:
            return  # no dangle pop for <br/>-style end tags
        entry = self._stack.pop()
        if entry == ("skip", tag):
            pass
        elif entry == ("heading", tag):
            self._finish_heading()
        elif entry is None and tag in BLOCK_TAGS:
            self._flush_para()
        # mismatched nesting: tolerate (calibre output is well-formed)

    def handle_data(self, data):
        if self._skipping():
            return
        if self._heading:
            self._heading_buf.append(data)
        else:
            self._para.append(data)

    def close(self):
        super().close()
        self._flush_para()
        if self._heading:
            self._finish_heading()


def _file_paragraphs(z: zipfile.ZipFile, member: str):
    parser = _TextExtractor()
    parser.feed(z.read(member).decode("utf-8"))
    parser.close()
    return parser._paras, parser._first_heading


# ---------------------------------------------------------------------------
# Chapter assembly (merge calibre continuation splits)


def _continues(prev: str, nxt: str) -> bool:
    """True when calibre cut a paragraph mid-sentence between two files:
    the previous fragment lacks sentence-final punctuation and the next
    fragment starts lowercase (verified across every split chapter)."""
    return not re.search(r"[.!?][\"'\u2019\u201d)\]]*$", prev) and nxt[:1].islower()


def _merge_files(file_paras):
    merged = []
    for paras, _ in file_paras:
        if not paras:
            continue
        if merged and _continues(merged[-1], paras[0]):
            merged[-1] += " " + paras[0]
            merged.extend(paras[1:])
        else:
            merged.extend(paras)
    return merged


class _Chapters(dict):
    """Ordered {chapter_id: paragraphs} with .attrs metadata (titles)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = {}


def chapter_paragraphs(epub_path):
    """{chapter_id: [normalized paragraphs]} for the selected chapters."""
    spine = read_spine(epub_path)
    by_chapter = {}
    order = []
    for chapter, member in spine:
        if chapter not in by_chapter:
            by_chapter[chapter] = []
            order.append(chapter)
        by_chapter[chapter].append(member)
    with zipfile.ZipFile(epub_path) as z:
        titles = {}
        out = _Chapters()
        for chapter in order:
            members = by_chapter[chapter]
            file_paras = [_file_paragraphs(z, m) for m in members]
            first = next((h for paras, h in file_paras if paras), None)
            if first:
                titles[chapter] = first
            out[chapter] = _merge_files(file_paras)
    out.attrs["titles"] = titles
    return out


# ---------------------------------------------------------------------------
# Typography + number normalization


_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_NUM_RE = re.compile(r"\d[\d,]*(?:[–-]\d[\d,]*)?(?:\.\d+)?")


def _small_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("-" + _ONES[o] if o else "")


def int_to_words(n: int) -> str:
    """Ordinary integer reading (British 'and'), n >= 0."""
    if n == 0:
        return "zero"

    def under_1000(n):
        if n == 0:
            return ""
        h, r = divmod(n, 100)
        parts = [_ONES[h] + " hundred"] if h else []
        if r:
            if h:
                parts.append("and")
            parts.append(_small_words(r))
        return " ".join(parts)

    if n < 1000:
        return under_1000(n)
    for unit, scale in (("billion", 10**9), ("million", 10**6), ("thousand", 10**3)):
        if n >= scale:
            q, r = divmod(n, scale)
            head = f"{under_1000(q)} {unit}"
            if not r:
                return head
            tail = under_1000(r)
            if r < 100:
                return f"{head} and {tail}"
            return f"{head} {tail}"
    raise AssertionError("unreachable")


def year_words(n: int) -> str:
    """Year reading for 1000-2099; plain reading otherwise."""
    if n == 1000:
        return "one thousand"
    if 1000 < n < 1100:
        lo = n - 1000
        return f"ten oh {_small_words(lo)}" if lo < 10 else f"ten {_small_words(lo)}"
    if 1000 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return f"{_small_words(hi)} hundred"
        if lo < 10:
            return f"{_small_words(hi)} oh {_small_words(lo)}"
        return f"{_small_words(hi)} {_small_words(lo)}"
    return int_to_words(n)


def _pluralize_last(words: str) -> str:
    parts = words.split()
    last = parts[-1]
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return " ".join(parts[:-1] + [last])


def _expand_token(tok: str) -> str:
    if "–" in tok or "-" in tok:
        a_s, b_s = re.split(r"[–-]", tok, maxsplit=1)
        a, b = int(a_s.replace(",", "")), int(b_s.replace(",", ""))
        if len(b_s) < len(a_s):  # elided range end: "1945-89" -> 1989
            b = (a // (10 ** len(b_s))) * (10 ** len(b_s)) + b
        return f"{_year_or_int(a)} to {_year_or_int(b)}"
    if "." in tok:
        whole, frac = tok.split(".", 1)
        frac_words = " ".join(_ONES[int(d)] for d in frac)
        return f"{int_to_words(int(whole.replace(',', '')))} point {frac_words}"
    return _year_or_int(int(tok.replace(",", "")))


def _year_or_int(n: int) -> str:
    return year_words(n) if 1000 <= n <= 2099 else int_to_words(n)


def expand_numbers(text: str) -> str:
    """Expand digit runs to words per rule N5 (decades handled via trailing 's')."""
    out = []
    pos = 0
    for m in _NUM_RE.finditer(text):
        out.append(text[pos:m.start()])
        tok = m.group(0)
        pos = m.end()
        decade = (
            pos < len(text) and text[pos] == "s"
            and (pos + 1 >= len(text) or not text[pos + 1].isalpha())
            and "-" not in tok and "–" not in tok and "." not in tok
        )
        if decade:
            # "1330s" -> pluralized year reading only ("thirteen thirties")
            out.append(_pluralize_last(_year_or_int(int(tok.replace(",", "")))))
            pos += 1
        else:
            out.append(_expand_token(tok))
    out.append(text[pos:])
    return "".join(out)

def normalize_typography(text: str) -> str:
    """Rules N1-N4."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+[–—-]{1,2}\s+", ", ", text)
    text = re.sub(r"[\u201c\u201d\"]", "", text)
    text = re.sub(r"[\u2018\u2019']", lambda m: "'" if _between_words(text, m.start()) else "", text)
    return re.sub(r"\s+", " ", text).strip()


def _between_words(text: str, i: int) -> bool:
    return i > 0 and i + 1 < len(text) and text[i - 1].isalpha() and text[i + 1].isalpha()


def normalize_paragraph(text: str) -> str:
    text = expand_numbers(normalize_typography(text))
    replacements = {
        "sinkiang": "Sinkiang", "new'Turkish": "new Turkish", "Ch'ing": "Qing",
        "Tz'u-hsi": "Tzu Hsi", "tz'u-hsi": "Tzu Hsi", "haj": "Hajj",
        "new'local": "new local", "Moscownowaimed": "Moscow now aimed",
        "Weltpolitik": "world politics", "Bagdadbahn": "Baghdad Bahn", "anti-Ching": "anti Qing",
        "eastasia": "East Asia", "thepost-war": "the post-war",
        "EastAsia": "East Asia", "toamass": "to a mass",
        "thepost": "the post", "tonineteen": "to nineteen",
        "nowlay": "now lay", "zeromillion": "zero million",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


_ABBREVIATIONS = {
    "a.m.", "c.", "cf.", "dr.", "e.g.", "etc.", "fig.", "i.e.", "mr.",
    "mrs.", "ms.", "no.", "nos.", "p.", "p.m.", "pp.", "prof.",
    "st.", "vol.", "vs.",
}

def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def clause_boundaries(text: str, sentence_start: int = 0) -> list[dict]:
    """Return eligible punctuation cuts in one exact sentence slice.

    Offsets are relative to ``text`` and ``source_offset`` carries the
    corresponding offset when callers pass a larger source string.
    """
    candidates = []
    for match in re.finditer(r"[:;—]|\),", text):
        cut = match.end()
        left_words = _word_count(text[:cut])
        right_words = _word_count(text[cut:])
        if left_words < 8 or right_words < 8:
            continue
        punctuation = match.group()
        candidates.append({
            "offset": cut,
            "source_offset": sentence_start + cut,
            "punctuation": punctuation,
            "kind": "parenthetical-comma" if punctuation == ")," else punctuation,
            "left_words": left_words,
            "right_words": right_words,
        })
    return candidates


def clause_spans(text: str, sentence_start: int = 0) -> list[dict]:
    """Split one sentence at eligible boundaries, preserving exact slices."""
    boundaries = clause_boundaries(text, sentence_start)
    spans = []
    start = 0
    for index, boundary in enumerate(boundaries):
        cut = boundary["offset"]
        # Keep inter-clause whitespace in the preceding slice so slices are
        # contiguous and can reconstruct the original source with join("").
        end = cut
        while end < len(text) and text[end].isspace():
            end += 1
        spans.append({
            "start": sentence_start + start,
            "end": sentence_start + end,
            "text": text[start:end],
            "words": _word_count(text[start:end]),
            "sentence_start": sentence_start,
            "clause_index": index,
            "boundary_after": dict(boundary),
        })
        start = end
    spans.append({
        "start": sentence_start + start,
        "end": sentence_start + len(text),
        "text": text[start:],
        "words": _word_count(text[start:]),
        "sentence_start": sentence_start,
        "clause_index": len(boundaries),
        "boundary_after": None,
    })
    return spans


def sentence_spans(text: str) -> list[dict]:
    """Return conservative, exact sentence slices of normalized prose."""
    spans = []
    start = 0
    for match in re.finditer(r"[.!?]+[)\]]?(?=\s|$)", text):
        end = match.end()
        token = text[start:match.start() + 1].rsplit(" ", 1)[-1]
        if (match.group()[0] == "." and
                (token.lower() in _ABBREVIATIONS or
                 re.fullmatch(r"(?:[A-Z]\.)+", token))):
            continue
        next_char = text[end:].lstrip()[:1]
        if next_char and next_char.islower():
            continue
        while start < end and text[start].isspace():
            start += 1
        sentence = text[start:end]
        spans.append({"start": start, "end": end, "text": sentence,
                      "words": _word_count(sentence),
                      "clause_boundaries": clause_boundaries(sentence, start),
                      "clause_spans": clause_spans(sentence, start)})
        start = end
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        sentence = text[start:]
        spans.append({"start": start, "end": len(text), "text": sentence,
                      "words": _word_count(sentence),
                      "clause_boundaries": clause_boundaries(sentence, start),
                      "clause_spans": clause_spans(sentence, start)})
    return spans or ([{"start": 0, "end": len(text), "text": text,
                       "words": _word_count(text),
                       "clause_boundaries": clause_boundaries(text),
                       "clause_spans": clause_spans(text)}] if text else [])


def split_sentences(text: str) -> list[str]:
    return [span["text"] for span in sentence_spans(text)]


def group_sentences(text: str, spans=None, target_words: int = 70,
                    max_words: int = 85) -> list[dict]:
    """Pack sentence and eligible clause slices toward target, under max."""
    spans = sentence_spans(text) if spans is None else spans
    units = []
    for sentence_index, sentence in enumerate(spans):
        for clause_index, clause in enumerate(sentence.get("clause_spans") or [sentence]):
            unit = dict(clause)
            unit["sentence_index"] = sentence_index
            unit["clause_index"] = clause_index
            units.append(unit)
    groups = []
    start = 0
    while start < len(units):
        end = start
        words = 0
        while end < len(units):
            unit_words = units[end]["words"]
            if end > start and (words >= target_words or words + unit_words > max_words):
                break
            words += unit_words
            end += 1
        selected = units[start:end]
        a, b = selected[0]["start"], selected[-1]["end"]
        sentence_indexes = sorted({u["sentence_index"] for u in selected})
        clause_indexes = [[u["sentence_index"], u["clause_index"]] for u in selected]
        chunk_text = text[a:b]
        groups.append({
            "text": chunk_text,
            "text_sha256": hashlib.sha256(chunk_text.encode()).hexdigest(),
            "start": a,
            "end": b,
            "sentence_span": [sentence_indexes[0], sentence_indexes[-1] + 1],
            "sentence_indexes": sentence_indexes,
            "sentence_count": len(sentence_indexes),
            "clause_indexes": clause_indexes,
            "clause_count": len(selected),
            "clause_spans": selected,
            "words": words,
        })
        start = end
    return groups


def sentence_chunks(text: str, max_words: int = 85) -> list[dict]:
    groups = group_sentences(text, max_words=max_words)
    return [{"text": g["text"], "sentence_start": g["sentence_span"][0],
             "sentence_end": g["sentence_span"][1],
             "sentence_count": g["sentence_count"], "words": g["words"],
             "clause_indexes": g["clause_indexes"],
             "clause_count": g["clause_count"],
             "clause_spans": g["clause_spans"]}
            for g in groups]



def extract_chapters(epub_path):
    """[{'id', 'title', 'paragraphs': [...]}] in spine order."""
    paras_by_chapter = chapter_paragraphs(epub_path)
    titles = paras_by_chapter.attrs["titles"]  # type: ignore[attr-defined]
    chapters = []
    for chapter_id, paras in paras_by_chapter.items():
        paragraphs = [text for p in paras if (text := normalize_paragraph(p))]
        chapters.append(
            {"id": chapter_id, "title": titles.get(chapter_id, chapter_id),
             "paragraphs": paragraphs}
        )
    return chapters


def _frozen_book_checks(chapters):
    """Extraction assertions against the frozen After Tamerlane edition.

    Guards the void/startend-tag regression end to end: headings must hold
    exactly the heading text (no body leaking into titles), and each chapter
    must keep its opening prose — ch02's heading is the prime case (two
    <br/> void tags inside <h2>). Returns True when all hold.
    """
    by_id = {c["id"]: c for c in chapters}
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok ' if cond else 'FAIL'} {name}")

    check("title is heading text only (ch02)", by_id["ch02"]["title"] == "2 Eurasia and the Age of Discovery")
    check("title is heading text only (ch01)", by_id["ch01"]["title"] == "1 Orientations")
    check("title is heading text only (ch09)", by_id["ch09"]["title"] == "9 Tamerlane’s Shadow")
    # ch02's first prose paragraph sits right after the <h2> + <img/>; it used
    # to be swallowed into the title when the heading stayed open.
    ch02_text = " ".join(by_id["ch02"]["paragraphs"])
    check("ch02 opening prose retained", "In retrospect, we can see" in ch02_text)
    pref_text = " ".join(by_id["preface"]["paragraphs"])
    check("preface opening prose retained", bool(pref_text.strip()))
    return ok


# ---------------------------------------------------------------------------
# Self-check + CLI (pure stdlib; no model, no EPUB required for selfcheck)


def selfcheck() -> int:
    """Unit-level checks of selection and normalization rules.

    Pure stdlib; never touches the book or any model.
    """
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")

    print("audiobook.epub selfcheck")
    check("classify preface/names", _classify("frontm_split_008.html") == "preface"
          and _classify("frontm_split_009.html") == "names")
    check("classify chapters + continuations", _classify("ch01.html") == "ch01"
          and _classify("ch01a.html") == "ch01" and _classify("ch09.html") == "ch09")
    check("classify excludes others", _classify("ch10.html") is None
          and _classify("notes.html") is None and _classify("frontm_split_000.html") is None)

    check("N1 double quotes removed", normalize_typography('He said "hi".') == "He said hi.")
    check("N2 apostrophe kept between words", normalize_typography("Tamerlane's") == "Tamerlane's")
    check("N2 quote-apostrophe removed", normalize_typography("'global'") == "global")
    check("N3 spaced dash to comma", normalize_typography("a - b") == "a, b"
          and normalize_typography("a -- b") == "a, b")
    check("N4 whitespace collapse", normalize_typography("a   b\n\tc") == "a b c")

    check("N5 year 1405", expand_numbers("1405") == "fourteen oh five")
    check("N5 year 1970", expand_numbers("1970") == "nineteen seventy")
    check("N5 century 1900", expand_numbers("1900") == "nineteen hundred")
    check("N5 year 1005", expand_numbers("1005") == "ten oh five")
    check("N5 full range", expand_numbers("1332-1406") == "thirteen thirty-two to fourteen oh six")
    check("N5 elided range", expand_numbers("1945-89") == "nineteen forty-five to nineteen eighty-nine")
    check("N5 decade", expand_numbers("1330s") == "thirteen thirties")
    check("N5 3-digit decade", expand_numbers("400s") == "four hundreds")
    check("N5 decimal", expand_numbers("1.3") == "one point three")
    check("N5 large integer", expand_numbers("600,000") == "six hundred thousand")
    check("int_to_words 1001", int_to_words(1001) == "one thousand and one")
    check("sentence split preserves abbreviations",
          split_sentences("Dr. Smith left. He returned at 3.5 p.m. quietly.") ==
          ["Dr. Smith left.", "He returned at 3.5 p.m. quietly."])
    grouped = sentence_chunks("One two. Three four five. Six seven.", max_words=5)
    check("sentence chunks bounded and exact",
          [c["text"] for c in grouped] == ["One two. Three four five.", "Six seven."]
          and " ".join(c["text"] for c in grouped) ==
          "One two. Three four five. Six seven.")
    oversized = sentence_chunks("One two three four five six.", max_words=3)
    check("oversized sentence stays intact", len(oversized) == 1
          and oversized[0]["words"] == 6)


    check("merge continuation", _continues("he said", "quietly") is True
          and _continues("He stopped.", "Then he ran") is False)

    # A heading containing a self-closing <br/> must stay one heading: the V
    # void end tag used to pop the heading off the stack, so </h2> couldn't
    # close it and the following body leaked into the title.
    _br = _TextExtractor()
    _br.feed("<h2>Title<br/>Line two</h2><p>Opening prose.</p>")
    _br.close()
    check("heading intact across <br/>",
          _br._first_heading == "Title Line two" and _br._paras == ["Opening prose."])
    long_tail = "one two three four five six seven eight"
    for mark, label in ((":", "colon"), (";", "semicolon"), ("—", "em dash")):
        sample = long_tail + mark + " " + long_tail + "."
        boundaries = clause_boundaries(sample)
        check(label + " clause split", len(boundaries) == 1 and boundaries[0]["punctuation"] == mark)
        check(label + " exact reconstruction", "".join(s["text"] for s in clause_spans(sample)) == sample)
    paren = long_tail + " (nine ten eleven twelve thirteen fourteen fifteen sixteen), " + long_tail + "."
    check("parenthetical comma split", any(b["kind"] == "parenthetical-comma" for b in clause_boundaries(paren)))
    short = "one two three : four five six seven eight nine ten eleven twelve."
    check("clause rejects short side", clause_boundaries(short) == [])
    comma = long_tail + ", " + long_tail + "."
    check("ordinary comma ignored", clause_boundaries(comma) == [])
    _bra = _TextExtractor()
    _bra.feed("<h2>A<br/>B</h2><p class=\"image\"><img src=\"x\"/></p><p>Body one.</p>")
    _bra.close()
    check("self-closing img doesn't drop paragraph",
          _bra._first_heading == "A B" and _bra._paras == ["Body one."])
    failed = [name for name, ok in results if not ok]
    print(f"selfcheck: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


def _default_book() -> pathlib.Path:
    """Configured book path from audiobook.toml found at or above the cwd."""
    from . import config

    root = config.find_root()
    return root / config.load_config(root).book


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="audiobook.epub",
        description="Pure-stdlib EPUB extraction self-check and dry run (no audio, no model).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck", help="run the normalization/splitting unit self-check")
    p = sub.add_parser("extract", help="dry-run extraction: print chapter/paragraph counts")
    p.add_argument("--book", help="EPUB path (default: <repo>/books/after-tamerlane.epub)")
    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return selfcheck()

    try:
        chapters = extract_chapters(pathlib.Path(args.book) if args.book else _default_book())
    except (ValueError, FileNotFoundError) as e:
        print(f"extract: {e}", file=sys.stderr)
        return 1
    for c in chapters:
        print(f"  {c['id']:<8} {len(c['paragraphs']):>5} paragraphs  {c['title'][:60]}")
    total = sum(len(c["paragraphs"]) for c in chapters)
    print(f"chapters={len(chapters)} paragraphs={total}")
    if not _frozen_book_checks(chapters):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
