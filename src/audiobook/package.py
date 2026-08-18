"""Package the adjudicated audiobook into a player-friendly M4B."""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import subprocess
import sys
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree

from . import runner, adjudicate
from .config import load_config

TITLE = "After Tamerlane"
INCLUDED = ("PASS", "ALLOW_LEXICAL", "ALLOW_LEXICAL_NAME")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _ffmeta(value):
    """Escape one value for an ffmetadata file."""
    return (str(value).replace("\\", "\\\\").replace("\n", "\\n")
            .replace("=", "\\=").replace(";", "\\;").replace("#", "\\#"))


def _epub_metadata(epub_path):
    """Read Dublin Core fields and the package cover without guessing values."""
    dc = "http://purl.org/dc/elements/1.1/"
    opf_ns = "http://www.idpf.org/2007/opf"
    with zipfile.ZipFile(epub_path) as z:
        container = ElementTree.fromstring(z.read("META-INF/container.xml"))
        cns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        opf = container.find(".//c:rootfile", cns).get("full-path")
        root = ElementTree.fromstring(z.read(opf))
        ns = {"dc": dc, "opf": opf_ns}
        metadata = root.find("opf:metadata", ns)
        values = {}
        for name in ("title", "creator", "publisher", "date", "language", "description"):
            node = metadata.find(f"dc:{name}", ns) if metadata is not None else None
            if node is not None and node.text:
                values[name] = node.text.strip()
        manifest = {item.get("id"): item for item in root.findall("opf:manifest/opf:item", ns)}
        cover_id = next((m.get("content") for m in metadata.findall("opf:meta", ns)
                         if m.get("name") == "cover"), None) if metadata is not None else None
        cover_item = manifest.get(cover_id)
        cover_member = None
        cover_type = None
        if cover_item is not None:
            cover_member = str(pathlib.PurePosixPath(opf).parent / cover_item.get("href"))
            cover_type = cover_item.get("media-type")
        description = values.get("description", "")
        parser = _TextExtractor()
        parser.feed(html.unescape(description))
        values["description"] = " ".join(" ".join(parser.parts).split())
        cover = z.read(cover_member) if cover_member in z.namelist() else None
    return values, cover, cover_type


def build(root, out_dir, with_chapters_wav):
    root = pathlib.Path(root)
    out = pathlib.Path(out_dir)
    cfg = load_config(root)
    epub_path = root / cfg.book
    epub_meta, cover_data, cover_type = _epub_metadata(epub_path)
    adj = adjudicate.Adjudicator(root, out)
    adj.classify()
    blocked = [d for d in adj.decisions if d[3].startswith("BLOCK")]
    if blocked:
        raise RuntimeError("current adjudication decisions contain blocked paragraphs: "
                           + ", ".join(d[0]["id"] for d in blocked[:8]))
    assembled = adj.concatenate()
    included = [d for d in adj.decisions if d[3] in INCLUDED]

    by_chapter = {}
    for paragraph, wav, rec, dec, det in included:
        by_chapter.setdefault(paragraph["chapter"], []).append((paragraph, wav))

    chapters = []
    cur = 0
    titles = {c["id"]: c["title"] for c in adj.plan["chapters"]}
    for plan_ch in adj.plan["chapters"]:
        ch = plan_ch["id"]
        paragraphs = by_chapter.get(ch)
        if not paragraphs:
            continue
        import soundfile as sf
        samples = sum(int(sf.info(str(wav)).frames) for _, wav in paragraphs)
        start_ms = int(round(cur * 1000 / runner.SAMPLE_RATE))
        cur += samples
        end_ms = int(round(cur * 1000 / runner.SAMPLE_RATE))
        chapters.append({"id": ch, "title": titles[ch], "start_ms": start_ms,
                         "end_ms": end_ms, "seconds": (end_ms - start_ms) / 1000,
                         "paragraphs": len(paragraphs)})

    meta_values = {
        "title": epub_meta.get("title", TITLE),
        "artist": epub_meta.get("creator", ""),
        "author": epub_meta.get("creator", ""),
        "album": epub_meta.get("title", TITLE),
        "album_artist": epub_meta.get("creator", ""),
        "genre": "History",
        "date": epub_meta.get("date", ""),
        "language": epub_meta.get("language", ""),
        "publisher": epub_meta.get("publisher", ""),
        "description": epub_meta.get("description", ""),
        "comment": epub_meta.get("description", ""),
        "track": "1/1",
        "disc": "1/1",
    }
    meta = [";FFMETADATA1"] + [f"{k}={_ffmeta(v)}" for k, v in meta_values.items() if v]
    for c in chapters:
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={c['start_ms']}",
                 f"END={c['end_ms']}", f"title={_ffmeta(c['title'])}"]
    meta_path = out / "chapters.ffmetadata"
    meta_path.write_text("\n".join(meta) + "\n")
    print(f"chapter metadata -> {meta_path.name} ({len(chapters)} chapters)")

    if with_chapters_wav:
        ch_dir = out / "chapters"
        ch_dir.mkdir(exist_ok=True)
        for c in chapters:
            paragraphs = by_chapter[c["id"]]
            total = sum(int(__import__("soundfile").info(str(wav)).frames)
                        for _, wav in paragraphs)
            dst = ch_dir / f"{c['id']}.wav"
            tmp = dst.with_name(dst.name + ".tmp")
            with open(tmp, "wb") as fh:
                fh.write(runner.Generator._wav_header_bytes(total))
                for _, wav in paragraphs:
                    with open(wav, "rb") as src:
                        src.seek(44)
                        for data in iter(lambda: src.read(1 << 20), b""):
                            fh.write(data)
            tmp.replace(dst)
            print(f"  chapter {c['id']} -> {dst.relative_to(root)} ({c['paragraphs']} paragraphs, {c['seconds']:.0f}s)")

    book_wav = out / adjudicate.BOOK_OVERRIDE_REL
    m4b = out / f"{TITLE.replace(' ', '')}.m4b"
    cover_path = None
    if cover_data and cover_type in {"image/jpeg", "image/png"}:
        suffix = ".jpg" if cover_type == "image/jpeg" else ".png"
        cover_path = out / (".cover" + suffix)
        cover_path.write_bytes(cover_data)
    cmd = ["ffmpeg", "-y", "-i", str(book_wav), "-i", str(meta_path)]
    if cover_path:
        cmd += ["-i", str(cover_path)]
    cmd += ["-map_metadata", "1", "-map", "0:a:0", "-metadata", f"artist={epub_meta.get('creator', '')}",
            "-metadata", f"publisher={epub_meta.get('publisher', '')}",
            "-metadata:s:a:0", "language=eng"]
    if cover_path:
        cmd += ["-map", "2:v:0", "-c:v", "copy", "-disposition:v:0", "attached_pic",
                "-metadata:s:v:0", "title=Cover", "-metadata:s:v:0", "comment=Cover"]
    cmd += ["-c:a", "aac", "-b:a", "64k", "-ar", "24000", "-ac", "1",
            "-movflags", "+faststart", "-f", "ipod", str(m4b)]
    print(f"transcoding -> {m4b.name} (AAC 64k mono, {assembled['seconds'] / 3600:.2f}h)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if cover_path:
        cover_path.unlink(missing_ok=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("ffmpeg failed")
    print("done:", m4b.name)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration:format=format_name", "-of", "json", str(m4b)],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        d = json.loads(probe.stdout)["format"]
        print(f"  m4b duration {float(d['duration']) / 3600:.2f} h, format {d.get('format_name')}")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="audiobook.package")
    ap.add_argument("--root", help="project root (default autodiscover)")
    ap.add_argument("--out", help="output dir (default outputs/qwen-book)")
    ap.add_argument("--chapters", action="store_true", help="also write per-chapter WAVs")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root) if args.root else runner.find_root()
    out = pathlib.Path(args.out) if args.out else root / "outputs" / "qwen-book"
    build(root, out, args.chapters)


if __name__ == "__main__":
    sys.exit(main())
