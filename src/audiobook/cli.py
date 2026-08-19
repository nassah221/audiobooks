"""CLI for the audiobook pipeline: preflight / generate / validate / eta.

Entry point ``audiobook.cli:main`` (pyproject [project.scripts]).
Thin dispatch over runner.py; all behavior lives there. Clear errors,
project-relative default paths (--root auto-discovered, --out defaults to
<root>/outputs/qwen-book).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from . import runner
from .runner import RunError

_HELP = "Frozen Qwen3-TTS + Bragg-reference audiobook generation (Apple Silicon MLX), using paragraph-bounded sentence chunks."


def _add_common(ap: argparse.ArgumentParser):
    ap.add_argument("--root", help="project root (default: auto-discovered upward via audiobook.toml)")
    ap.add_argument("--out", help="output dir (default: <root>/outputs/qwen-book)")


def _resolve(args) -> tuple:
    root = pathlib.Path(args.root) if args.root else runner.find_root()
    out = pathlib.Path(args.out) if args.out else root / "outputs" / "qwen-book"
    return root, out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audiobook", description=_HELP)
    _add_common(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight", help="inputs + hashes, model cache, env, extraction, measured pilot")
    p.add_argument("--sentence", default=runner.PILOT_SENTENCE,
                   help="measured preflight pilot text (default: the frozen pilot sentence)")
    p.add_argument("--no-benchmark", action="store_true",
                   help="checks + extraction only; no model load")
    p.add_argument("--offline", action="store_true", help="forbid downloads (fail if cache incomplete)")
    p.add_argument("--validate/--no-validate", dest="validate", action=argparse.BooleanOptionalAction,
                   default=True, help="ASR-validate the measured pilot (default: on)")

    p = sub.add_parser("generate", help="resumable generation (sentence-bounded PCM16 checkpoints)")
    p.add_argument("--chapters", help="comma list with ranges, e.g. 1-9,preface,names (default: all)")
    p.add_argument("--limit", type=int, help="max sentence chunks (smoke runs)")
    p.add_argument("--offline", action="store_true", help="forbid downloads (fail if cache incomplete)")
    p.add_argument("--force", action="store_true",
                   help="on a matching state: regenerate everything. On a state mismatch "
                        "(inputs/plan changed): archive the old state.json and carry forward "
                        "whatever done chunks still match this run, instead of erroring")
    p.add_argument("--discard-done", action="store_true",
                   help="on a state mismatch, explicitly wipe recorded done/failed progress "
                        "(the old state.json is archived first, never silently discarded)")
    p.add_argument("--resume-from", help="skip chapters before this one (e.g. ch03)")

    p = sub.add_parser("validate", help="persistent Whisper ASR validation of generated chunks")
    p.add_argument("--chapters", help="comma list with ranges, e.g. 1-9,preface,names (default: all)")
    p.add_argument("--limit", type=int, help="max chunks to validate")

    sub.add_parser("eta", help="projected generation + ASR cost from the measured preflight pilot")

    return ap


def _print_preflight(rpt: dict) -> None:
    print("audiobook preflight")
    print(f"  root: {rpt['root']}")
    inputs = rpt["inputs"]
    bad = [rel for rel, f in inputs.items() if not f["ok"]]
    print(f"  inputs: {len(inputs) - len(bad)}/{len(inputs)} sha256 match")
    cache = rpt["model_cache"]
    state = "complete" if cache["cache_complete"] else f"incomplete (missing {len(cache['missing'])} files)"
    print(f"  model cache: {state} ({cache['repo']}@{cache['revision']}, refs/main {'OK' if cache['refs_match'] else 'MISMATCH'})")
    pkgs = rpt["packages"]
    missing_pkgs = [n for n, v in pkgs.items() if v is None]
    print("  packages: " + ", ".join(f"{n} {v}" for n, v in pkgs.items() if v) +
          (f"  [missing: {', '.join(missing_pkgs)}]" if missing_pkgs else ""))
    ex = rpt["extraction"]
    counts = ", ".join(f"{c['id']}={c['paragraphs']}" for c in ex["chapters"])
    print(f"  extraction: {len(ex['chapters'])} chapters, {ex['total_paragraphs']} paragraphs ({counts})")
    pilot = rpt["benchmark"]
    if pilot is None:
        print("  measured pilot: skipped (--no-benchmark)")
        return
    m = pilot["metrics"]
    asr_line = ""
    if pilot["asr"]:
        a = pilot["asr"]
        asr_line = f" asr rtf {a['rtf']} ({a['verdict']})" if a["rtf"] else f" asr {a['verdict']}"
    print(f"  measured pilot: {pilot['verdict']} gen {m['generation_seconds']}s "
          f"load {pilot['load_seconds']}s audio {m.get('audio_seconds')}s{asr_line}")
    if pilot["verdict"] != "PASS":
        for e in pilot.get("errors", []):
            print(f"    - {e}")


def _print_validate(rpt: dict) -> None:
    print(f"audiobook validate: chunks={rpt['chunks']} passed={rpt['passed']} "
          f"failed={rpt['failed']} cached={rpt['cached']}")
    for f in rpt["failures"]:
        print(f"  FAIL {f['chunk_id']}: {f['reasons'][0]}")
    print(f"  asr: {rpt['asr']['model_repo']} rtf={rpt['asr']['rtf']} "
          f"asr_s={rpt['asr']['asr_seconds']:.2f} audio_s={rpt['asr']['audio_seconds']:.2f}")
    print(f"  records: {rpt['records']}")
    if rpt.get("book"):
        b = rpt["book"]
        print(f"  book: assembled {b['wav']} ({b['seconds']}s audio, {b['chapters']} chapters)")
    elif rpt.get("failed_chunks"):
        print(f"  book: NOT assembled -- {len(rpt['failed_chunks'])} chunk(s) in the "
              f"continue-on-failure set: {rpt['book_blocked_reason']}")
        for cid in rpt["failed_chunks"]:
            print(f"    {cid}")


def _print_eta(rpt: dict) -> None:
    print("audiobook eta")
    print(f"  plan: {rpt['chapters']} chapters, {rpt['paragraphs']} paragraphs, {rpt['words']} words")
    print(f"  estimated audio: {rpt['estimated_audio_seconds'] / 60:.1f} min")
    print(f"  generation:     {rpt['generation_seconds'] / 60:.1f} min "
          f"(model load {rpt['model_load_seconds']}s once)")
    if rpt["asr_seconds"] is not None:
        print(f"  asr:            {rpt['asr_seconds'] / 60:.1f} min")
    print(f"  total wall:     {rpt['total_wall_seconds'] / 60:.1f} min")
    print(f"  basis: {rpt['note']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root, out = _resolve(args)
        if args.cmd == "preflight":
            rpt = runner.preflight(
                root, out, sentence=args.sentence,
                benchmark=not args.no_benchmark,
                offline=True if args.offline else None,
                validate=args.validate,
            )
            _print_preflight(rpt)
            return 0 if rpt["verdict"] == "PASS" else 1
        elif args.cmd == "generate":
            summary = runner.Generator(
                root, out, chapters=args.chapters, limit=args.limit,
                force=args.force, discard_done=args.discard_done,
                resume_from=args.resume_from,
                offline=True if args.offline else None,
            ).run()
            print(f"generated {summary['generated']} chunk(s); {summary['done_total']}/"
                  f"{summary['plan']['groups']} done "
                  f"(gen {summary['generation_seconds']}s, audio {summary['audio_seconds']}s)")
            if summary.get("failed_total"):
                print(f"  {summary['failed_total']} chunk(s) FAILED (deferred, continue-on-failure):")
                for f in summary["failed"]:
                    print(f"    {f['chunk_id']}: {f['reason']}")
        elif args.cmd == "validate":
            _print_validate(runner.validate_generated(
                root, out, chapters=args.chapters, limit=args.limit))
        elif args.cmd == "eta":
            _print_eta(runner.estimate(root, out))
        return 0
    except RunError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; completed chunks are checkpointed — re-run to resume",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
