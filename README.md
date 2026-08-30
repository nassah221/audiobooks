# audiobook

This project turns an EPUB and a short voice reference into a validated, chaptered audiobook on Apple Silicon.

The current pipeline uses Qwen3-TTS for voice-cloned speech and Whisper for validation. It plans paragraph-bounded chunks, checkpoints every accepted WAV, resumes interrupted runs, records failures instead of hiding them, and refuses assembly until the selected audio is complete.

## Requirements

- Apple Silicon Mac running macOS 14 or later
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` for M4B packaging

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install ffmpeg
uv sync --locked
```

You do not need to activate the virtual environment. Run project commands through `uv run`.

## Configuration

`audiobook.toml` is the source of truth for a run. It contains:

- the EPUB path and SHA-256;
- the reference WAV, transcript, and hashes;
- the pinned Qwen model and revision;
- the pinned Whisper model and revision;
- the spoken language, token limit, and base seed.

The repository includes the configured reference:

- `references/bragg/arab-conquests-v3.wav`
- `references/bragg/arab-conquests-v3.txt`

To use a different book or voice, copy the files into the project and update the matching paths and hashes in `audiobook.toml`:

```sh
shasum -a 256 books/my-book.epub references/voice/reference.wav references/voice/reference.txt
```

Keep the config fixed after generation starts. Changing the book, reference, or pinned model changes run identity. Use a new output directory unless you deliberately intend to reconcile an existing run.

## Qualify a voice reference

A reference should contain one speaker, no music or overlap, an exact transcript, natural punctuation, a complete declarative ending, and clean margins.

Run the mastery check before using a new reference:

```sh
uv run python -m audiobook.mastery reference.wav reference.txt --json
```

It checks speech coverage, duration, boundaries, SNR, interior pauses, peak level, spectral balance, true LUFS, clipping, and pinned DNSMOS P.835 quality. Measurements help reject bad candidates; listening decides marginal perceptual cases.

## Preflight

```sh
uv run audiobook preflight
```

Preflight verifies input hashes, package versions, model cache, EPUB extraction, and a measured generation pilot. For a cache-only machine:

```sh
uv run audiobook preflight --offline
```

Use the measured result for an ETA:

```sh
uv run audiobook eta
```

## Generate

```sh
caffeinate -dimsu uv run audiobook generate
```

Generation is resumable. Each accepted chunk is written as an atomic PCM16 WAV checkpoint. Re-running the command continues from saved state.

Useful options:

```sh
uv run audiobook generate --chapters ch03 --offline
uv run audiobook generate --limit 12 --offline
uv run audiobook --out outputs/my-run generate --chapters ch03
```

- `--chapters` selects chapters or ranges.
- `--limit` creates a small smoke run.
- global `--out` selects the runtime directory.
- `--force` regenerates everything when state matches; when state has drifted, it archives the old state and carries forward completed chunks that still match.
- `--discard-done` explicitly discards recorded progress after archiving state.
- `--resume-from` skips earlier selected chapters.

The generation policy is `icl-nocontext-eos-replacement-v4`. Every chunk starts from the canonical reference. Rolling generated-audio context is disabled because it caused cumulative timbre drift.

At the first sampled end-of-speech token, the generator resamples that step with EOS disabled, keeps the one replacement codec frame, and stops. This supplies decoder right context while avoiding unnecessary continuation audio. The decoder uses exact codec-frame lengths to avoid the mlx-audio valid-length bug.

## Validate and resolve failures

Generation validates each take automatically. You can also validate an existing run:

```sh
uv run audiobook validate
uv run audiobook validate --chapters ch03
```

Validation checks expected word order, required content, terminal completion, repetition, punctuation pauses, signal structure, tail activity, and final sibilants. When retries are exhausted, the last generated WAV is kept if generation produced one; the failed-state entry preserves its reasons and hashes.

For a name-heavy book, Whisper may write correct speech using another spelling. Review those failures with the lexical adjudicator:

```sh
uv run python -m audiobook.adjudicate --dry-run
uv run python -m audiobook.adjudicate --assemble
```

Adjudication records lexical exceptions and preserves the original validator result. It does not permit missing content, clipped speech, bad tails, or structural failures.

To request new takes for selected completed chunks, put one chunk ID per line in a file:

```sh
uv run audiobook regenerate --chunks chunks-to-redo.txt
uv run audiobook generate --offline
```

The old WAV and validation record are archived rather than silently deleted.

## Assembly

Normal assembly uses accepted parent chunks or complete validated child sets. It preserves each chunk's PCM samples and inserts only enough silence to reach:

- 250 ms within a paragraph;
- 500 ms between paragraphs.

Native silence counts toward those limits. Production assembly does not use continuous room tone, crossfades, EQ, compression, denoising, tempo changes, or loudness normalization.

A generated run writes its state, chunks, validator cache, records, and assembled WAV under the chosen output directory. `outputs/` is local runtime storage and is ignored by Git.

## Package as M4B

```sh
uv run python -m audiobook.package
```

Packaging rebuilds the adjudicated WAV, reads metadata and cover art from the EPUB, and writes a chaptered AAC M4B. Add `--chapters` to also write per-chapter PCM WAVs.

## Current pilot

The audited Chapter 3 v4 pilot is stored locally at:

```text
outputs/pilots/ch03-v4/
```

It contains the assembled WAV, effective selection, override ledger, boundary audit, readiness record, and `promotion-manifest.json`. The manifest records permanent relative paths and SHA-256 hashes for all six promoted files.

The pilot covers 305 of 305 source groups, uses 381 audio parts, and audits all 380 joins. Twelve representative listening clips were marked complete, artifact-free, and well paced.

The pilot is intentionally not in Git because its WAV is about 337 MB. Preserve that directory separately when moving or backing up the project.

## Code map

```text
audiobook.toml
    ↓
config.py → epub.py → runner.py → qwenfix.py
                         ↓
                       asr.py
                         ↓
                    adjudicate.py
                         ↓
                     package.py
```

- `config.py` loads and verifies configured inputs.
- `epub.py` extracts chapters and builds paragraph-bounded plans.
- `runner.py` manages generation, validation, retries, state, and assembly.
- `qwenfix.py` owns Qwen sampling and exact codec-frame decoding.
- `asr.py` normalizes spoken text and validates generated takes.
- `mastery.py` qualifies voice references.
- `adjudicate.py` records reviewed lexical exceptions.
- `package.py` writes the chaptered M4B.

The design and its current limits are documented in [`docs/audiobook_pipeline_architecture.md`](docs/audiobook_pipeline_architecture.md).

## Commands

```sh
uv run audiobook preflight
uv run audiobook eta
uv run audiobook generate
uv run audiobook validate
uv run audiobook regenerate --chunks chunks-to-redo.txt
uv run python -m audiobook.mastery reference.wav reference.txt --json
uv run python -m audiobook.adjudicate --dry-run
uv run python -m audiobook.adjudicate --assemble
uv run python -m audiobook.package
```

Run any command with `--help` for its available options.
