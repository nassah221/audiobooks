# audiobook — make an audiobook from a book and a reference voice

This tool turns an EPUB text file and a short voice sample into a spoken
audiobook on an Apple Silicon Mac. It reads a book, speaks it one sentence at
a time in a voice cloned from a reference recording, checks each sentence
automatically, and resumes where it left off if you stop it. It does **not**
edit the audio after it is generated — no loudness tweaks, no mastering.

Everything about the book, the voice, and the model lives in one small file,
`audiobook.toml`. You edit that file, then run a few commands.

## What you need

- An **Apple Silicon Mac** (the M2 Pro is the target) on **macOS 14 (Sonoma)
  or later**. The frozen `mlx` package needs macOS 14 and has no fallback.
- `uv` (installs Python and the pinned packages for you):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- The book and any replacement reference audio. Keep copyrighted inputs out of
  git. The private repository includes only the configured 10-second canonical
  reference sample and its exact transcript.
- `ffmpeg` for the final M4B package:

```sh
brew install ffmpeg
```


## Setup

Run this once in the project folder:

```sh
uv sync --locked
```

This installs an exact, pinned set of packages into `.venv/`. You never need
to activate a virtual environment by hand — always run the tool through
`uv run`, which uses that environment automatically.

The first time you run it, the model is downloaded into the Hugging Face
cache (`~/.cache/huggingface` by default). Later runs work offline.

## Add your book and voice

The repository includes the configured canonical reference sample. To use a
different book or voice, put the replacement files in the project:

```sh
cp path/to/your-book.epub books/
cp /path/to/reference.wav references/voice/
cp /path/to/reference.txt references/voice/
```

The **reference** is a short recording of the voice you want (the WAV) plus
the exact text of what is said in it (the TXT). The tool clones that voice.
Chapter and book selection is still done on the command line (see Commands),
not in the config.

> The private repository includes the canonical sample and frozen example
> hashes. When you use your own files, replace those hashes in the next step,
> or `preflight` will refuse to start.

## Configure `audiobook.toml`

Open `audiobook.toml`. It has four short sections. All paths are relative to
the project root.

| Setting | What it is | Example |
|---|---|---|
| `[book] path` | EPUB file to read | `books/mine.epub` |
| `[book] sha256` | Checksum of that EPUB (keeps the book honest) | `6bd14…` |
| `[voice] audio` | Reference WAV (the voice to clone) | `references/voice/ref.wav` |
| `[voice] audio_sha256` | Checksum of the WAV | `049da…` |
| `[voice] transcript` | Exact text of what the WAV says | `references/voice/ref.txt` |
| `[voice] transcript_sha256` | Checksum of the transcript | `2c809…` |
| `[model] repo` | Which model to use (leave alone) | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| `[model] revision` | Exact model version (leave alone) | `a6eb4f…` |
| `[model] language` | Language to speak | `English` |
| `[model] max_tokens` | Max tokens per sentence | `4096` |
| `[asr] repo` | Model used to check sentences (leave alone) | `mlx-community/whisper-large-v3-turbo` |
| `[asr] revision` | Exact ASR model version (leave alone) | `a4aae…` |

To fill in the hashes:

```sh
shasum -a 256 books/mine.epub references/voice/ref.wav references/voice/ref.txt
```

Put the three long strings into the matching `sha256` fields above.

The only settings you normally change are the book and voice paths and hashes.
You can also change the spoken language or `max_tokens`. Do not change `repo`
or `revision`.

> **Warning:** once you start generating, keep the config still. Changing any
> book, voice, or model/ASR setting — or swapping in different voice files —
> changes the run fingerprint, so output already generated no longer matches,
> and you would need a new `--out` folder or `--force` to regenerate.

## Check everything (preflight)

```sh
uv run audiobook preflight
```

This checks, in order: that the files exist, that their hashes match, that the
model is cached, that the environment is right, and that the text extracts
cleanly. It also does a short **measured pilot** generation so later timing
estimates are real, not guesses.

If anything is wrong, it tells you exactly what (for example, a hash that does
not match, or macOS older than 14).

## Estimate the time (eta)

```sh
uv run audiobook eta
```

This reads the measured pilot and the number of sentences, then projects how
long generation and checking will take on the machine running it. Run
`preflight` on the M2 Pro first; its measured result is the only reliable ETA.

## Generate (and keep the Mac awake)

```sh
caffeinate -dimsu uv run audiobook generate
```

`caffeinate` stops your Mac from sleeping during a long run.

Generation is **checkpointed after every sentence**. If you stop it, or the
Mac sleeps, or it crashes, the next `generate` picks up exactly where it
stopped — you lose at most the one sentence it was working on. Chapters are
assembled only after validation (below), not as you go.

Useful flags:

- `--limit N` — generate only the first `N` sentences (a quick smoke test).
- `--force` — ignore the saved progress and regenerate everything.
- `--out DIR` — put output somewhere other than the default `outputs/qwen-book`.

## Validate each sentence

```sh
uv run audiobook validate
```

Every generated sentence is checked by automated speech recognition. The
strict validator keeps failed sentences for review and records why they
failed. It assembles `book.wav` only when every planned sentence passes.

For this name-heavy book, Whisper can spell correct speech differently. Apply
the recorded lexical policy after validation to distinguish those differences
from clipping, silence, repetition, and other audio defects:

```sh
uv run python -m audiobook.adjudicate --dry-run
uv run python -m audiobook.adjudicate --assemble
```

The first command reports every decision without writing files. The second
backs up failed sentence WAVs, appends every decision to the ledger, and writes
`book.lexical-advisory-v1.wav`. It refuses assembly while any structural or
low-coverage failure remains.

## Package the audiobook

```sh
uv run python -m audiobook.package
```

This rebuilds the adjudicated WAV from the current decisions, reads the title,
author, description, cover, and chapter names from the EPUB, then writes an
AAC M4B with embedded chapters and cover art. Add `--chapters` to also write
one PCM16 WAV per chapter.

## What you get

The commands write runtime state, generated audio, validation records, the
adjudication ledger, and the packaged audiobook under `outputs/qwen-book`.
That directory and all source books and audio are excluded from git.

## Copy to the M2 Pro

The target machine is the M2 Pro. To move a finished project there:

1. Copy the project folder (code, `audiobook.toml`, your book, reference WAV
   and TXT) to the M2 Pro. Do not copy git history if you keep this private.
2. Run `uv sync --locked` there. Keep the model cache, or let the first
   `preflight` re-download it (it works offline afterwards).
3. Run the commands above from that folder. The config and checkpoints travel
   with the project, so you can resume where you left off.

## How it fits together (for tweakers)

The pipeline modules under `src/audiobook/` form this flow:

```
audiobook.toml -> config.py -> epub.py -> runner.py -> asr.py
                                      generation    strict validation
                                                        |
                                                        v
                                               adjudicate.py -> package.py
                                               recorded policy    M4B
```

File map:

- `config.py` reads `audiobook.toml`, finds the project root, and holds the
  settings.
- `cli.py` provides the `preflight`, `eta`, `generate`, and `validate` commands.
- `epub.py` extracts chapter text, normalizes it, and splits it into sentences.
- `runner.py` generates sentence WAVs, saves atomic checkpoints, and records
  progress. It also runs the measured pilot and calculates the ETA.
- `asr.py` checks each sentence and writes strict PASS or FAIL records.
- `adjudicate.py` records lexical overrides while retaining structural gates.
- `package.py` writes the chaptered M4B from the current adjudication decisions.

The rules it keeps:

- **One model load** per run — loading is slow, so it loads once and reuses.
- **Sentence-atomic checkpoints** — each sentence is its own saved file, so a
  stop or crash loses at most the current sentence.
- **Config fingerprint** — every run records the book/voice/model/asr hashes.
  If any of them change (or the actual voice files change), old progress no
  longer matches.
- **Exact concatenation, no processing** — chapters are byte-joined from the
  saved sentence WAVs; nothing is edited, normalized, or mastered.

Where to change what:

- New book or voice / hashes → `audiobook.toml`.
- Text cleanup (numbers, punctuation, sentence splitting) → `epub.py`.
- Sound of generation / checkpoints → `runner.py`.
- How sentences are verified → `asr.py`.
- Lexical override policy and assembly → `adjudicate.py`.
- M4B metadata, cover, and chapter packaging → `package.py`.
- New or changed commands / flags → `cli.py`.

Keep those invariants in mind when you edit: changing inputs or generation
details must change the fingerprint, and never post-process the saved
sentence audio.

## Performance experiments (benchmark-only, nothing enabled)

`audiobook/perf.py` is an opt-in benchmark/profiling harness that compares the
frozen sequential generation against two *unadopted* speedups. **Neither is
enabled in production** — they only run here, in isolated fresh subprocesses,
writing under `outputs/perf` (gitignored):

- **speaker-cache** (default): caches the canonical reference speaker
  embedding instead of re-extracting it every sentence.
- **batch2 / batch4**: Qwen `batch_generate` with a shared reference. This is
  **output-changing** (it forces `repetition_penalty >= 1.5` and a per-seq
  max-token cap) and only runs with `--accept-output-changing`.

```sh
uv run python -m audiobook.perf profile          # measure phase timings
uv run python -m audiobook.perf benchmark        # baseline vs speaker-cache
uv run python -m audiobook.perf benchmark --include-batch --accept-output-changing
```

Qwen has no seed, so percentages are indicative only; objective structure and
persistent-ASR gates plus blind human A/B samples are required before any
adoption. This tool never touches production state, output, or generation
defaults.

## Commands at a glance

```sh
uv run audiobook preflight                         # check inputs and measure the pilot
uv run audiobook eta                               # project time from the measured pilot
uv run audiobook generate                          # generate resumable sentence WAVs
uv run audiobook validate                          # record strict ASR decisions
uv run python -m audiobook.adjudicate --assemble   # record overrides and assemble the WAV
uv run python -m audiobook.package                 # write the chaptered M4B
```

Run any of them with `--help` for the full list of flags.
