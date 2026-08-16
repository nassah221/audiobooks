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

- The book and the reference audio. These are **copyrighted material for your
  private use only** — keep them out of git and do not share them. This
  project is a private pipeline, so it is not set up to be published.

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

`uv sync` copies no book and no audio (they are not stored in git). Put yours
in the project:

```sh
cp path/to/your-book.epub books/
cp /path/to/reference.wav references/voice/
cp /path/to/reference.txt references/voice/
```

The **reference** is a short recording of the voice you want (the WAV) plus
the exact text of what is said in it (the TXT). The tool clones that voice.
Chapter and book selection is still done on the command line (see Commands),
not in the config.

> The project ships frozen example hashes for the original book and voice.
> When you use your own files you must replace those hashes (next step), or
> `preflight` will refuse to start.

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
| `[asr] repo` | Model used to check sentences (leave alone) | `mlx-community/whisper-tiny` |
| `[asr] revision` | Exact ASR model version (leave alone) | `78c52a…` |

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

Every generated sentence is checked by an automated speech-recognition
pass. Sentences that fail are kept (so you do not lose the audio) but marked
for your review. When *every* planned sentence passes, `validate` itself
assembles the final `book.wav` by joining the sentence files byte-for-byte
(exact concatenation, no processing). If any sentence fails, that blocks
assembly until you deal with it or re-run.

## What you get

`preflight`, `generate`, `validate`, and `eta` write into the output folder
(`outputs/qwen-book` by default): the generated audio, per-sentence notes, the
validation records, and the timing data. Once every sentence passes
validation, the chapters are assembled into the final audiobook.

Output audio is saved as PCM16 WAVs — the raw pipeline result, with no
mastering or post-processing.

## Copy to the M2 Pro

The target machine is the M2 Pro. To move a finished project there:

1. Copy the project folder (code, `audiobook.toml`, your book, reference WAV
   and TXT) to the M2 Pro. Do not copy git history if you keep this private.
2. Run `uv sync --locked` there. Keep the model cache, or let the first
   `preflight` re-download it (it works offline afterwards).
3. Run the commands above from that folder. The config and checkpoints travel
   with the project, so you can resume where you left off.

## How it fits together (for tweakers)

The pipeline is five Python files under `src/audiobook/`. The whole flow is:

```
audiobook.toml -> config.py -> epub.py -> runner.py -> asr.py -> book.wav
  book/voice/      reads the     reads the   speaks each   checks each   final
  model/asr        settings      book text   sentence,     sentence      audiobook
  settings                       chapters    WAV checkpointed

File map:

- `config.py` — reads `audiobook.toml` (stdlib `tomllib`), finds the project
  root, and holds the settings.
- `cli.py` — the `audiobook` commands (`preflight`, `eta`, `generate`,
  `validate`) and their flags.
- `epub.py` — turns the EPUB into chapter text: extracts, normalizes
  typography/numbers, splits into sentences.
- `runner.py` — the core. Loads the model once, and for every sentence
  generates a WAV, saves it as an atomic checkpoint, and records progress.
  Also does the measured pilot, ETA math, and chapter assembly.
- `asr.py` — the speech-recognition pass that checks each sentence and writes
  PASS/FAIL records.

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
uv run audiobook preflight    # check inputs, hashes, model, env; measure pilot
uv run audiobook eta          # projected time from the measured pilot
uv run audiobook generate     # resumable, sentence-by-sentence generation
uv run audiobook validate     # check each sentence, block final assembly on failures
```

Run any of them with `--help` for the full list of flags.
