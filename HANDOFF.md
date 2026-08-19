# Audiobook pipeline — hand-off

State as of 2026-08-19. Repo: local commits only, nothing pushed. Model: mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 (pinned revision in `audiobook.toml`), mlx-audio 0.4.8, Whisper large-v3-turbo for validation.

## Current state

- Chapter 1: complete, validated, assembled (`outputs/qwen-book-v2`). Hassan reviewed; do not regenerate.
- Chapter 2: 193 done, 184 staged as pending (`regenerate` archived their wavs as `.superseded-*`; restorable by renaming back + restoring `state.json.bak-*`). Frozen on Hassan's stop order.
- Chapter 3: 96/305 done, 2 deferred failures. Frozen.
- Chapters 4–9: not started. Endnotes: excluded. Preface/"note on names": undecided.
- Generation policy: `icl-nocontext-v3` (rolling context reverted — see below). Validation policy: `paragraph-v36`.
- Watch deliverable: AAC 96k .m4a per chapter (Galaxy Watch7). Final target: chaptered M4B via `package.py`.

## Voice reference requirements (plug & play starts here)

The ICL reference controls voice, pace, brightness, and termination behavior. Qualify the sample before any generation:

- **Duration**: 10–20 s, single speaker, zero music/SFX/crosstalk.
- **Room tone**: 200–500 ms of clean room tone at BOTH head and tail. Confirmed helpful for Qwen: the tail tone teaches the model to terminate cleanly (EOS behavior) instead of clipping final words.
- **Phrase-bounded**: starts at a phrase onset, ends on a completed sentence with a natural final fall. No mid-word or mid-breath cuts. Current file: `references/bragg/phrase-bounded.wav`.
- **Pace transfers.** The model clones the reference's words-per-minute. Current sample reads slightly fast; a slower documentary sample (~150 wpm) will slow the narration. This is the correct knob — do not post-process tempo.
- **Brightness transfers, with loss.** Generated takes measured ~50% of the reference's 2–8 kHz energy fraction. Start from a bright, close-mic'd, uncompressed sample; a dull reference compounds into a dull book.
- **Crisp sibilants** in the reference help final-sibilant rendering (we gate on it).
- **Technical**: 24 kHz+ mono, no clipping (peak ≤ −3 dBFS), no aggressive denoising artifacts, even loudness.
- **Verbatim transcript** of the reference is required (`ref_text` drives ICL token alignment). Transcribe it exactly, punctuation included.
- **Register match**: measured narration, not conversation. Varied punctuation in the sample (a comma, a period) helps pause prosody.

When swapping references: re-run a 12-chunk listening slice first; the tail gates (`tail_frame_peak`, sibilant) and pause gates are reference-agnostic, but re-check pace and brightness by ear before committing a chapter.

## What was built (one line each)

- **EOS-hold tail-safe generation** (`qwenfix.py`): vendored ICL loop; on first EOS, suppress and generate 6 extra frames; fixes clipped final words. Never pad with a fixed room-tone motif (audible stutter).
- **Structural gates**: tail truncation, final-sibilant energy, active ratio, room-tone-tail overrun; retries are fresh draws, gates decide acceptance.
- **Validation policy v7→v36**: number/word equivalence, abbreviation-aware boundaries, two narrow pause exemptions (colon+"that", semicolon+"and/or"), NFKD accent folding, hyphen/compound concatenation, ise/ize canonicalization, vowel-swap + curated transliteration pairs, phonetic demotion with audit records, ASR-fragile lexicon, per-token tolerance inside phrases, dual-parse ambiguous numbers. Every non-exact acceptance is recorded in the validation record.
- **Speak-time normalization** (`normalize_for_tts`): text fed to TTS and used as validation reference, plan identity untouched. Regnal numerals ("Abbas I" → "the First" — the model audibly misreads bare numerals), "c." → "circa", publisher-typo repairs ("entrepô t", glued "350ships").
- **Defer-mode**: a failed chunk records into `state.json.failed` and the run continues; assembly refuses while real failures exist; failures triage in batches at chapter boundaries.
- **State machinery**: run fingerprint = immutable inputs only; generation policy recorded per chunk (mixed-policy dirs legal); superset-plan resume; `--force` archives instead of zeroing (`--discard-done` for real discard); `regenerate --chunks <file>` for selective redo.
- **Rolling context: built, gated, REVERTED.** Conditioning on prior generated codes inherits and compounds spectral dullness ("underwater" drift). A drift gate bounded but did not eliminate it. Machinery remains in code behind `ROLLING_CONTEXT_ENABLED = False`. Do not re-enable without conditioning that carries prosody without timbre.
- **Batch-4 generation: parked** on branch `perf-batch4` (`BATCH4-STATUS.md` there). Correct (logit-equivalent) but never demonstrated ≥1.3×; two benchmarks were confounded; needs step-internal timing on a zero-retry slice.

## Known limitations (ranked)

1. **Text-identical mispronunciation is invisible**: if Whisper writes what the source says, a wrong pronunciation passes. Regnal numerals fixed at speak time; homographs and foreign names remain exposed.
2. **No per-take timbre gate**: spectral metrics are recorded, not gated. The naive high-band metric is confounded by consonant-sparse text; a gate needs a reference-relative measure.
3. **Single ASR judge**: the equivalence stack exists to compensate Whisper's habits. A second judge (forced alignment) would shrink it.
4. **No end-to-end audit**: chunks validate individually; nothing re-transcribes the assembled chapter against the full text.
5. **Digit fuzzy-matching**: "1505" can fuzzy-match "1500" (pending fix chip).

## Product direction (senior-engineer view)

Deterministic outputs are the value. Context-free generation made every chunk a pure function of (reference, chunk text, seed) — no cross-chunk dependencies. Byte-identical replays were observed across process restarts; treat full determinism as unsettled (one counterexample) and pin it down with a controlled experiment. What plug-&-play needs:

1. **Build lockfile**: one manifest per run — book hash, reference hash + transcript, model revision, generation + validation policy versions, seed base. Same lockfile → same book. The fingerprint split was the first step; formalize it as an artifact.
2. **Reference qualifier**: automated check of a candidate sample (duration, room tone at head/tail, SNR, clipping, pace estimate, brightness) so "plug in any ref" fails fast instead of generating a dull book.
3. **Metrics as a library**: one unit-tested module for all spectral/pause measurements. Two workers re-implementing "the same" metric produced 10×-different scales; never again.
4. **Book-specific tables, not code**: transliteration pairs, fragile lexicon, typo repairs, speak-time rules should load from a per-book TOML overlay. New book = new table, zero code.
5. **Second judge + end-to-end audit**: forced alignment per chunk; whole-chapter re-transcription diffed against source as the final gate before packaging.
6. **Extraction-time text repair**: fold speak-time publisher-typo fixes into extraction at the next full-book rerun, so plan identity is built from clean text.
7. **Throughput**: overlap ASR validation with generation (~19% serial today); revisit batch-4 only with honest instrumentation.
8. **One driver**: chapter loop, boundary triage, packaging as a single supervised command instead of orchestrated shell calls.

## Operational notes

- Never touch `outputs/qwen-book` (protected legacy run). `--force` on a populated dir only after reading the archive semantics. `--chapters` plans are widening-compatible now, but per-chapter dirs remain the layout.
- Blind re-resume can replay a failing draw byte-identically (seed is a pure function of chunk+attempt). Check wav hashes before burning a retry.
- Selfcheck suites: asr 242, runner 143, epub 34, adjudicate 18. Run all four after any pipeline change; add a failing check before every fix.
- Worker handoff for the chapter loop: `outputs/handoff-ch3.md`. Drift analysis: `outputs/drift-paired-report.md`, `outputs/drift-sweep-ch02.md`.
