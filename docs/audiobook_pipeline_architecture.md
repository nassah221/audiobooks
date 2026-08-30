# Audiobook pipeline architecture

This is the pipeline we run now. It turns an EPUB and a short voice reference into validated PCM chunks, an assembled WAV, and a chaptered M4B.

## Flow

```text
audiobook.toml
    ↓
EPUB extraction and paragraph-bounded planning
    ↓
Qwen voice-cloned generation
    ↓
Whisper and signal validation
    ↓
recorded lexical adjudication when needed
    ↓
source-aware WAV assembly
    ↓
M4B packaging
```

`audiobook.toml` pins the book, voice reference, model, validator, and hashes. A run checks those inputs before using an existing output directory.

## Text and planning

`epub.py` extracts chapter text and keeps paragraph boundaries. It groups complete sentences toward 70 words with an 85-word ceiling. Long or repeatedly failing chunks can fall back to planned clause children; assembly uses a parent only when it passes, otherwise it requires the complete child set.

`normalize_for_tts` performs narrow speak-time repairs without changing plan IDs or source-text hashes. Current repairs cover known number forms, regnal numerals, abbreviations, glued words, publisher defects, and the lost footnote boundary after “sixteen hundred.” The same spoken text is used as the ASR expectation.

Historical spelling equivalence is curated. For example, `K'ang-hsi` and `Kangxi` are accepted as the same name. Garbled renderings are not accepted automatically.

## Voice reference

The configured reference is:

- `references/bragg/arab-conquests-v3.wav`
- `references/bragg/arab-conquests-v3.txt`

This is the production reference. Its pace, diction, and terminal behavior have been validated on representative generated speech.

A new reference should be contiguous, single-speaker narration with an exact transcript, a natural declarative ending, useful punctuation, and clean margins. Run the reference mastery check before generation:

```sh
uv run python -m audiobook.mastery reference.wav reference.txt --json
```

The check reports ASR coverage, duration, boundary levels, SNR, pauses, true LUFS, clipping, spectral balance, and pinned DNSMOS P.835 scores. These are screening measurements. Listening remains the final decision when a boundary or perceptual score is marginal.

## Generation and the end of speech

`qwenfix.py` contains the Qwen ICL generation loop and exact codec-frame decoder path.

The stock mlx-audio valid-length calculation can shorten audio when codec token zero appears. We bypass it and decode the exact number of generated frames.

When Qwen first samples EOS, we discard that EOS token and resample the same step with EOS disabled. We keep that single replacement frame and stop. The replacement gives the causal decoder one frame of right context so it can finish the previous speech release.

We stop after the replacement frame. Additional continuation frames are not part of the current generation contract.
The current policy is `icl-nocontext-eos-replacement-v4`:

- one EOS replacement frame;
- zero later held frames;
- rolling context disabled;
- bounded tail trimming and a short final fade;
- one model load per run;
- atomic WAV checkpoints.

Rolling context remains disabled because it caused cumulative dull, underwater timbre drift. Every chunk is conditioned on the canonical reference rather than the previous generated chunk.

## Validation

Every take is checked before it can enter normal assembly.

The validator checks:

- expected speech order and coverage;
- mandatory words and phrases;
- terminal phrase completion;
- repeated text;
- required punctuation pauses;
- signal level and active audio;
- speech extending into the reserved tail;
- high-frequency evidence for final sibilants.

When retries are exhausted, the last generated WAV is retained when one exists, together with its reasons, seed, hashes, and failed-state entry. Generation defers that chunk rather than stopping the whole run. Assembly refuses incomplete or structurally failed material.

The pipeline does not currently have a reliable automatic detector for brief clicks, stutters, or clipped breaths. Those defects require representative listening during a pilot or chapter audit.

## Assembly

Assembly reloads the accepted PCM16 WAVs and preserves their samples exactly. It inserts only enough silence to reach the current minimum boundary:

- 250 ms between units in the same paragraph;
- 500 ms across paragraphs.

Native boundary silence counts toward that minimum. There are no crossfades, continuous room-tone loops, EQ, compression, expansion, denoising, or loudness processing in the production assembly.

`package.py` reads EPUB metadata and packages the adjudicated WAV as AAC in an M4B container with chapters and cover art.

## Current evidence

The Chapter 3 v4 pilot is stored locally at `outputs/pilots/ch03-v4/`. It is intentionally gitignored because the WAV is about 337 MB.

The pilot has:

- 305 of 305 source groups represented;
- 381 effective audio parts;
- 380 audited joins;
- 381 of 381 source WAVs sample-identical outside inserted silence;
- 12 of 12 representative listening clips marked complete, artifact-free, and well paced.

`promotion-manifest.json` contains the permanent relative paths and SHA-256 hashes for the WAV and its five provenance files.

## Current design boundaries

The production pipeline does not use:

- rolling generated-audio context;
- continuous room tone under the book;
- generic EQ, expansion, or mastering repairs;
- universal chunk crossfades;
- generated text rewriting with a language model.

Current gaps:

- no trustworthy automatic detector for clicks, stutters, and clipped breaths;
- no second speech judge or forced aligner for difficult names;
- no full assembled-chapter transcription against source text;
- book-specific aliases and text repairs still live in Python;
- reviewed human overrides are assembled outside the normal strict run;
- large audited pilot artifacts require storage outside Git.
