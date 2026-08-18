"""Tail-safe ICL generation for the mlx-audio 0.4.8 Qwen3-TTS path.

Three defects clip or degrade the final word's release:

1. ``speech_tokenizer.decode`` computes valid length as
   ``(audio_codes[..., 0] > 0).sum() * decode_upsample_rate``. Token id 0
   is a valid codebook entry (the official implementation counts ``> -1``
   to skip only batch padding), so every generated frame whose first
   codebook token is 0 chops one frame of samples (80 ms) off the end of
   the decoded audio and shifts the proportional reference cut.
2. Every decoder upsample layer is a causal transpose conv that trims
   ``kernel - stride`` samples from the right. The final frame's
   overlap-add tail is discarded when no frame follows it, so a release
   that ends on the last generated frame loses its decay.
3. The model sometimes samples codec EOS before it has emitted the final
   consonant's codes at all. No decode-side repair can restore audio the
   model never generated.

``generate_icl_tail_safe`` replaces ``model.generate`` for this pipeline.
It is a vendored copy of ``_generate_icl``'s sampling loop with EOS-hold:
when the model first samples EOS, the step is resampled with EOS
suppressed and generation continues for ``EOS_HOLD_FRAMES`` more frames.
Measured on truncated and complete takes, the held frames decode to the
model's own room tone, which (a) supplies the right-context that restores
the trimmed release and (b) ends every chunk in take-specific natural
silence instead of a repeated pad motif (an earlier fixed room-tone pad
produced an audible stutter at every chunk end). Decoding is done here
with exact trims, bypassing the valid-length bug.

Defect 3 is caught by two validation metrics computed by the runner:
``tail_frame_peak`` (speech energy in the final 80 ms, which should be
room tone) and ``final_sibilant_high_frac`` (a final word that ends in a
sibilant must show 4-10 kHz energy near the end of speech; a clipped
"collapsed" measured 0.012 where complete takes measured 0.061-0.56).

MLX generation is not bit-reproducible across runs even with fixed seeds
(one floating-point flip at a sampling boundary changes the whole take),
so a retry is a fresh draw and these gates decide acceptance.
"""
from __future__ import annotations

import re

# Frames generated after the first sampled EOS (80 ms each).
EOS_HOLD_FRAMES = 6
# Keep at most this much trailing quiet after the last speech sample.
TAIL_MAX_SILENCE_SECONDS = 0.24
# Linear fade applied to the very end of the kept tail.
TAIL_FADE_SECONDS = 0.06
# The final 80 ms must be free of speech; room tone peaks at ~0.009.
TAIL_GATE_SECONDS = 0.08
TAIL_FRAME_PEAK_MAX = 0.02
# Final words that end in a sibilant sound (including -sed/-ced/-xed
# forms like "collapsed" /st/) must show high-band energy near the end.
SIBILANT_FINAL = re.compile(r"(s|se|ss|ce|x|z|sh|ch|ge|dge)(ed)?$", re.I)
SIBILANT_HIGH_FRAC_MIN = 0.03
SIBILANT_WINDOW_SECONDS = 0.35

_SILENCE_AMPLITUDE = 0.01


def generate_icl_tail_safe(model, text, ref_audio, ref_text, language,
                           max_tokens, hold_frames: int = EOS_HOLD_FRAMES):
    """Generate one chunk and decode it with exact trims.

    ref_audio is the reference waveform as an mx.array at the model's
    sample rate. Returns float32 numpy audio: generated speech plus a
    bounded, faded natural-silence tail.
    """
    import mlx.core as mx
    import numpy as np

    (input_embeds, trailing_text_hidden, tts_pad_embed, ref_codes) = (
        model._prepare_icl_generation_inputs(
            text=text, ref_audio=ref_audio, ref_text=ref_text,
            language=language))
    config = model.config.talker_config
    eos_id = config.codec_eos_token_id
    specials = list(range(config.vocab_size - 1024, config.vocab_size))
    suppress = [i for i in specials if i != eos_id]

    cache = model.talker.make_cache()
    code_cache = model.talker.code_predictor.make_cache()
    generated_codes = []
    generated_token_ids = []
    trailing_idx = 0
    eos_step = None
    held = 0

    for step in range(max_tokens):
        logits, hidden = model.talker(input_embeds, cache=cache)
        post_eos = eos_step is not None
        next_token = model._sample_token(
            logits, temperature=0.9, top_k=50, top_p=1.0,
            repetition_penalty=1.5,
            generated_tokens=(generated_token_ids or None),
            suppress_tokens=(specials if post_eos else suppress),
            eos_token_id=(None if post_eos else eos_id),
        )
        if not post_eos and bool((next_token[0, 0] == eos_id).item()):
            eos_step = step
            post_eos = True
            next_token = model._sample_token(
                logits, temperature=0.9, top_k=50, top_p=1.0,
                repetition_penalty=1.5,
                generated_tokens=(generated_token_ids or None),
                suppress_tokens=specials, eos_token_id=None,
            )

        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None
            c.values = None
            c.offset = 0
        for code_idx in range(config.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = model.talker.get_input_embeddings()(next_token)
                code_input = mx.concatenate(
                    [code_hidden, code_0_embed], axis=1)
            else:
                code_input = model.talker.code_predictor.codec_embedding[
                    code_idx - 1](code_tokens[-1])
            code_logits, code_cache, _ = model.talker.code_predictor(
                code_input, cache=code_cache, generation_step=code_idx)
            code_tokens.append(model._sample_token(
                code_logits, temperature=0.9, top_k=50, top_p=1.0))

        all_codes = mx.concatenate(code_tokens, axis=1)

        if trailing_idx < trailing_text_hidden.shape[1]:
            text_embed = trailing_text_hidden[
                :, trailing_idx:trailing_idx + 1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed
        codec_embed = model.talker.get_input_embeddings()(next_token)
        for i, code in enumerate(code_tokens[1:]):
            codec_embed = (codec_embed
                           + model.talker.code_predictor.codec_embedding[i](code))
        input_embeds = text_embed + codec_embed
        mx.eval(input_embeds)

        generated_token_ids.append(int(next_token[0, 0]))
        generated_codes.append(all_codes)

        if post_eos:
            held += 1
            if held >= hold_frames:
                break

    if not generated_codes:
        raise RuntimeError("generation produced no codec frames")

    gen_codes = mx.stack(generated_codes, axis=1)
    ref_t = mx.transpose(ref_codes, (0, 2, 1))
    full = mx.concatenate([ref_t, gen_codes], axis=1)

    # Decode everything in one pass; cut the reference region exactly.
    wav = model.speech_tokenizer.decoder.chunked_decode(
        mx.transpose(full, (0, 2, 1))).squeeze(1)
    mx.eval(wav)
    upsample = wav.shape[1] // full.shape[1]
    ref_len = ref_codes.shape[2]
    audio = np.asarray(wav[0, ref_len * upsample:].astype(mx.float32))

    sample_rate = model.sample_rate
    nonquiet = np.flatnonzero(np.abs(audio) > _SILENCE_AMPLITUDE)
    if nonquiet.size:
        keep = min(audio.shape[0],
                   int(nonquiet[-1]) + 1
                   + int(TAIL_MAX_SILENCE_SECONDS * sample_rate))
        audio = audio[:keep]
    fade = int(TAIL_FADE_SECONDS * sample_rate)
    if fade > 0 and audio.shape[0] > fade:
        audio = audio.copy()
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return audio


def tail_frame_peak(audio, sample_rate) -> float:
    """Peak amplitude of the final 80 ms, which should be room tone."""
    import numpy as np

    tail = np.asarray(audio)[-int(TAIL_GATE_SECONDS * sample_rate):]
    return float(np.max(np.abs(tail))) if tail.size else 0.0


def final_sibilant_high_frac(audio, sample_rate, text):
    """4-10 kHz energy fraction near the end of speech, or None.

    Returns None when the chunk's final word does not end in a sibilant
    sound; otherwise the maximum high-band fraction over 25 ms windows
    (RMS above 0.002) in the last 350 ms of speech. A complete sibilant
    measures well above SIBILANT_HIGH_FRAC_MIN; a clipped one does not.
    """
    import numpy as np

    words = [w for w in re.findall(r"[\w']+", text)]
    if not words or not SIBILANT_FINAL.search(words[-1].lower()):
        return None
    x = np.asarray(audio)
    nonquiet = np.flatnonzero(np.abs(x) > _SILENCE_AMPLITUDE)
    end = int(nonquiet[-1]) + 1 if nonquiet.size else x.shape[0]
    seg = x[max(0, end - int(SIBILANT_WINDOW_SECONDS * sample_rate)):end]
    w = int(0.025 * sample_rate)
    best = 0.0
    for i in range(len(seg) // w):
        win = seg[i * w:(i + 1) * w] * np.hanning(w)
        if float(np.sqrt(np.mean(win ** 2))) < 0.002:
            continue
        spec = np.abs(np.fft.rfft(win)) ** 2
        freqs = np.fft.rfftfreq(w, 1 / sample_rate)
        high = spec[(freqs >= 4000) & (freqs <= 10000)].sum()
        best = max(best, float(high / (spec.sum() + 1e-12)))
    return best
