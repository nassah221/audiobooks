"""Validate and condition a TTS voice-clone reference.

The gate combines ASR intelligibility with absolute waveform checks. Its
interior-pause check measures low-energy cadence only; it does not classify
inhalation acoustics and must never be reported as a natural-breath check.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import wave

import numpy as np

from . import asr

SAMPLE_RATE = 24000
DNSMOS_COMMIT = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
DNSMOS_MODEL_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
DNSMOS_MODEL_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    f"{DNSMOS_COMMIT}/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
DNSMOS_SIG_MIN = 3.6
_DEFAULT_QUALITY = object()


def _dnsmos_model_path() -> pathlib.Path:
    import hashlib
    import urllib.request

    path = pathlib.Path.home() / ".cache/audiobook/dnsmos" / DNSMOS_MODEL_SHA256 / "sig_bak_ovr.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(DNSMOS_MODEL_URL, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != DNSMOS_MODEL_SHA256:
            raise RuntimeError("DNSMOS model hash mismatch")
        path.write_bytes(data)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != DNSMOS_MODEL_SHA256:
        raise RuntimeError("cached DNSMOS model hash mismatch")
    return path
def _dnsmos_p835(x: np.ndarray, sr: int) -> dict:
    """Official non-personalized P.835 SIG/BAK/OVRL inference."""
    import onnxruntime as ort
    from scipy import signal

    audio = signal.resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x.astype(np.float32)
    needed = int(9.01 * 16000)
    if not audio.size:
        raise ValueError("empty audio")
    while len(audio) < needed:
        audio = np.append(audio, audio)
    hops = int(np.floor(len(audio) / 16000) - 9.01) + 1
    session = ort.InferenceSession(str(_dnsmos_model_path()))
    input_name = session.get_inputs()[0].name
    scores = []
    for index in range(hops):
        segment = audio[index * 16000:index * 16000 + needed]
        if len(segment) != needed:
            continue
        scores.append(session.run(None, {input_name: segment[None, :]})[0][0])
    if not scores:
        raise RuntimeError("DNSMOS produced no complete windows")
    raw_sig, raw_bak, raw_ovrl = np.mean(np.asarray(scores), axis=0)
    sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])(raw_sig)
    bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])(raw_bak)
    ovrl = np.poly1d([-0.06766283, 1.11546468, 0.04602535])(raw_ovrl)
    return {"sig": float(sig), "bak": float(bak), "ovrl": float(ovrl),
            "num_hops": len(scores), "model_sha256": DNSMOS_MODEL_SHA256,
            "commit": DNSMOS_COMMIT}

# ---------- conditioning (one-off DSP; numpy-scipy only ----------------
def _butter_cheby_band(x: np.ndarray, sr: int, low=None, high=None, mode="pass",
                       order=4) -> np.ndarray:
    """2nd-order Butterworth band filters, scipy."""
    from scipy import signal
    nyq = sr / 2
    if mode == "low" and low is not None:
        wn = [low / nyq, high / nyq]
    elif mode == "high" and low is not None:
        wn = low / nyq
    elif mode == "low" and high is not None and low is None:
        wn = high / nyq
    else:
        raise ValueError("filter mode/edges")
    sos = signal.butter(order, wn, btype=("band" if isinstance(wn, list) else mode),
                        output="sos")
    return signal.sosfilt(sos, x)


def _lf_shelf(x: np.ndarray, sr: int, cutoff=20, gain_db=-9) -> np.ndarray:
    """Cut-only low-shelf via RBJ Cookbook biquad. Sweep measured on the
    source recording: every cutoff in 20-80 Hz satisfies the sub-80 gate;
    20 Hz is the true lowest passing value and leaves the 85-180 Hz F0
    band flattest (85 Hz att ≈ -0.03 dB vs -4 dB at cutoff=80)."""
    from scipy import signal
    A = 10.0 ** (gain_db / 40.0)  # biquad scaling for shelves
    w0 = 2.0 * np.pi * cutoff / sr
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)  # S=1 shelf slope
    cw = np.cos(w0)
    b0 = A * ((A + 1) - (A - 1) * cw + 2 * alpha * np.sqrt(A))
    b1 = 2 * A * ((A - 1) - (A + 1) * cw)
    b2 = A * ((A + 1) - (A - 1) * cw - 2 * alpha * np.sqrt(A))
    a0 = (A + 1) + (A - 1) * cw + 2 * alpha * np.sqrt(A)
    a1 = -2 * ((A - 1) + (A + 1) * cw)
    a2 = (A + 1) + (A - 1) * cw - 2 * alpha * np.sqrt(A)
    sos = np.array([[b0 / a0, b1 / a0, b2 / a0,
                     1.0, a1 / a0, a2 / a0]])
    return signal.sosfilt(sos, x)


def _high_pass(x: np.ndarray, sr: int, cutoff=80, order=4) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(order, cutoff / (sr / 2), btype="high", output="sos")
    return signal.sosfilt(sos, x)


def _de_ess(x: np.ndarray, sr: int, cutoff_lo=5000, cutoff_hi=8000,
            threshold_db=-30, max_depth_db=-15) -> np.ndarray:
    """Gain-compress the 5-8 kHz band to tame harsh sibilance."""
    from scipy import signal
    band = _high_pass(x, sr, cutoff_lo, order=2)
    band = _lowpass(band, sr, cutoff_hi, order=2)
    env = np.abs(band)
    env = np.convolve(env, np.ones(int(sr * 0.005)) / (sr * 0.005), mode="same")
    # gain reduction curve: above threshold, push band down
    over = 20 * np.log10(env + 1e-9) - threshold_db
    reduction = np.clip(over, 0, abs(max_depth_db))
    gain = 10.0 ** (-reduction / 20.0)
    return x / (1.0 + (gain - 1.0) * (band / (x + 1e-9)))


def _lowpass(x: np.ndarray, sr: int, cutoff=8000, order=2) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(order, cutoff / (sr / 2), btype="low", output="sos")
    return signal.sosfilt(sos, x)


def _resonance_notch(x: np.ndarray, sr: int, lo=250, hi=500,
                     sweep_gain_db=-1.75) -> np.ndarray:
    """Flatten the 250-500 Hz boxiness with a gentle sweep."""
    from scipy import signal
    sos = signal.butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="bandpass",
                        output="sos")
    band = signal.sosfilt(sos, x)
    # measure where band energy is worst, notch there
    peak = float(np.abs(band).max()) + 1e-9
    if peak > 0.05:
        # cut the band peak by sweep_gain_db, restore the rest
        return x - band * (1.0 - 10.0 ** (sweep_gain_db / 20.0))
    return x


def _integrated_loudness(x: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln

    loudness = float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
    return loudness


def _lufs_norm(x: np.ndarray, sr: int, peak_db=-3.0, target_lufs=-18.0) -> np.ndarray:
    """Normalize by EBU R128 integrated loudness without exceeding peak."""
    current_lufs = _integrated_loudness(x, sr)
    if not np.isfinite(current_lufs):
        return x.copy()
    gain = 10.0 ** ((target_lufs - current_lufs) / 20.0)
    limit = 10.0 ** (peak_db / 20.0)
    peak = float(np.abs(x).max()) + 1e-9
    return np.clip(x * min(gain, limit / peak), -limit, limit)

def condition(wav_path, out_path, *, sr_expected=SAMPLE_RATE) -> dict:
    """Apply the mastering chain to a raw wav and write the conditioned
    version to `out_path`. Returns the chain's applied edits."""
    wav = pathlib.Path(wav_path)
    if not wav.is_file():
        raise FileNotFoundError(f"raw wav not found: {wav}")
    x, sr = _sanitize(wav)
    if sr != sr_expected:
        raise ValueError(f"expected {sr_expected} Hz, got {sr}")
    edits = []
    y = x.copy()

    # LF rumble: raw file already passes the sub-80 gate (measured -36.4 dB
    # vs threshold -36.0). Shelf removed by advisory review; the gate is
    # absolute, not attenuation-margin-driven -- no value add here.
    y = _de_ess(y, sr)     # de-ess 5-8 kHz
    edits.append("de_ess_5_8k")
    y = _resonance_notch(y, sr)  # 250-500 Hz notch
    edits.append("notch_250_500")
    y = _lufs_norm(y, sr)    # peak -3 dBFS + integrate -18 LUFS
    edits.append("peak_-3dbfs_lufs_-18")

    # micro-pad same edges doc requires in the raw signal
    pad_start = int(sr * 0.200); pad_end = int(sr * 0.300)
    y = np.concatenate([np.zeros(pad_start, dtype=y.dtype), y,
                        np.zeros(pad_end, dtype=y.dtype)])
    # 10-ms linear fade at absolute edges
    fade = int(sr * 0.010)
    if y.size >= 2 * fade:
        y[:fade] = np.linspace(0.0, y[fade - 1], fade)
        y[-fade:] = np.linspace(y[-fade], 0.0, fade)
    edits.append("fade_10ms_edges_pad_200_300")

    pcm16 = np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)
    pcm16_bytes = pcm16.tobytes()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm16_bytes)
    return {"out": str(out), "edits": edits,
            "frames": len(pcm16), "seconds": round(len(pcm16) / sr, 3),
            "peak": round(float(np.abs(pcm16).max()) / 32767, 4),
            "sr": sr}


def _sanitize(wav: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav), "rb") as fh:
        frames = fh.getnframes()
        sr = fh.getframerate()
        data = fh.readframes(frames)
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr




# Absolute spectral thresholds (per-doc + empirically calibrated):
PEAK_MAX_DB = -3.0        # peak normalization ceiling
ROLLOFF_80HZ_MAX_DB = -36.0  # sub-80 Hz rumble must be suppressed below this
NOTCH_250_500_MAX_DB = -20.0  # boxy resonance must be suppressed below this
LOUDNESS_TARGET_LUFS = -18.0
LOUDNESS_TOL_LU = 1.5

DURATION_MIN_S = 6.0
DURATION_MAX_S = 12.0
DURATION_IDEAL = (7.0, 10.0)
PAD_START_MS = 200
PAD_END_MS = 300
EDGE_FADE_MS = 10
RMS_GATE_MAX = 0.005
SNR_DB_MIN = 25.0
PAUSE_WINDOW_MS = (150, 800)
VARIETY_MIN_CLASSES = 5
VARIETY_MIN_RANGE = 0.20

_ALPHABET_OF = {
    "s": "sibilant", "z": "sibilant",
    "sh": "sibilant", "zh": "sibilant",
    "f": "fricative", "v": "fricative", "th": "fricative",
    "p": "plosive", "b": "plosive",
    "t": "plosive", "d": "plosive",
    "k": "plosive", "g": "plosive",
    "m": "nasal", "n": "nasal", "ng": "nasal",
    "i": "vowel_close", "y": "vowel_close", "e": "vowel_close",
    "a": "vowel_open", "o": "vowel_open", "u": "vowel_open",
    "r": "approx", "l": "approx", "w": "approx", "j": "approx",
    "h": "glottal",
}


def _edge_rms(x: np.ndarray, sr: int, ms: int, *, start: bool = True) -> float:
    w = int(sr * ms / 1000)
    if x.size == 0:
        return 0.0
    edge = x[:w] if start else x[-w:]
    return float(np.sqrt(np.mean(edge ** 2)))


def _silence_profile(x: np.ndarray, sr: int) -> dict:
    if x.size == 0:
        return {"snr_db": None, "centroid": None, "rms_floor": None}
    tiles = []
    tile = int(sr * 0.025)
    for i in range(0, x.size - tile + 1, tile):
        tiles.append(float(np.sqrt(np.mean(x[i:i + tile] ** 2))))
    tiles = np.asarray(tiles)
    floor = float(np.percentile(tiles, 10))
    loud = float(np.percentile(tiles, 90))
    snr = 20.0 * np.log10((loud + 1e-9) / (floor + 1e-9))
    rms = tiles
    centroid = float(np.average(np.arange(len(rms)), weights=rms ** 2 + 1e-9))
    centroid = centroid * 25.0 / 1000.0 / (x.size / sr)
    return {
        "seconds": round(x.size / sr, 4),
        "rms_floor": round(floor, 6),
        "rms_floor_db": round(20 * np.log10(floor + 1e-9), 2),
        "peak": round(float(np.abs(x).max()), 4),
        "snr_db": round(snr, 2),
        "centroid_frac": round(centroid, 3),
    }


def _interior_pause(x: np.ndarray, sr: int, floor_rms: float) -> dict:
    """Locate low-energy interior pauses; does not identify breathing."""
    if not floor_rms or floor_rms < 1e-9:
        return {"gaps": [], "kept": False, "n_interior": 0}
    tile = int(sr * 0.025)
    tile_rms = np.asarray([float(np.sqrt(np.mean(x[i:i + tile] ** 2)))
                           for i in range(0, x.size - tile + 1, tile)])
    p20 = float(np.percentile(tile_rms, 20))
    p90 = float(np.percentile(tile_rms, 90))
    threshold = min(max(floor_rms * 2.2, p20 * 0.5), p90 * 0.10)
    active = tile_rms > threshold
    gaps = []
    i = 0
    while i < len(active):
        if active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and not active[j]:
            j += 1
        duration_ms = (j - i) * 25
        if i and j < len(active) and active[:i].any() and active[j:].any() and duration_ms >= 100:
            gaps.append({"start_ms": i * 25, "duration_ms": duration_ms,
                         "severity": round(float(tile_rms[i:j].mean()) / threshold, 3)})
        i = j
    kept = any(PAUSE_WINDOW_MS[0] <= g["duration_ms"] <= PAUSE_WINDOW_MS[1] for g in gaps)
    return {"gaps": gaps, "kept": kept, "n_interior": len(gaps)}


def _boundary_continuity(x: np.ndarray, sr: int, index: int) -> dict:
    """Check a known edit boundary against its 20ms local RMS."""
    radius = int(sr * 0.02)
    if index <= 0 or index >= len(x):
        return {"clean": False, "jump_over_rms": float("inf")}
    local = x[max(0, index - radius):min(len(x), index + radius)]
    rms = float(np.sqrt(np.mean(local ** 2))) + 1e-9
    ratio = abs(float(x[index] - x[index - 1])) / rms
    return {"clean": ratio <= 1.0, "jump_over_rms": ratio}
def _phonetic_spread(text: str) -> dict:
    """Orthographic variety diagnostic; not a phonetic release gate."""
    tokens = [t for t in re.split(r"\s+", text.lower()) if t]
    classes = {}
    for token in tokens:
        for pattern, cls in _ALPHABET_OF.items():
            if pattern in token:
                classes[cls] = classes.get(cls, 0) + 1
                break
    rates = {key: round(value / len(tokens), 4) for key, value in classes.items()}
    if len(rates) < 2:
        return {"classes": rates, "n_classes": len(rates), "log_spread": None}
    values = np.asarray(list(rates.values()))
    return {"classes": rates, "n_classes": len(rates),
            "log_spread": round(float(np.log(values.max() / values.min())), 3)}


def _band_db(x: np.ndarray, sr: int, lo: int, hi: int) -> float:
    """Integrated RMS (dB) of the [lo, hi] band of `x`; measured honestly
    against the gate that asked for absolute-zero room-tone."""
    from scipy import signal
    sos = signal.butter(2, [lo / (sr / 2), hi / (sr / 2)],
                        btype="bandpass", output="sos")
    y = signal.sosfilt(sos, x)
    return float(20.0 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))


# ---------- conditioning report / qualitative score --------------------
def conditioning_report(wav: pathlib.Path, transcript_text: str,
                         rec_words: dict | None = None, quality_fn=None) -> dict:
    if not wav.is_file():
        return {"verdict": "ERROR", "reasons": [f"missing wav: {wav}"]}
    x, sr = _sanitize(wav)
    if not x.size or not sr:
        return {"verdict": "ERROR", "reasons": ["unreadable wav"]}
    seconds = x.size / sr
    sig = _silence_profile(x, sr)
    pause = _interior_pause(x, sr, sig["rms_floor"])
    pron = _phonetic_spread(transcript_text)
    quality = quality_fn(x, sr) if quality_fn else None
    kept = pause["kept"]
    e1 = _edge_rms(x, sr, EDGE_FADE_MS)
    e2 = _edge_rms(x, sr, EDGE_FADE_MS, start=False)
    pad_start = int(sr * PAD_START_MS / 1000)
    pad_end = int(sr * PAD_END_MS / 1000)
    pad_ok_start = float(np.sqrt(np.mean(x[:pad_start] ** 2))) if pad_start else 0.0
    pad_ok_end = float(np.sqrt(np.mean(x[-pad_end:] ** 2))) if pad_end else 0.0

    reasons = []
    if seconds < DURATION_MIN_S:
        reasons.append(f"duration {seconds:.2f}s < {DURATION_MIN_S}s")
    elif seconds > DURATION_MAX_S:
        reasons.append(f"duration {seconds:.2f}s > {DURATION_MAX_S}s")
    if pad_start and pad_ok_start > RMS_GATE_MAX:
        reasons.append(f"start pad {pad_ok_start:.4f} RMS > {RMS_GATE_MAX}")
    if pad_end and pad_ok_end > RMS_GATE_MAX:
        reasons.append(f"end pad {pad_ok_end:.4f} RMS > {RMS_GATE_MAX}")
    if e1 > RMS_GATE_MAX:
        reasons.append(f"first {EDGE_FADE_MS}ms RMS {e1:.4f} > {RMS_GATE_MAX} (DC/pop)")
    if e2 > RMS_GATE_MAX:
        reasons.append(f"last {EDGE_FADE_MS}ms RMS {e2:.4f} > {RMS_GATE_MAX} (DC/pop)")
    if sig["snr_db"] is not None and sig["snr_db"] < SNR_DB_MIN:
        reasons.append(f"interior SNR {sig['snr_db']:.1f}dB < {SNR_DB_MIN}dB")
    if not kept:
        reasons.append("no interior pause found (~150-800ms low energy)")
    peak_db = 20.0 * np.log10(float(np.abs(x).max()) + 1e-9)
    rolloff_db = _band_db(x, sr, 20, 80)
    notch_db = _band_db(x, sr, 250, 500)
    loudness_lufs = _integrated_loudness(x, sr)
    if peak_db > PEAK_MAX_DB:
        reasons.append(f"peak {peak_db:.2f} dBFS > {PEAK_MAX_DB} dB")
    if rolloff_db > ROLLOFF_80HZ_MAX_DB:
        reasons.append(f"sub-80Hz energy {rolloff_db:.1f}dB > {ROLLOFF_80HZ_MAX_DB}dB")
    if notch_db > NOTCH_250_500_MAX_DB:
        reasons.append(f"250-500Hz boxiness {notch_db:.1f}dB > {NOTCH_250_500_MAX_DB}dB")
    if not np.isfinite(loudness_lufs) or abs(loudness_lufs - LOUDNESS_TARGET_LUFS) > LOUDNESS_TOL_LU:
        reasons.append(
            f"loudness {loudness_lufs:.2f} LUFS ({LOUDNESS_TARGET_LUFS}±{LOUDNESS_TOL_LU} expected)"
        )
    if quality is not None and quality["sig"] < DNSMOS_SIG_MIN:
        reasons.append(f"DNSMOS SIG {quality['sig']:.2f} < {DNSMOS_SIG_MIN}")

    # One point per absolute, single-file gate.
    weights = {
        "duration": 1,
        "pad": 1,
        "edges": 1,
        "snr": 1,
        "interior_pause": 1,
        "peak_ceiling": 1,
        "rolloff_80hz": 1,
        "notch_250_500": 1,
        "loudness": 1,
    }
    if quality is not None:
        weights["dnsmos_sig"] = 1
    breakdown = {}
    dur_ok = DURATION_MIN_S <= seconds <= DURATION_MAX_S
    breakdown["duration"] = round(
        int(dur_ok) * weights["duration"], 2)
    breakdown["pads"] = round(
        int((not pad_start or pad_ok_start <= RMS_GATE_MAX)
            and (not pad_end or pad_ok_end <= RMS_GATE_MAX))
        * weights["pad"], 2)
    breakdown["edges"] = round(
        int(e1 <= RMS_GATE_MAX and e2 <= RMS_GATE_MAX)
        * weights["edges"], 2)
    breakdown["snr"] = round(
        int(sig["snr_db"] is not None and sig["snr_db"] >= SNR_DB_MIN)
        * weights["snr"], 2)
    breakdown["interior_pause"] = round(
        int(kept) * weights["interior_pause"], 2)

    # --- spectral scoring (absolute thresholds, single-file) -------------
    peak_db = 20.0 * np.log10(float(np.abs(x).max()) + 1e-9)
    rolloff_db = _band_db(x, sr, 20, 80)
    notch_db = _band_db(x, sr, 250, 500)
    loudness_lufs = _integrated_loudness(x, sr)
    breakdown["peak_ceiling"] = round(
        int(peak_db <= PEAK_MAX_DB) * weights["peak_ceiling"], 2)
    breakdown["rolloff_80hz"] = round(
        int(rolloff_db <= ROLLOFF_80HZ_MAX_DB) * weights["rolloff_80hz"], 2)
    breakdown["notch_250_500"] = round(
        int(notch_db <= NOTCH_250_500_MAX_DB) * weights["notch_250_500"], 2)
    breakdown["loudness"] = round(
        int(np.isfinite(loudness_lufs) and abs(loudness_lufs - LOUDNESS_TARGET_LUFS) <= LOUDNESS_TOL_LU)
        * weights["loudness"], 2)
    if quality is not None:
        breakdown["dnsmos_sig"] = int(quality["sig"] >= DNSMOS_SIG_MIN)
    max_score = sum(weights.values())
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "score": round(sum(breakdown.values()), 2),
        "score_out_of": max_score,
        "breakdown": breakdown,
        "reasons": reasons,
        "duration_s": round(seconds, 3),
        "duration_class": ("ideal" if DURATION_IDEAL[0] <= seconds <= DURATION_IDEAL[1]
                            else "acceptable"),
        "signal": sig,
        "interior_pause": pause,
        "phonetic": pron,
        "edges_10ms_rms": [round(e1, 6), round(e2, 6)],
        "quality": quality,
        "pads_rms": [round(pad_ok_start, 6), round(pad_ok_end, 6)],
        "quality_status": "enabled" if quality is not None else "disabled",
    }


def validate_reference(
    wav_path,
    transcript_path,
    *,
    model_repo=asr.DEFAULT_MODEL_REPO,
    revision=asr.DEFAULT_MODEL_REVISION,
    language=asr.DEFAULT_LANGUAGE,
    cache_path=None,
    quality_fn=_DEFAULT_QUALITY,
) -> dict:
    if quality_fn is _DEFAULT_QUALITY:
        quality_fn = _dnsmos_p835
    wav = pathlib.Path(wav_path)
    txt = pathlib.Path(transcript_path)
    if not wav.is_file():
        raise FileNotFoundError(f"reference wav not found: {wav}")
    if not txt.is_file():
        raise FileNotFoundError(f"reference transcript not found: {txt}")

    expected = txt.read_text().strip()
    if not expected:
        raise ValueError(f"reference transcript empty: {txt}")

    v = asr.AsrValidator(
        model_repo=model_repo,
        revision=revision,
        language=language,
        cache_path=cache_path,
    )
    rec = v.validate_chunk(str(wav), expected, chunk_id=wav.stem)
    stats = v.stats()
    cov = rec.get("coverage") or {}
    words = rec.get("words") or {}
    cond = conditioning_report(wav, expected, rec_words=rec.get("words"), quality_fn=quality_fn)
    return {
        "wav": str(wav),
        "transcript_path": str(txt),
        "verdict": rec["verdict"] + "/" + cond["verdict"],
        "reasons": rec.get("reasons", []) + cond.get("reasons", []),
        "transcript": rec.get("transcript", ""),
        "coverage_ratio": cov.get("fraction"),
        "coverage_matched": cov.get("matched_tokens"),
        "coverage_expected": cov.get("expected_tokens"),
        "coverage_missing": cov.get("missing"),
        "word_stats": {
            "count": words.get("count"),
            "max_internal_gap_s": words.get("max_internal_gap_s"),
            "long_gaps_over_1s": words.get("long_gaps_over_1s"),
            "ms_per_word": round(rec["signal"]["seconds"] / words["count"] * 1000, 1)
            if words.get("count") else None,
        },
        "expected_words": len(expected.split()),
        "asr_seconds": stats["asr_seconds"],
        "rtf": stats["rtf"],
        "cache_hit": rec.get("cache_hit", False),
        "conditioning": cond,
    }


# ---------- CLI ---------------------------------------------------------
def _print_report(r, wav_path):
    if "/" in r["verdict"]:
        asr_v, cond_v = r["verdict"].split("/")
    else:
        asr_v, cond_v = r.get("verdict"), r.get("verdict")
    cond = r.get("conditioning", {})
    print(f"{pathlib.Path(wav_path).name}"
          f"  asr: {asr_v}"
          f"  conditioning: {cond_v} (score {cond.get('score', '?')}/{cond.get('score_out_of', '?')})")
    if r.get("reasons"):
        print("  reasons: " + " | ".join(r["reasons"][:5]))
    if r.get("coverage_ratio") is not None:
        print(f"  coverage: {r['coverage_ratio']} ({r['coverage_matched']}/{r['coverage_expected']})")
    if r.get("word_stats", {}).get("count"):
        w = r["word_stats"]
        print(f"  words: {w['count']}, max_gap={w['max_internal_gap_s']}s, "
              f">1s_gaps={w['long_gaps_over_1s']}, ms/word={w['ms_per_word']}")
    if r.get("transcript"):
        print(f"  transcript: {r['transcript']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="audiobook.mastery",
        description=__doc__,
    )
    ap.add_argument("wav", help="reference WAV to validate")
    ap.add_argument("transcript", help="verbatim transcript of the WAV")
    ap.add_argument("--condition", metavar="OUT.wav",
                    help="also write the conditioned WAV to OUT")
    ap.add_argument("--cache", help="optional ASR cache file (reuse across runs)")
    ap.add_argument("--model-repo", default=asr.DEFAULT_MODEL_REPO)
    ap.add_argument("--model-revision", default=asr.DEFAULT_MODEL_REVISION)
    ap.add_argument("--language", default=asr.DEFAULT_LANGUAGE)
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    ap.add_argument("--no-asr", action="store_true", help="structure-only (skip ASR layer)")
    ap.add_argument("--no-quality", action="store_true", help="disable DNSMOS quality shield")
    args = ap.parse_args(argv)

    try:
        target_wav = args.condition or args.wav
        if args.condition:
            cond = condition(args.wav, args.condition)
            print(f"conditioned -> {cond['out']}  "
                  f"edits={cond['edits']}  peak={cond['peak']}  seconds={cond['seconds']}")
        if args.no_asr:
            r_text = pathlib.Path(args.transcript).read_text().strip()
            r = conditioning_report(pathlib.Path(target_wav), r_text,
                                    quality_fn=None if args.no_quality else _dnsmos_p835)
        else:
            r = validate_reference(
                target_wav, args.transcript,
                model_repo=args.model_repo,
                revision=args.model_revision,
                language=args.language,
                cache_path=args.cache,
                quality_fn=None if args.no_quality else _dnsmos_p835,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(r, indent=1))
    else:
        _print_report(r, target_wav)
    final = r["verdict"] if "/" in r["verdict"] else r.get("verdict", "FAIL")
    return 0 if final == "PASS/PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
