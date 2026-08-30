import contextlib
import io
import pathlib
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

from audiobook import mastery


class MasteryTest(unittest.TestCase):
    def wav(self, path, x, sr=24000):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(np.clip(np.rint(x * 32767), -32768, 32767).astype(np.int16).tobytes())

    def test_condition_cli_validates_output(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, txt = map(pathlib.Path, (f"{d}/in.wav", f"{d}/out.wav", f"{d}/t.txt"))
            self.wav(src, np.zeros(24000 * 7, np.float32))
            txt.write_text("sample")
            seen = []
            with mock.patch.object(mastery, "condition", return_value={"out": str(out), "edits": [], "peak": 0, "seconds": 7}), \
                 mock.patch.object(mastery, "validate_reference", side_effect=lambda wav, *_a, **_k: seen.append(wav) or {"verdict": "PASS/PASS", "conditioning": {}}), \
                 contextlib.redirect_stdout(io.StringIO()):
                mastery.main([str(src), str(txt), "--condition", str(out)])
            self.assertEqual(pathlib.Path(seen[0]), out)


    def test_no_quality_reaches_asr_validation(self):
        with tempfile.TemporaryDirectory() as d:
            src, txt = pathlib.Path(d) / "in.wav", pathlib.Path(d) / "t.txt"
            self.wav(src, np.zeros(24000 * 7, np.float32))
            txt.write_text("sample")
            seen = []
            with mock.patch.object(mastery, "validate_reference", side_effect=lambda *_a, **kw: seen.append(kw["quality_fn"]) or {"verdict": "PASS/PASS", "conditioning": {}}), \
                 contextlib.redirect_stdout(io.StringIO()):
                mastery.main([str(src), str(txt), "--no-quality"])
            self.assertEqual(seen, [None])
    def test_score_schema_and_pause_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.wav"
            x = np.zeros(24000 * 7, np.float32)
            x[4800:-7200] = .1 * np.sin(2 * np.pi * 220 * np.arange(len(x[4800:-7200])) / 24000)
            self.wav(p, x)
            with mock.patch.object(mastery, "_integrated_loudness", return_value=-18.0):
                r = mastery.conditioning_report(
                    p, "soft sibilant plosive cadence",
                    quality_fn=lambda _x, _sr: {"sig": 4.0, "bak": 4.0, "ovrl": 4.0})
            self.assertIn("score", r)
            self.assertNotIn("score_0_10", r)
            self.assertIn("interior_pause", r["breakdown"])
            self.assertNotIn("breath", r["breakdown"])
    def test_edge_rms_measures_start_and_end_independently(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.wav"
            sr = 24000
            w = int(sr * mastery.EDGE_FADE_MS / 1000)
            first = f"first {mastery.EDGE_FADE_MS}ms RMS"
            last = f"last {mastery.EDGE_FADE_MS}ms RMS"
            with mock.patch.object(mastery, "_integrated_loudness", return_value=-18.0):
                # Energy only in the first 10ms: only the start edge may fail.
                x = np.zeros(sr * 7, np.float32)
                x[:w] = 0.5
                self.wav(p, x)
                r = mastery.conditioning_report(p, "soft sibilant plosive cadence")
                reasons = " | ".join(r["reasons"])
                self.assertIn(first, reasons)
                self.assertNotIn(last, reasons)
                self.assertNotEqual(r["edges_10ms_rms"][0], r["edges_10ms_rms"][1])
                self.assertAlmostEqual(r["edges_10ms_rms"][0], 0.5, places=4)
                self.assertAlmostEqual(r["edges_10ms_rms"][1], 0.0, places=6)
                # Energy only in the last 10ms: only the end edge may fail.
                x = np.zeros(sr * 7, np.float32)
                x[-w:] = 0.5
                self.wav(p, x)
                r = mastery.conditioning_report(p, "soft sibilant plosive cadence")
                reasons = " | ".join(r["reasons"])
                self.assertNotIn(first, reasons)
                self.assertIn(last, reasons)
                self.assertNotEqual(r["edges_10ms_rms"][0], r["edges_10ms_rms"][1])
                self.assertAlmostEqual(r["edges_10ms_rms"][0], 0.0, places=6)
                self.assertAlmostEqual(r["edges_10ms_rms"][1], 0.5, places=4)

    def test_dnsmos_failure_does_not_override_other_failures(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.wav"
            self.wav(p, np.zeros(24000 * 7, np.float32))
            with mock.patch.object(mastery, "_integrated_loudness", return_value=-18.0):
                r = mastery.conditioning_report(
                    p, "soft sibilant plosive cadence",
                    quality_fn=lambda _x, _sr: {"sig": 3.0, "bak": 4.0, "ovrl": 3.5})
            reasons = " | ".join(r["reasons"])
            self.assertIn("DNSMOS SIG", reasons)
            self.assertIn("no interior pause", reasons)
            self.assertEqual(r["breakdown"]["dnsmos_sig"], 0)

    def test_hard_splice_metric_rejects_step(self):
        x = np.r_[np.full(2400, -.5, np.float32), np.full(2400, .5, np.float32)]
        self.assertFalse(mastery._boundary_continuity(x, 24000, 2400)["clean"])

    def test_dnsmos_repeats_short_audio(self):
        session = mock.Mock()
        inp = mock.Mock()
        inp.name = "audio"
        session.get_inputs.return_value = [inp]
        session.run.side_effect = lambda _out, feed: [np.array([[4.0, 4.0, 4.0]], np.float32)]
        source = np.arange(16000, dtype=np.float32)
        with mock.patch("onnxruntime.InferenceSession", return_value=session), \
             mock.patch.object(mastery, "_dnsmos_model_path", return_value=pathlib.Path("model")):
            result = mastery._dnsmos_p835(source, 16000)
        calls = [call.args[1]["audio"][0] for call in session.run.call_args_list]
        self.assertEqual(result["num_hops"], 7)
        self.assertTrue(all(window.shape == (144160,) for window in calls))
        repeated = np.tile(source, 16)
        for index, window in enumerate(calls):
            np.testing.assert_array_equal(window, repeated[index * 16000:index * 16000 + 144160])

    def test_dnsmos_averages_long_audio_hops(self):
        session = mock.Mock()
        inp = mock.Mock()
        inp.name = "audio"
        session.get_inputs.return_value = [inp]
        session.run.side_effect = [
            [np.array([[3.0, 3.0, 3.0]], np.float32)],
            [np.array([[4.0, 4.0, 4.0]], np.float32)],
            [np.array([[5.0, 5.0, 5.0]], np.float32)],
        ]
        with mock.patch("onnxruntime.InferenceSession", return_value=session), \
             mock.patch.object(mastery, "_dnsmos_model_path", return_value=pathlib.Path("model")):
            result = mastery._dnsmos_p835(np.ones(12 * 16000, np.float32), 16000)
        self.assertEqual(int(np.floor(12) - 9.01) + 1, 3)
        self.assertEqual(result["num_hops"], 3)
        self.assertEqual(session.run.call_count, 3)
        expected_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])(4.0)
        self.assertAlmostEqual(result["sig"], expected_sig)

    def test_dnsmos_nonpersonalized_polynomials_match_upstream(self):
        raw_sig, raw_bak, raw_ovrl = 4.0, 3.5, 3.75
        expected = {
            "sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439])(raw_sig),
            "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546])(raw_bak),
            "ovrl": np.poly1d([-0.06766283, 1.11546468, 0.04602535])(raw_ovrl),
        }
        session = mock.Mock()
        inp = mock.Mock(); inp.name = "input_1"
        session.get_inputs.return_value = [inp]
        session.run.return_value = [np.array([[raw_sig, raw_bak, raw_ovrl]], np.float32)]
        with mock.patch("onnxruntime.InferenceSession", return_value=session), \
             mock.patch.object(mastery, "_dnsmos_model_path", return_value=pathlib.Path("model")):
            actual = mastery._dnsmos_p835(np.ones(10 * 16000, np.float32), 16000)
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value)


if __name__ == "__main__":
    unittest.main()
