"""Tests for TOM Speech-to-Text (STT) module."""
import time
import unittest
from pathlib import Path
from voice.stt import SpeechToText, transcribe, TranscriptionResult

MIC_TEST_WAV = Path(__file__).resolve().parent / "mic_test.wav"


class TestSTT(unittest.TestCase):
    """Test suite for TOM Speech-to-Text functionality."""

    def setUp(self):
        """Ensure mic_test.wav exists for testing."""
        self.assertTrue(MIC_TEST_WAV.exists(), f"Missing required test fixture: {MIC_TEST_WAV}")

    def test_transcribe_mic_test_wav(self):
        """Test transcribing mic_test.wav and verify return structure and fields."""
        t0 = time.time()
        result = transcribe(MIC_TEST_WAV)
        elapsed = time.time() - t0

        # Verify type and structure
        self.assertIsInstance(result, TranscriptionResult)
        self.assertIsInstance(result, dict)

        # Verify fields
        self.assertIn("text", result)
        self.assertIn("language", result)
        self.assertIn("language_probability", result)

        self.assertIsInstance(result.text, str)
        self.assertIsInstance(result.language, str)
        self.assertIsInstance(result.language_probability, float)

        # Attribute access
        self.assertEqual(result.text, result["text"])
        self.assertEqual(result.language, result["language"])
        self.assertEqual(result.language_probability, result["language_probability"])

        # Tuple unpacking
        text, lang, prob = result
        self.assertEqual(text, result.text)
        self.assertEqual(lang, result.language)
        self.assertEqual(prob, result.language_probability)

        # Probability range
        self.assertGreaterEqual(result.language_probability, 0.0)
        self.assertLessEqual(result.language_probability, 1.0)

        # Performance on i5-1235U for 5s clip should be under 5 seconds
        self.assertLess(elapsed, 5.0)

    def test_reusable_model_instance(self):
        """Verify model weights are loaded once and reused across transcriptions."""
        stt = SpeechToText(model_size="base", device="cpu", compute_type="int8")
        self.assertIsNone(stt._model)

        # First call loads model
        model_ref_1 = stt._ensure_model_loaded()
        self.assertIsNotNone(model_ref_1)

        # Second call reuses same model instance without reloading
        model_ref_2 = stt._ensure_model_loaded()
        self.assertIs(model_ref_1, model_ref_2)

    def test_language_support_hints(self):
        """Verify explicit language codes for English ('en') and Tamil ('ta') are accepted."""
        stt = SpeechToText(model_size="base", device="cpu", compute_type="int8")

        # Test English hint
        res_en = stt.transcribe(MIC_TEST_WAV, language="en")
        self.assertEqual(res_en.language, "en")

        # Test Tamil hint
        res_ta = stt.transcribe(MIC_TEST_WAV, language="ta")
        self.assertEqual(res_ta.language, "ta")

    def test_missing_audio_file(self):
        """Verify FileNotFoundError is raised when the audio path does not exist."""
        stt = SpeechToText()
        with self.assertRaises(FileNotFoundError):
            stt.transcribe("non_existent_audio_file.wav")


if __name__ == "__main__":
    unittest.main()
