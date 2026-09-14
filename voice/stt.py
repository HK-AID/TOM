"""Speech-to-Text (STT) module for TOM using faster-whisper."""
from pathlib import Path
from typing import Any
from faster_whisper import WhisperModel


class TranscriptionResult(dict):
    """Result of a speech-to-text transcription.

    Supports:
    - Dict-style access: result["text"], result["language"], result["language_probability"]
    - Attribute access: result.text, result.language, result.language_probability
    - Tuple unpacking: text, language, prob = result
    """

    def __init__(self, text: str, language: str, language_probability: float) -> None:
        super().__init__(
            text=text,
            language=language,
            language_probability=language_probability,
        )
        self.text = text
        self.language = language
        self.language_probability = language_probability

    def __iter__(self):
        yield self.text
        yield self.language
        yield self.language_probability


class SpeechToText:
    """Reusable Speech-to-Text engine using faster-whisper on CPU with int8 quantization."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None

    def _ensure_model_loaded(self) -> WhisperModel:
        """Load model weights once and cache the instance."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file using the reusable Whisper model with VAD enabled.

        Args:
            audio_path: Path to the WAV audio file.
            language: Optional language code ('en', 'ta', etc.). If None, auto-detects.

        Returns:
            TranscriptionResult containing text, language, and language_probability.
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._ensure_model_loaded()

        # Run transcription with Voice Activity Detection (VAD) filter enabled
        segments, info = model.transcribe(
            str(audio_file),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()

        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
        )


# Global reusable instance to avoid reloading weights across multiple calls
_default_stt: SpeechToText | None = None


def get_stt() -> SpeechToText:
    """Retrieve or initialize the global reusable SpeechToText instance."""
    global _default_stt
    if _default_stt is None:
        _default_stt = SpeechToText()
    return _default_stt


def transcribe(
    audio_path: str | Path,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio using the default reusable SpeechToText instance."""
    return get_stt().transcribe(audio_path, language=language)
