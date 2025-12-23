from faster_whisper import WhisperModel
import numpy as np


class WhisperTranscriber:
    def __init__(
        self,
        model_size="small",
        device="cpu",
        compute_type="int8"
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe_segment(self, audio_segment: np.ndarray, sample_rate: int):
        """
        Transcribe a single speech segment (numpy array)
        """
        segments, info = self.model.transcribe(
            audio_segment,
            language="en",
            beam_size=5
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        return " ".join(text_parts).strip()


