import numpy as np


class VoiceActivityDetector:
    def __init__(
        self,
        energy_threshold=0.01,
        min_speech_duration=0.5,
        sample_rate=48000
    ):
        self.energy_threshold = energy_threshold
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.sample_rate = sample_rate

    def rms_energy(self, audio_chunk: np.ndarray) -> float:
        """
        Compute RMS energy of audio chunk
        """
        return np.sqrt(np.mean(np.square(audio_chunk)))

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        energy = self.rms_energy(audio_chunk)
        return energy > self.energy_threshold

    def detect_speech_segments(self, audio: np.ndarray):
        """
        Splits audio into speech segments based on energy
        """
        speech_segments = []
        current_segment = []

        for frame in audio:
            if self.is_speech(frame):
                current_segment.append(frame)
            else:
                if len(current_segment) >= self.min_speech_samples:
                    speech_segments.append(np.array(current_segment))
                current_segment = []

        # Catch last segment
        if len(current_segment) >= self.min_speech_samples:
            speech_segments.append(np.array(current_segment))

        return speech_segments
