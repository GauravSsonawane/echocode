import numpy as np


class VoiceActivityDetector:
    def __init__(
        self,
        energy_threshold=0.005,
        frame_duration=0.02,      # 20 ms
        min_speech_duration=0.3,
        sample_rate=48000
    ):
        self.energy_threshold = energy_threshold
        self.frame_size = int(frame_duration * sample_rate)
        self.min_speech_frames = int(min_speech_duration / frame_duration)
        self.sample_rate = sample_rate

    def rms_energy(self, frame: np.ndarray) -> float:
        return np.sqrt(np.mean(frame ** 2))

    def detect_speech_segments(self, audio: np.ndarray):
        speech_segments = []
        current_segment = []

        num_frames = len(audio) // self.frame_size

        for i in range(num_frames):
            frame = audio[i * self.frame_size:(i + 1) * self.frame_size]
            energy = self.rms_energy(frame)

            if energy > self.energy_threshold:
                current_segment.append(frame)
            else:
                if len(current_segment) >= self.min_speech_frames:
                    speech_segments.append(
                        np.concatenate(current_segment)
                    )
                current_segment = []

        # Catch last segment
        if len(current_segment) >= self.min_speech_frames:
            speech_segments.append(
                np.concatenate(current_segment)
            )

        return speech_segments
