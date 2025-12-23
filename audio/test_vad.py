import soundfile as sf
import numpy as np
from vad import VoiceActivityDetector

AUDIO_FILE = "system_audio_test.wav"

audio, sr = sf.read(AUDIO_FILE)

# Convert stereo → mono
if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

vad = VoiceActivityDetector(
    energy_threshold=0.01,
    min_speech_duration=0.3,
    sample_rate=sr
)

segments = vad.detect_speech_segments(audio)

print(f"🎯 Detected {len(segments)} speech segments")

for i, seg in enumerate(segments):
    duration = len(seg) / sr
    print(f"Segment {i+1}: {duration:.2f} seconds")
