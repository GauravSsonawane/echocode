import soundfile as sf
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.vad import VoiceActivityDetector
from stt.whisper_engine import WhisperTranscriber

AUDIO_FILE = "system_audio_test.wav"

# Load audio
audio, sr = sf.read(AUDIO_FILE)

# Convert stereo → mono
if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

# VAD (reuse Day 2)
vad = VoiceActivityDetector(
    energy_threshold=0.005,
    frame_duration=0.02,
    min_speech_duration=0.3,
    sample_rate=sr
)

speech_segments = vad.detect_speech_segments(audio)

print(f"🎯 Found {len(speech_segments)} speech segments")

# Whisper
transcriber = WhisperTranscriber(
    model_size="small",   # fast + good quality
    device="cpu",
    compute_type="int8"
)

print("\n🧠 Transcriptions:\n")

for i, segment in enumerate(speech_segments):
    text = transcriber.transcribe_segment(segment, sr)
    print(f"[Segment {i+1}] {text}")

    
