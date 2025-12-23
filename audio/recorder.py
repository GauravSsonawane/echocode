import soundcard as sc
import numpy as np
import soundfile as sf
import time

SAMPLE_RATE = 48000
RECORD_SECONDS = 10
OUTPUT_FILE = "system_audio_test.wav"


def record_system_audio():
    print("🔍 Searching for Realtek loopback microphone...\n")

    loopback_mic = None
    for mic in sc.all_microphones(include_loopback=True):
        if "realtek" in mic.name.lower():
            loopback_mic = mic
            break

    if loopback_mic is None:
        raise RuntimeError("❌ Realtek loopback microphone not found!")

    print(f"✅ Using loopback device: {loopback_mic.name}")

    print("\n🎧 Recording system audio...")
    print("▶️ Play any audio now (YouTube, music, etc.)")
    time.sleep(2)

    with loopback_mic.recorder(
        samplerate=SAMPLE_RATE,
        channels=2
    ) as recorder:
        data = recorder.record(numframes=SAMPLE_RATE * RECORD_SECONDS)

    print("💾 Saving audio file...")

    if np.max(np.abs(data)) == 0:
        raise RuntimeError("❌ Silence detected. Make sure audio is playing.")

    sf.write(OUTPUT_FILE, data, SAMPLE_RATE)

    print(f"✅ Saved as {OUTPUT_FILE}")


if __name__ == "__main__":
    record_system_audio()

