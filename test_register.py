"""
test_register.py - Test registration directly without browser.
Run this while app.py is running in another terminal:
    python test_register.py
"""
import urllib.request
import urllib.error
import json
import os
import wave
import struct
import math

def generate_sine_wav(filename, duration=3, freq=440, sample_rate=16000):
    """Generate a real sine wave WAV file to simulate voice."""
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_samples = int(duration * sample_rate)
        samples = []
        for i in range(n_samples):
            # Mix multiple frequencies to simulate voice-like audio
            val = (
                0.5 * math.sin(2 * math.pi * freq * i / sample_rate) +
                0.3 * math.sin(2 * math.pi * freq * 2 * i / sample_rate) +
                0.2 * math.sin(2 * math.pi * freq * 3 * i / sample_rate)
            )
            samples.append(struct.pack('<h', int(val * 32767)))
        wf.writeframes(b''.join(samples))
    print(f"Generated test audio: {filename} ({os.path.getsize(filename)} bytes)")

def test_register(username="testuser_debug"):
    print("=" * 55)
    print("Direct Server Registration Test")
    print("=" * 55)

    # Generate test WAV
    wav_file = "test_audio.wav"
    generate_sine_wav(wav_file)

    # Read WAV bytes
    with open(wav_file, 'rb') as f:
        audio_data = f.read()
    os.remove(wav_file)

    # Build multipart form manually
    boundary = "----TestBoundary1234567890"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="username"\r\n\r\n'
        f"{username}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="voice.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/register",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    print(f"\nSending {len(audio_data):,} bytes of WAV audio...")
    print(f"Username: {username}")

    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            response = json.loads(res.read().decode())
            print(f"\n✓ Status: {res.status}")
            print(f"✓ Response: {response}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"\n✗ HTTP Error {e.code}: {body}")
    except urllib.error.URLError as e:
        print(f"\n✗ Connection Error: {e}")
        print("  → Make sure 'python app.py' is running")
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")

    print("=" * 55)

if __name__ == "__main__":
    test_register()