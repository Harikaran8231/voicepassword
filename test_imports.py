"""
Run this to check which library is crashing.
python test_imports.py
"""
import sys
print(f"Python: {sys.version}")
print()

libs = [
    ("numpy", "import numpy as np; print(np.__version__)"),
    ("librosa", "import librosa; print(librosa.__version__)"),
    ("soundfile", "import soundfile as sf; print(sf.__version__)"),
    ("scipy", "import scipy; print(scipy.__version__)"),
    ("flask", "import flask; print(flask.__version__)"),
]

for name, code in libs:
    try:
        exec(code)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name} FAILED: {e}")

print()
print("Testing librosa audio load...")
try:
    import numpy as np
    import wave, struct, math, tempfile, os

    # Create test WAV
    tmp = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = [struct.pack('<h', int(0.5 * math.sin(2*math.pi*440*i/16000) * 32767)) for i in range(48000)]
        wf.writeframes(b''.join(samples))

    import librosa
    y, sr = librosa.load(tmp, sr=16000, mono=True)
    os.remove(tmp)
    print(f"  ✓ librosa.load works: {len(y)} samples, {sr}Hz")

    print("\nTesting MFCC extraction...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    print(f"  ✓ MFCC: {mfcc.shape}")

    print("\nTesting pyin (pitch)...")
    f0, vf, vp = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    print(f"  ✓ pyin: {f0.shape}")

    print("\nAll tests passed! librosa is working correctly.")

except Exception as e:
    import traceback
    print(f"  ✗ FAILED: {e}")
    print(traceback.format_exc())