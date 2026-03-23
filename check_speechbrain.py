"""
Run this script to diagnose why SpeechBrain isn't loading.
Usage: python check_speechbrain.py
"""

print("=" * 60)
print("SpeechBrain Diagnostic Check")
print("=" * 60)

# Step 1: Check imports
print("\n[1] Checking imports...")
try:
    import torch
    print(f"    ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"    ✗ torch NOT installed: {e}")
    print("      Fix: pip install torch")

try:
    import torchaudio
    print(f"    ✓ torchaudio {torchaudio.__version__}")
except ImportError as e:
    print(f"    ✗ torchaudio NOT installed: {e}")
    print("      Fix: pip install torchaudio")

try:
    import speechbrain
    print(f"    ✓ speechbrain {speechbrain.__version__}")
except ImportError as e:
    print(f"    ✗ speechbrain NOT installed: {e}")
    print("      Fix: pip install speechbrain")

# Step 2: Try loading the model
print("\n[2] Attempting to load ECAPA-TDNN model...")
try:
    from speechbrain.pretrained import EncoderClassifier
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )
    print("    ✓ ECAPA-TDNN model loaded successfully!")
    print("    ✓ SpeechBrain is READY — your app will use it automatically.")
except Exception as e:
    print(f"    ✗ Model loading failed: {e}")
    print("\n    Possible fixes:")
    print("    - Check internet connection (model downloads ~100MB on first run)")
    print("    - Try: pip install --upgrade speechbrain")

print("\n" + "=" * 60)