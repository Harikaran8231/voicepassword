"""
redownload_ckpt.py — Force re-download of corrupted .ckpt files.

The HuggingFace cache stores files as blobs with symlinks pointing to them.
Copying the symlink can result in a truncated file. This script deletes the
local copies and re-downloads them directly using requests (no symlinks involved).

Run: python redownload_ckpt.py
"""

import os
import sys
import shutil

SAVEDIR = os.path.join(os.getcwd(), "pretrained_models", "spkrec-ecapa-voxceleb")
os.makedirs(SAVEDIR, exist_ok=True)

BASE_URL = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/"

FILES = [
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "hyperparams.yaml",
    "label_encoder.txt",
]

print("=" * 60)
print("VoiceKey — Force Re-download Model Files")
print("=" * 60)
print(f"Saving to: {SAVEDIR}\n")

try:
    import requests
except ImportError:
    print("requests not installed — installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

failed = []
for fname in FILES:
    dest = os.path.join(SAVEDIR, fname)

    # Delete existing (possibly truncated) copy
    if os.path.exists(dest):
        os.remove(dest)
        print(f"  ✗ Deleted old:  {fname}")

    url = BASE_URL + fname
    print(f"  ↓ Downloading:  {fname} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size = os.path.getsize(dest)
        print(f"✓  ({size:,} bytes)")
        if size < 100:
            print(f"    ✗ WARNING: {fname} is suspiciously small ({size} bytes) — may have failed")
            failed.append(fname)
    except Exception as e:
        print(f"✗  FAILED: {e}")
        failed.append(fname)

# Create empty custom.py
custom_py = os.path.join(SAVEDIR, "custom.py")
open(custom_py, "w").close()
print(f"\n  ✓ Created empty custom.py")

print("\n" + "=" * 60)
if failed:
    print(f"✗ Failed: {failed}")
    print("\n  Check your internet connection and retry.")
    sys.exit(1)
else:
    print("✓ All files downloaded successfully!\n")
    print("  Files:")
    for f in sorted(os.listdir(SAVEDIR)):
        fpath = os.path.join(SAVEDIR, f)
        print(f"    {f:40s}  {os.path.getsize(fpath):>12,} bytes")
    print("\n  Now run:")
    print("    python test_speechbrain.py")
    print("=" * 60)