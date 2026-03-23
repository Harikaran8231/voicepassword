"""
sb_patch.py - Must be imported before any speechbrain import in voice_processing.py.
Patches torchaudio compatibility for SpeechBrain on Python 3.14 + torchaudio 2.x.
"""
import sys
import os
import shutil
import glob

# ── 1. Patch torchaudio.list_audio_backends ───────────────────────────────────
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['ffmpeg', 'soundfile']

# ── 2. Patch torch_audio_backend.py ──────────────────────────────────────────
for path in sys.path:
    f1 = os.path.join(path, 'speechbrain', 'utils', 'torch_audio_backend.py')
    if os.path.exists(f1):
        with open(f1, 'r', encoding='utf-8') as f:
            code = f.read()
        if 'list_audio_backends()' in code and 'getattr(torchaudio' not in code:
            code = code.replace(
                'available_backends = torchaudio.list_audio_backends()',
                'available_backends = getattr(torchaudio, "list_audio_backends", lambda: ["ffmpeg", "soundfile"])()'
            )
            with open(f1, 'w', encoding='utf-8') as f:
                f.write(code)
        break

# ── 3. Copy cached model files to savedir ─────────────────────────────────────
savedir = os.path.join(os.getcwd(), "pretrained_models", "spkrec-ecapa-voxceleb")
os.makedirs(savedir, exist_ok=True)

cache_base = os.path.expanduser(
    r"~\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb\snapshots"
)
if os.path.exists(cache_base):
    snapshots = glob.glob(os.path.join(cache_base, "*"))
    if snapshots:
        snapshot_dir = max(snapshots, key=os.path.getmtime)
        for file in os.listdir(snapshot_dir):
            src = os.path.join(snapshot_dir, file)
            dst = os.path.join(savedir, file)
            if not os.path.exists(dst):
                try:
                    real_src = os.path.realpath(src)
                    shutil.copy2(real_src, dst)
                except Exception:
                    pass

# ── 4. Register fake modules ──────────────────────────────────────────────────
import types
for mod_name in ['torchaudio.utils.ffmpeg_utils', 'torchaudio.backend.utils']:
    if mod_name not in sys.modules:
        fake = types.ModuleType(mod_name)
        fake.get_audio_decoders = lambda: {}
        fake.get_audio_encoders = lambda: {}
        sys.modules[mod_name] = fake