"""
test_speechbrain.py - Load ECAPA-TDNN via raw torch weights, bypassing from_hparams.

hyperpyyaml (used by from_hparams) is broken on Python 3.14 — AST changes
cause 'str' object has no attribute 'keys'. We don't need the YAML at all:
the .ckpt files are already local, so we load them directly with torch.load
and manually instantiate the ECAPA_TDNN model class.

Run: python test_speechbrain.py
"""

import sys, os, glob, shutil

print(f"Python: {sys.version}\n")

# ── Step 1: Fix fetching.py (SYMLINK -> COPY) ─────────────────────────────────
print("[1] Fixing fetching.py...")
for path in sys.path:
    fp = os.path.join(path, 'speechbrain', 'utils', 'fetching.py')
    if os.path.exists(fp):
        with open(fp, 'r', encoding='utf-8') as f:
            code = f.read()
        code = code.replace(
            'local_strategy: LocalStrategy = LocalStrategy.SYMLINK',
            'local_strategy: LocalStrategy = LocalStrategy.COPY'
        )
        code = code.replace("local_strategy='symlink'", "local_strategy='copy'")
        code = code.replace('local_strategy="symlink"', 'local_strategy="copy"')
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"    ✓ Fixed: {fp}")
        break

# ── Step 2: Patch torchaudio ──────────────────────────────────────────────────
print("\n[2] Patching torchaudio...")
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['ffmpeg', 'soundfile']
    print("    ✓ Patched list_audio_backends")
else:
    print("    ✓ Already patched")

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
            print(f"    ✓ Patched torch_audio_backend.py")
        else:
            print(f"    ✓ torch_audio_backend.py already patched")
        break

# ── Step 3: Locate and copy cached .ckpt files to savedir ────────────────────
print("\n[3] Copying cached model files...")
savedir = os.path.join(os.getcwd(), "pretrained_models", "spkrec-ecapa-voxceleb")
os.makedirs(savedir, exist_ok=True)

cache_base = os.path.expanduser(
    r"~\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb\snapshots"
)

copied = 0
if os.path.exists(cache_base):
    snapshots = sorted(glob.glob(os.path.join(cache_base, "*")), key=os.path.getmtime, reverse=True)
    if snapshots:
        snapshot_dir = snapshots[0]
        print(f"    Using snapshot: {snapshot_dir}")
        for fname in os.listdir(snapshot_dir):
            src = os.path.join(snapshot_dir, fname)
            dst = os.path.join(savedir, fname)
            try:
                real_src = os.path.realpath(src)  # resolve symlinks
                if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                    shutil.copy2(real_src, dst)
                    print(f"    ✓ Copied:  {fname}  ({os.path.getsize(dst):,} bytes)")
                    copied += 1
                else:
                    print(f"    ✓ Exists:  {fname}  ({os.path.getsize(dst):,} bytes)")
            except Exception as e:
                print(f"    ✗ Failed:  {fname} — {e}")
    else:
        print("    No snapshots found in cache")
else:
    print(f"    Cache not found at: {cache_base}")

# Always create empty custom.py so SpeechBrain never fetches it from HuggingFace
custom_py = os.path.join(savedir, "custom.py")
if not os.path.exists(custom_py):
    open(custom_py, "w").close()
    print(f"    ✓ Created empty custom.py")

print(f"\n    Files in savedir ({savedir}):")
for f in sorted(os.listdir(savedir)):
    fpath = os.path.join(savedir, f)
    print(f"      {f:40s}  {os.path.getsize(fpath):>12,} bytes")

# ── Step 4: Verify required .ckpt files exist ─────────────────────────────────
print("\n[4] Checking required weight files...")
required = {
    "embedding_model.ckpt":   os.path.join(savedir, "embedding_model.ckpt"),
    "mean_var_norm_emb.ckpt": os.path.join(savedir, "mean_var_norm_emb.ckpt"),
}
for name, fpath in required.items():
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"    ✓ {name}  ({os.path.getsize(fpath):,} bytes)")
    else:
        print(f"    ✗ MISSING: {name}")
        print("      Run: python download_model.py")
        sys.exit(1)

# ── Step 5: Load via raw torch weights (bypasses from_hparams + hyperpyyaml) ──
print("\n[5] Loading ECAPA-TDNN via raw torch weights...")
print("    (skipping from_hparams — hyperpyyaml broken on Python 3.14)")
try:
    import torch
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization

    emb_path  = required["embedding_model.ckpt"]
    norm_path = required["mean_var_norm_emb.ckpt"]

    print("    Building ECAPA_TDNN architecture...")
    ecapa_model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    )

    print("    Loading embedding_model.ckpt weights...")
    state = torch.load(emb_path, map_location="cpu", weights_only=False)
    ecapa_model.load_state_dict(state)
    ecapa_model.eval()
    print("    ✓ Weights loaded")

    print("    Loading mean_var_norm_emb.ckpt normalizer...")
    normalizer = InputNormalization(norm_type="sentence", std_norm=False)
    norm_state = torch.load(norm_path, map_location="cpu", weights_only=False)
    missing, unexpected = normalizer.load_state_dict(norm_state, strict=False)
    for key in unexpected:
        if key in norm_state:
            setattr(normalizer, key, norm_state[key])
    print(f"    ✓ Normalizer loaded (set {len(unexpected)} extra attrs)")

except Exception as e:
    import traceback
    print(f"    ✗ Failed: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# ── Step 6: Test embedding extraction ─────────────────────────────────────────
print("\n[6] Testing embedding extraction...")
try:
    import torchaudio.transforms as T

    # 5 seconds of dummy audio at 16kHz
    dummy = torch.randn(1, 80000)

    fbank = T.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=400,
        hop_length=160, n_mels=80, f_min=20, f_max=7600,
    )(dummy)
    fbank = torch.log(fbank + 1e-6).squeeze(0).T.unsqueeze(0)  # (1, T, 80)

    with torch.no_grad():
        lens    = torch.tensor([1.0])
        fbank_n = normalizer(fbank, lens)
        emb     = ecapa_model(fbank_n, lens)

    result = emb.squeeze().cpu().numpy().flatten()
    print(f"    ✓ Embedding shape: {result.shape}")

    if result.shape[0] == 192:
        print("\n" + "=" * 60)
        print("  ECAPA-TDNN FULLY WORKING via raw weights!")
        print("  encoder_type = ecapa_raw")
        print("  dim = 192")
        print("\n  Start the app:")
        print("    python app.py")
        print("=" * 60)
    else:
        print(f"    ✗ Unexpected dim {result.shape[0]} — expected 192")
        sys.exit(1)

except Exception as e:
    import traceback
    print(f"    ✗ {e}")
    print(traceback.format_exc())
    sys.exit(1)