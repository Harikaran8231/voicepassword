"""
fix_speechbrain.py - One-time fix for all SpeechBrain symlink issues on Windows.
Run ONCE: python fix_speechbrain.py
Then run: python test_speechbrain.py
"""
import sys, os

print("Fixing all SpeechBrain files...\n")

sb_base = None
for path in sys.path:
    candidate = os.path.join(path, 'speechbrain')
    if os.path.isdir(candidate):
        sb_base = candidate
        print(f"Found SpeechBrain at: {sb_base}\n")
        break

if not sb_base:
    print("SpeechBrain not found!")
    sys.exit(1)

# Walk ALL Python files in speechbrain and replace every SYMLINK reference
fixed_files = []
for root, dirs, files in os.walk(sb_base):
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                original = f.read()
        except Exception:
            continue

        code = original

        # Replace all SYMLINK defaults and usages with COPY
        replacements = [
            ('LocalStrategy.SYMLINK', 'LocalStrategy.COPY'),
            ('"symlink"', '"copy"'),
            ("'symlink'", "'copy'"),
            ('strategy=LocalStrategy.SYMLINK', 'strategy=LocalStrategy.COPY'),
            ('local_strategy: LocalStrategy = LocalStrategy.COPY', 'local_strategy: LocalStrategy = LocalStrategy.COPY'),  # idempotent
        ]

        for old, new in replacements:
            code = code.replace(old, new)

        # Also patch torchaudio.list_audio_backends
        code = code.replace(
            'available_backends = torchaudio.list_audio_backends()',
            'available_backends = getattr(torchaudio, "list_audio_backends", lambda: ["ffmpeg", "soundfile"])()'
        )

        if code != original:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(code)
            rel = os.path.relpath(fpath, sb_base)
            fixed_files.append(rel)
            print(f"  ✓ Fixed: {rel}")

print(f"\nFixed {len(fixed_files)} files.")
print("\nNow run: python test_speechbrain.py")