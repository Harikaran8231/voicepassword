"""
voice_processing.py - Voice Embedding Extraction Module

ECAPA-TDNN only for all authentication flows.
No librosa fallback in the auth pipeline.

Key fixes vs previous version:
  1. `import sb` (was wrongly `import sb_patch` — patches never applied)
  2. _ensure_model_files() downloads each file individually via hf_hub_download
     so custom.py 404 can never crash the load (it's skipped and created empty locally)
  3. _load_ecapa() always loads from local savedir after ensuring files exist —
     from_hparams never points at HuggingFace repo ID, eliminating the 404 entirely
  4. Step 3 raw-weights path bypasses from_hparams + yaml entirely for Python 3.14
  5. No librosa auth fallback — raises RuntimeError loudly on ECAPA failure
  6. extract_speaker_embedding_librosa() kept as a utility for call analysis / testing
"""

import os
import shutil
import numpy as np
import tempfile
import logging
import subprocess

logger = logging.getLogger(__name__)

# ── Fix #1: correct module name is sb, not sb_patch ──────────────────────────
try:
    import sb  # noqa: F401
    print("[voice_processing] sb patches loaded OK")
except ModuleNotFoundError:
    pass
except Exception as _sb_err:
    print(f"[voice_processing] sb patch FAILED: {_sb_err}")

# ── Model state ───────────────────────────────────────────────────────────────
_encoder      = None
_encoder_type = None   # "ecapa" | "ecapa_raw"

# ── VAD thresholds ────────────────────────────────────────────────────────────
MIN_VOICE_ENERGY_DB  = -55.0
MIN_AUDIO_DURATION_S =  1.0

# ── Model save directory ──────────────────────────────────────────────────────
_SAVEDIR = os.path.join(os.getcwd(), "pretrained_models", "spkrec-ecapa-voxceleb")


# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_audio(y: np.ndarray, sr: int, denoise: bool = True, mode: str = "ecapa_api") -> np.ndarray:
    """
    Preprocessing pipeline applied before embedding extraction.

    mode="ecapa_api"  -- SpeechBrain encode_batch path.
                         encode_batch computes FBank + normalisation internally.
                         Only denoise + trim here -- do NOT amplitude-normalise,
                         that shifts the signal off the model's training distribution.

    mode="ecapa_raw"  -- Manual FBank pipeline.
                         Denoise + trim + amplitude normalisation so FBank inputs
                         are clean. Pre-emphasis still skipped (MelSpectrogram
                         handles spectral shaping via the filterbank).

    mode="librosa"    -- MFCC utility path (not used in auth).

    Noise profile: uses the quietest 0.3s window in the recording rather than
    the first 0.5s, so speech at the very start is not used as the noise template.
    """
    import librosa

    # Step 1: Noise reduction
    if denoise:
        try:
            import noisereduce as nr
            window = int(sr * 0.3)
            if len(y) >= window * 2:
                rms_frames = np.array([
                    np.sqrt(np.mean(y[i:i + window] ** 2))
                    for i in range(0, len(y) - window, window // 2)
                ])
                quietest_idx = int(np.argmin(rms_frames)) * (window // 2)
                noise_sample = y[quietest_idx: quietest_idx + window]
            else:
                noise_sample = y[: sr // 2] if len(y) > sr // 2 else y

            y = nr.reduce_noise(
                y=y, sr=sr, y_noise=noise_sample,
                prop_decrease=0.75, stationary=True,
            ).astype(np.float32)
            logger.debug("[Preprocess] Noise reduction applied")
        except ImportError:
            logger.debug("[Preprocess] noisereduce not installed -- skipping")
        except Exception as e:
            logger.warning(f"[Preprocess] Noise reduction skipped: {e}")

    # Step 2: Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) >= sr * 0.5:
        y = y_trimmed

    # Step 3: Amplitude normalisation -- NOT for ecapa_api (encode_batch does its own)
    if mode in ("ecapa_raw", "librosa"):
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.95

    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model file management
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_model_files():
    """
    Download each model file individually via hf_hub_download.

    Why not let from_hparams download?
    from_hparams fetches every file listed in hyperparams.yaml, including
    custom.py which does NOT exist in the HuggingFace repo -- this causes a 404
    that crashes the entire load. hf_hub_download fetches one file at a time so
    we simply never request custom.py, and create it empty locally instead.

    Files land directly in _SAVEDIR so Method 1 (local load) always succeeds
    on subsequent app restarts with no network access required.
    """
    from huggingface_hub import hf_hub_download

    REPO_FILES = [
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        "label_encoder.txt",
    ]

    for fname in REPO_FILES:
        dest = os.path.join(_SAVEDIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info(f"[ECAPA] Already cached: {fname}")
            continue
        try:
            path = hf_hub_download(
                repo_id="speechbrain/spkrec-ecapa-voxceleb",
                filename=fname,
                local_dir=_SAVEDIR,
            )
            # hf_hub_download may nest the file in a cache subdir -- copy to root
            if os.path.abspath(path) != os.path.abspath(dest) and os.path.exists(path):
                shutil.copy2(path, dest)
            logger.info(f"[ECAPA] Downloaded: {fname}")
        except Exception as e:
            logger.warning(f"[ECAPA] Could not download {fname}: {e}")

    # custom.py does NOT exist in the repo (causes 404).
    # Create it empty locally so SpeechBrain's fetcher short-circuits on it.
    custom_py = os.path.join(_SAVEDIR, "custom.py")
    if not os.path.exists(custom_py):
        open(custom_py, "w").close()
        logger.info("[ECAPA] Created empty custom.py (prevents HuggingFace 404)")


# ─────────────────────────────────────────────────────────────────────────────
# ECAPA-TDNN loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_ecapa():
    """
    Load ECAPA-TDNN. Raises RuntimeError if every method fails.

    Strategy -- never let SpeechBrain's fetcher hit HuggingFace:

      Step 1  Download all files individually (_ensure_model_files).
              custom.py is created empty locally so it is never fetched.

      Step 2  Load via SpeechBrain API with source=_SAVEDIR (local directory).
              Because source is a path (not a repo ID), from_hparams reads files
              directly from disk with zero network calls.

      Step 3  If SpeechBrain API still fails (Python 3.14 import / YAML errors),
              load raw torch weights directly. No from_hparams at all --
              just torch.load + manually instantiated ECAPA_TDNN.
    """
    global _encoder, _encoder_type

    if _encoder is not None:
        return  # already loaded -- idempotent

    os.makedirs(_SAVEDIR, exist_ok=True)

    # Patch torchaudio BEFORE any speechbrain import
    try:
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["ffmpeg", "soundfile"]
            logger.info("[ECAPA] Patched torchaudio.list_audio_backends")
    except ImportError:
        pass

    # ── Step 1: Ensure files are local (safe individual downloads) ────────────
    try:
        _ensure_model_files()
    except Exception as e:
        logger.warning(f"[ECAPA] File download step failed ({e}) -- will try cached files")

    has_required = (
        os.path.exists(os.path.join(_SAVEDIR, "embedding_model.ckpt")) and
        os.path.exists(os.path.join(_SAVEDIR, "hyperparams.yaml"))
    )

    # ── Step 2: SpeechBrain API from local savedir -- zero network calls ──────
    if has_required:
        for api_path in ["speechbrain.inference.speaker", "speechbrain.pretrained"]:
            try:
                import importlib
                mod = importlib.import_module(api_path)
                EncoderClassifier = mod.EncoderClassifier
                _encoder = EncoderClassifier.from_hparams(
                    source=_SAVEDIR,    # local directory -- never hits HuggingFace
                    savedir=_SAVEDIR,
                    run_opts={"device": "cpu"},
                )
                _encoder_type = "ecapa"
                logger.info(f"[ECAPA] Loaded from local savedir via {api_path} ✓")
                return
            except Exception as e:
                logger.warning(f"[ECAPA] {api_path} local load failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
    else:
        logger.warning("[ECAPA] Required files missing after download step")

    # ── Step 3: Raw torch weights -- no from_hparams, no YAML, no fetcher ────
    # Works even when SpeechBrain YAML/fetching is broken (Python 3.14 etc.)
    try:
        import torch
        logger.info("[ECAPA] Trying raw weight loading (bypasses from_hparams)...")

        emb_path  = os.path.join(_SAVEDIR, "embedding_model.ckpt")
        norm_path = os.path.join(_SAVEDIR, "mean_var_norm_emb.ckpt")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"embedding_model.ckpt not in {_SAVEDIR}")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"mean_var_norm_emb.ckpt not in {_SAVEDIR}")

        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        ecapa_model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=192,
        )
        state = torch.load(emb_path, map_location="cpu", weights_only=False)
        ecapa_model.load_state_dict(state)
        ecapa_model.eval()

        normalizer = InputNormalization(norm_type="sentence", std_norm=False)
        norm_state = torch.load(norm_path, map_location="cpu", weights_only=False)
        missing, unexpected = normalizer.load_state_dict(norm_state, strict=False)
        for key in unexpected:
            if key in norm_state:
                setattr(normalizer, key, norm_state[key])

        _encoder      = {"model": ecapa_model, "normalizer": normalizer}
        _encoder_type = "ecapa_raw"
        logger.info("[ECAPA] Loaded via raw torch weights (ecapa_raw) ✓")
        return

    except Exception as e:
        logger.warning(f"[ECAPA] Raw weight loading failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # ── All steps failed ──────────────────────────────────────────────────────
    raise RuntimeError(
        "\n\n"
        "==========================================================\n"
        "  ECAPA-TDNN FAILED TO LOAD\n"
        "==========================================================\n"
        "  Run this first to download model files:\n"
        "    python test_speechbrain.py\n"
        "\n"
        "  If that also fails, switch to Python 3.11:\n"
        "    py -3.11 -m venv venv311\n"
        "    venv311\\Scripts\\activate\n"
        "    pip install -r requirements.txt speechbrain torch torchaudio\n"
        "    python test_speechbrain.py\n"
        "    python app.py\n"
        "==========================================================\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio format conversion
# ─────────────────────────────────────────────────────────────────────────────

def _convert_to_wav(input_path: str) -> str:
    """Convert any audio format to 16kHz mono WAV. Tries ffmpeg then soundfile."""
    wav_path = input_path + "_conv.wav"

    try:
        cmd = ["ffmpeg", "-y", "-i", input_path,
               "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", wav_path]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            logger.info(f"[ffmpeg] Converted: {os.path.getsize(wav_path)} bytes")
            return wav_path
        logger.warning(f"[ffmpeg] Failed: {result.stderr.decode('utf-8', errors='ignore')[-200:]}")
    except FileNotFoundError:
        logger.warning("[ffmpeg] Not found in PATH")
    except Exception as e:
        logger.warning(f"[ffmpeg] Error: {e}")

    try:
        import soundfile as sf
        data, sr = sf.read(input_path)
        sf.write(wav_path, data, sr, subtype="PCM_16")
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            logger.info(f"[soundfile] Converted: {os.path.getsize(wav_path)} bytes")
            return wav_path
    except Exception as e:
        logger.warning(f"[soundfile] Failed: {e}")

    return input_path  # return original if all conversion failed


# ─────────────────────────────────────────────────────────────────────────────
# Voice Activity Detection
# ─────────────────────────────────────────────────────────────────────────────

def check_voice_activity(y: np.ndarray, sr: int) -> dict:
    """
    VAD on pre-loaded audio array.
    Returns dict with has_voice (bool) and reason (str).
    Run before preprocessing so energy levels are unaffected by normalisation.
    """
    import librosa

    duration = len(y) / sr
    logger.info(f"[VAD] duration={duration:.2f}s samples={len(y)}")

    if duration < MIN_AUDIO_DURATION_S:
        return {
            "has_voice": False,
            "reason": f"Recording too short ({duration:.1f}s). Please speak for at least 1 second.",
        }

    rms            = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db         = librosa.amplitude_to_db(rms + 1e-9, ref=1.0)
    mean_energy_db = float(np.mean(rms_db))
    logger.info(f"[VAD] mean_energy_db={mean_energy_db:.1f}dB")

    if mean_energy_db < MIN_VOICE_ENERGY_DB:
        return {
            "has_voice": False,
            "reason": "No voice detected -- recording appears silent. Please speak clearly.",
        }

    noise_floor  = np.percentile(rms, 30)
    active_ratio = float(np.mean(rms > noise_floor * 1.5))
    logger.info(f"[VAD] active_ratio={active_ratio:.2f}")

    if active_ratio < 0.08:
        return {
            "has_voice": False,
            "reason": "Insufficient voice detected. Please speak louder or closer to the microphone.",
        }

    logger.info(f"[VAD] PASSED -- energy={mean_energy_db:.1f}dB active={active_ratio:.2f}")
    return {
        "has_voice":    True,
        "reason":       "Voice detected.",
        "energy_db":    mean_energy_db,
        "active_ratio": active_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ECAPA-TDNN embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ecapa_embedding(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract 192-dim ECAPA-TDNN embedding from preprocessed audio.

    ecapa     -- uses SpeechBrain EncoderClassifier.encode_batch()
    ecapa_raw -- uses manually instantiated ECAPA_TDNN + torchaudio MelSpectrogram
    """
    import torch

    if _encoder_type == "ecapa":
        y_proc = y.copy()
        min_len = sr * 5   # pad to 5s minimum for stable embeddings
        if len(y_proc) < min_len:
            y_proc = np.pad(y_proc, (0, min_len - len(y_proc)), mode="constant")

        signal = torch.tensor(y_proc.astype(np.float32)).unsqueeze(0)  # (1, samples)
        with torch.no_grad():
            embedding = _encoder.encode_batch(signal)
        result = embedding.squeeze().cpu().numpy().flatten()
        logger.info(f"[ECAPA] encode_batch embedding: {result.shape}")
        return result

    elif _encoder_type == "ecapa_raw":
        import torchaudio.transforms as T

        y_proc = y.copy()
        min_len = sr * 5
        if len(y_proc) < min_len:
            y_proc = np.pad(y_proc, (0, min_len - len(y_proc)), mode="constant")

        audio_tensor = torch.tensor(y_proc, dtype=torch.float32).unsqueeze(0)

        # 80-dim log mel filterbank -- matches ECAPA training configuration
        fbank = T.MelSpectrogram(
            sample_rate=sr, n_fft=512, win_length=400,
            hop_length=160, n_mels=80, f_min=20, f_max=7600,
        )(audio_tensor)
        fbank = torch.log(fbank + 1e-6).squeeze(0).T.unsqueeze(0)  # (1, T, 80)

        model      = _encoder["model"]
        normalizer = _encoder["normalizer"]
        with torch.no_grad():
            lens      = torch.tensor([1.0])
            fbank_n   = normalizer(fbank, lens)
            embedding = model(fbank_n, lens)

        result = embedding.squeeze().cpu().numpy().flatten()
        logger.info(f"[ECAPA] raw weights embedding: {result.shape}")
        return result

    raise RuntimeError(f"Unexpected encoder type: {_encoder_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Librosa embedding (utility only -- NOT used in auth pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def extract_speaker_embedding_librosa(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Speaker embedding using librosa features.
    NOT used in authentication -- ECAPA-TDNN handles all auth flows.
    Kept as a utility for call analysis, offline testing, or research comparisons.

    Feature breakdown (460-dim total):
      - 60 MFCCs x mean+std           = 120-dim  (vocal tract coarse + fine detail)
      - MFCC delta x mean+std         = 120-dim  (spectral velocity)
      - MFCC delta2 x mean+std        = 120-dim  (spectral acceleration)
      - spectral centroid mean+std    =   2-dim  (brightness)
      - spectral rolloff mean+std     =   2-dim  (energy distribution)
      - spectral contrast mean+std    =  14-dim  (formant clarity, n_bands=6)
      - zero crossing rate mean+std   =   2-dim  (voicing / fricative character)
      - 40-mel log-mel mean+std       =  80-dim  (spectral texture)

    n_mfcc=60 (raised from 40): coefficients 13-60 carry speaker-discriminative
    formant information that the first 12 miss. Only session-stable features kept.

    Returns raw (unnormalized) float32 vector.
    Pass through normalize_embedding() before cosine comparison.
    """
    import librosa

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    def agg2d(f): return np.concatenate([np.mean(f, axis=1), np.std(f, axis=1)])
    def agg1d(f): return np.array([float(np.mean(f)), float(np.std(f))])

    features = np.concatenate([
        agg2d(mfcc),                                                       # 120-dim
        agg2d(mfcc_delta),                                                 # 120-dim
        agg2d(mfcc_delta2),                                                # 120-dim
        agg1d(librosa.feature.spectral_centroid(y=y, sr=sr)),              #   2-dim
        agg1d(librosa.feature.spectral_rolloff(y=y, sr=sr)),               #   2-dim
        agg2d(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)),   #  14-dim
        agg1d(librosa.feature.zero_crossing_rate(y)),                      #   2-dim
        agg2d(librosa.power_to_db(                                         #  80-dim
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000),
            ref=np.max)),
    ])  # total: 460-dim

    logger.info(f"[librosa] Embedding: {features.shape} (utility -- not used in auth)")
    return features.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_embedding(audio_path: str) -> np.ndarray:
    """
    Full pipeline: convert -> load -> VAD -> preprocess -> ECAPA embed -> delete audio.

    ECAPA-TDNN only. No librosa fallback.
    Raises ValueError  for VAD failures (bad audio, too short, silent).
    Raises RuntimeError for all other failures (ECAPA not loaded, decode error).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    logger.info(f"[Extract] Starting: {audio_path} ({os.path.getsize(audio_path):,} bytes)")
    converted_path = None

    try:
        # Step 1: Convert to 16kHz mono WAV
        converted_path = _convert_to_wav(audio_path)
        target = (
            converted_path
            if converted_path != audio_path
            and os.path.exists(converted_path)
            and os.path.getsize(converted_path) > 0
            else audio_path
        )
        logger.info(f"[Extract] Target: {target} ({os.path.getsize(target):,} bytes)")

        # Step 2: Load audio
        import librosa
        try:
            y, sr = librosa.load(target, sr=16000, mono=True, res_type="kaiser_fast")
            logger.info(f"[Extract] Loaded: {len(y)} samples, duration={len(y)/sr:.2f}s")
        except Exception as e:
            raise RuntimeError(f"Failed to decode audio: {e}")

        if len(y) == 0:
            raise ValueError("VAD_FAIL: Audio is empty after loading.")

        # Step 3: VAD -- run before preprocessing so energy levels are unaffected
        vad = check_voice_activity(y, sr)
        if not vad["has_voice"]:
            raise ValueError(f"VAD_FAIL: {vad['reason']}")

        # Step 4: Load ECAPA (raises RuntimeError with clear instructions if all fail)
        _load_ecapa()

        # Step 5: Preprocess -- mode depends on which encoder path is active
        mode = "ecapa_api" if _encoder_type == "ecapa" else "ecapa_raw"
        y = preprocess_audio(y, sr, denoise=True, mode=mode)
        logger.info(f"[Extract] Preprocessed (mode={mode}): {len(y)} samples")

        # Step 6: Extract ECAPA-TDNN embedding
        embedding = _extract_ecapa_embedding(y, sr)
        logger.info(f"[Extract] ECAPA embedding: shape={embedding.shape}")

        # Sanity check -- ECAPA-TDNN always produces 192-dim vectors
        if embedding.shape[0] != 192:
            raise RuntimeError(
                f"Unexpected embedding dim {embedding.shape[0]} -- expected 192. "
                "Model may have loaded incorrectly."
            )

        return embedding

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        import traceback
        logger.error(f"[Extract] Pipeline failed:\n{traceback.format_exc()}")
        raise RuntimeError(f"Embedding extraction failed: {e}")

    finally:
        for path in set(filter(None, [audio_path, converted_path])):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"[Extract] Deleted: {path}")
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_audio_temp(audio_data: bytes, suffix: str = ".webm") -> str:
    """
    Save raw audio bytes to a temp file in the uploads/ directory.
    Auto-detects format from magic bytes and adjusts the extension.
    """
    os.makedirs("uploads", exist_ok=True)
    if len(audio_data) >= 4:
        if audio_data[:4] == b"OggS":
            suffix = ".ogg"
        elif len(audio_data) >= 8 and audio_data[4:8] == b"ftyp":
            suffix = ".mp4"
        elif audio_data[:4] in (b"RIFF", b"riff"):
            suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="uploads")
    tmp.write(audio_data)
    tmp.close()
    logger.info(f"[Save] {tmp.name} ({len(audio_data):,} bytes, {suffix})")
    return tmp.name


def validate_audio(audio_data: bytes, min_size_kb: int = 1) -> bool:
    """Return True if audio_data is non-empty and above the minimum size threshold."""
    if not audio_data:
        return False
    size_kb = len(audio_data) / 1024
    if size_kb < min_size_kb:
        logger.warning(f"[Validate] Audio too small: {size_kb:.1f}KB < {min_size_kb}KB")
        return False
    return True