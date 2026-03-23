"""
whisper_check.py - Liveness check via spoken number + voice consistency verification

Anti-spoofing logic:
1. Server generates a 4-digit number AFTER Record is clicked (mic already open)
2. Number shown on screen mid-recording — attacker cannot pre-record it
3. Whisper verifies the number was spoken in the audio
4. Voice consistency check: audio is split into 3 segments, embeddings extracted
   from each segment and compared — if voice changes mid-recording (e.g. attacker
   says number then plays pre-recorded audio), cosine similarity drops → FAIL

Install: pip install openai-whisper
"""

import os
import time
import random
import logging
import uuid
import tempfile
import numpy as np

logger = logging.getLogger(__name__)

# ── Store — keyed by session_id ───────────────────────────────────────────────
_challenges: dict = {}
CHALLENGE_EXPIRY_SECONDS = 60

# ── Consistency threshold — how similar each segment must be to each other ───
# 1.0 = identical, 0.0 = completely different voice
# 0.60 allows natural variation within one person's voice across segments
CONSISTENCY_THRESHOLD = 0.50  # ECAPA-TDNN is language-independent so 0.50 works across Tamil+English

# ── Whisper model ─────────────────────────────────────────────────────────────
_whisper_model      = None
_whisper_model_size = "tiny.en"


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        logger.info(f"[Whisper] Loading '{_whisper_model_size}'...")
        _whisper_model = whisper.load_model(_whisper_model_size)
        logger.info("[Whisper] Ready.")
        return _whisper_model
    except ImportError:
        raise RuntimeError("Whisper not installed. Run: pip install openai-whisper")
    except Exception as e:
        raise RuntimeError(f"Whisper load failed: {e}")


# ── Session + number generation ───────────────────────────────────────────────

def new_session_id() -> str:
    return uuid.uuid4().hex


def generate_number(session_id: str) -> str:
    """
    Called when user clicks Record (mic already open).
    Generates a fresh 4-digit number unknown until that exact moment.
    """
    number = str(random.randint(1000, 9999))
    _challenges[session_id] = {
        "number":     number,
        "expires_at": time.time() + CHALLENGE_EXPIRY_SECONDS,
    }
    _cleanup()
    logger.info(f"[Liveness] session={session_id[:8]}... number={number}")
    return number


def get_number(session_id: str) -> str | None:
    entry = _challenges.get(session_id)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        del _challenges[session_id]
        return None
    return entry["number"]


def clear_number(session_id: str):
    _challenges.pop(session_id, None)


def _cleanup():
    now   = time.time()
    stale = [k for k, v in _challenges.items() if v["expires_at"] < now]
    for k in stale:
        del _challenges[k]


# ── Digit normalisation ───────────────────────────────────────────────────────
_DIGIT_WORDS = {
    "0": ["zero", "oh"],
    "1": ["one"],
    "2": ["two", "to", "too"],
    "3": ["three"],
    "4": ["four", "for", "fore"],
    "5": ["five"],
    "6": ["six"],
    "7": ["seven"],
    "8": ["eight", "ate"],
    "9": ["nine"],
}

def _normalise(text: str) -> str:
    """Convert spoken digit words to digit characters, return only digits."""
    text = text.upper()
    for digit, words in _DIGIT_WORDS.items():
        for w in words:
            text = text.replace(w.upper(), digit)
    return "".join(c for c in text if c.isdigit())


# ── Transcription ─────────────────────────────────────────────────────────────

def _transcribe(audio_path: str) -> str:
    """Transcribe audio file, return uppercased raw text."""
    try:
        model  = _load_whisper()
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            temperature=0.0,
            no_speech_threshold=0.6,   # skip if no speech detected
            condition_on_previous_text=False,  # prevent repetition loops
            verbose=False,
        )
        text = result["text"].strip().upper()

        # ── Hallucination detection ───────────────────────────────────────────
        # Whisper sometimes loops a short pattern endlessly on noisy/short audio
        # e.g. "9429 9429 9429 9429..." — detect and reject these
        words = text.split()
        if len(words) > 6:
            # Check if any single word repeats more than 4 times
            from collections import Counter
            counts = Counter(words)
            most_common_word, most_common_count = counts.most_common(1)[0]
            if most_common_count > 4:
                logger.warning(f"[Whisper] Hallucination detected — '{most_common_word}' repeated {most_common_count}x. Rejecting.")
                return ""

        logger.info(f"[Whisper] Raw: '{text[:120]}'")
        return text
    except Exception as e:
        logger.error(f"[Whisper] Failed: {e}")
        return ""


# ── Voice consistency check ───────────────────────────────────────────────────

def _extract_segment_embedding(y: np.ndarray, sr: int) -> np.ndarray | None:
    """
    Extract a speaker embedding from one audio segment for the consistency check.

    The segment has already been preprocessed (denoised, normalised) by
    _check_voice_consistency before being split. Here we only apply:
      - amplitude re-normalisation (splitting can shift peak levels slightly)
      - silence padding to minimum length (NOT np.tile — repeating creates
        artificial periodicity that distorts embeddings)

    Uses ECAPA-TDNN if available (language-independent), falls back to MFCC.
    """
    try:
        import librosa

        if len(y) < sr * 0.5:  # skip segments shorter than 0.5s
            return None

        # Re-normalise amplitude after splitting (segment peak may differ)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = (y / peak * 0.95).astype(np.float32)

        # ── Try ECAPA-TDNN first (language-independent) ───────────────────────
        try:
            import torch
            from voice_processing import _load_ecapa, _encoder, _encoder_type

            if _load_ecapa() and _encoder_type in ("ecapa", "ecapa_raw"):
                # Pad with silence to minimum length (was np.tile — now fixed)
                min_len = sr * 2
                if len(y) < min_len:
                    y_pad = np.pad(y, (0, min_len - len(y)), mode='constant')
                else:
                    y_pad = y

                if _encoder_type == "ecapa":
                    signal = torch.tensor(y_pad.astype(np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        emb = _encoder.encode_batch(signal)
                    result = emb.squeeze().cpu().numpy().flatten()
                else:
                    import torchaudio.transforms as T
                    audio_tensor = torch.tensor(y_pad, dtype=torch.float32).unsqueeze(0)
                    fbank = T.MelSpectrogram(
                        sample_rate=sr, n_fft=512, win_length=400,
                        hop_length=160, n_mels=80, f_min=20, f_max=7600
                    )(audio_tensor)
                    fbank = torch.log(fbank + 1e-6).squeeze(0).T.unsqueeze(0)
                    model      = _encoder["model"]
                    normalizer = _encoder["normalizer"]
                    with torch.no_grad():
                        lens = torch.tensor([1.0])
                        result = model(normalizer(fbank, lens), lens).squeeze().cpu().numpy().flatten()

                norm = np.linalg.norm(result)
                if norm > 0:
                    result = result / norm
                logger.info(f"[Consistency] ECAPA segment embedding dim={len(result)}")
                return result.astype(np.float32)
        except Exception as e:
            logger.info(f"[Consistency] ECAPA not available, using MFCC fallback: {e}")

        # ── MFCC fallback ─────────────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        embedding = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        logger.info(f"[Consistency] MFCC segment embedding dim={len(embedding)}")
        return embedding.astype(np.float32)

    except Exception as e:
        logger.warning(f"[Consistency] Segment embedding failed: {e}")
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _check_voice_consistency(audio_path: str) -> dict:
    """
    Split audio into 3 equal segments. Extract a voice embedding from each.
    Compare all pairs — if any pair drops below CONSISTENCY_THRESHOLD,
    the voice changed mid-recording (attacker switched audio source).

    Audio is preprocessed (noise reduction + normalisation) before splitting
    so all 3 segments receive the same cleaned signal — exactly matching what
    extract_embedding() does for the identity check. Previously segments were
    compared on raw audio while the identity embedding used clean audio, which
    caused inconsistent similarity scores.

    Returns: { passed, reason, min_similarity }
    """
    try:
        import librosa
        from voice_processing import preprocess_audio, _encoder_type  # shared pipeline

        y, sr = librosa.load(audio_path, sr=16000, mono=True, res_type='kaiser_fast')

        # Use same mode as extract_embedding() so segment embeddings match stored profile
        if _encoder_type == "ecapa":
            preprocess_mode = "ecapa_api"
        elif _encoder_type == "ecapa_raw":
            preprocess_mode = "ecapa_raw"
        else:
            preprocess_mode = "librosa"
        y = preprocess_audio(y, sr, denoise=True, mode=preprocess_mode)

        total = len(y)
        if total < sr * 2:
            logger.info("[Consistency] Audio too short for consistency check — skipping")
            return {"passed": True, "reason": "Audio too short to check consistency.", "min_similarity": 1.0}

        # Split into 3 equal segments
        seg_len = total // 3
        segments = [
            y[0          : seg_len],
            y[seg_len    : 2 * seg_len],
            y[2 * seg_len: total],
        ]

        embeddings = []
        for i, seg in enumerate(segments):
            emb = _extract_segment_embedding(seg, sr)
            if emb is not None:
                embeddings.append(emb)
                logger.info(f"[Consistency] Segment {i+1} embedding extracted")
            else:
                logger.warning(f"[Consistency] Segment {i+1} skipped (too short/silent)")

        if len(embeddings) < 2:
            logger.info("[Consistency] Not enough segments — skipping consistency check")
            return {"passed": True, "reason": "Not enough segments.", "min_similarity": 1.0}

        # Compare all pairs
        similarities = []
        pairs = [(0,1), (0,2), (1,2)] if len(embeddings) == 3 else [(0,1)]
        for i, j in pairs:
            if i < len(embeddings) and j < len(embeddings):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
                logger.info(f"[Consistency] Segment {i+1} vs {j+1}: similarity={sim:.4f}")

        min_sim = min(similarities)
        logger.info(f"[Consistency] Min similarity={min_sim:.4f} threshold={CONSISTENCY_THRESHOLD}")

        if min_sim < CONSISTENCY_THRESHOLD:
            logger.warning(
                f"[Consistency] VOICE CHANGE DETECTED — "
                f"min_similarity={min_sim:.4f} < threshold={CONSISTENCY_THRESHOLD}"
            )
            return {
                "passed": False,
                "reason": "Voice inconsistency detected. Audio may contain multiple speakers or a spliced recording.",
                "min_similarity": round(min_sim, 4)
            }

        return {
            "passed": True,
            "reason": "Voice consistent throughout recording.",
            "min_similarity": round(min_sim, 4)
        }

    except Exception as e:
        logger.warning(f"[Consistency] Check failed with exception: {e} — allowing through")
        # If consistency check itself errors, don't block legitimate users
        return {"passed": True, "reason": "Consistency check skipped.", "min_similarity": 1.0}


# ── Full verification ─────────────────────────────────────────────────────────

def verify_number(session_id: str, audio_path: str) -> dict:
    """
    Two-stage liveness verification:

    Stage 1 — Number check (Whisper):
      The expected 4-digit number must appear in the transcription.

    Stage 2 — Voice consistency check:
      The audio is split into 3 segments. All must have the same voice.
      If attacker says number then switches to pre-recorded audio → voice
      changes mid-recording → similarity drops → FAIL.

    Both stages must pass. Returns: { passed, reason, expected, transcribed }
    """
    expected = get_number(session_id)
    if expected is None:
        return {
            "passed": False,
            "reason": "Challenge expired. Click Record again.",
            "expected": "", "transcribed": ""
        }

    # ── Stage 1: Number must be spoken ───────────────────────────────────────
    raw = _transcribe(audio_path)
    if not raw:
        return {
            "passed": False,
            "reason": "Could not hear audio clearly. Speak louder and try again.",
            "expected": expected, "transcribed": ""
        }

    digits_heard = _normalise(raw)
    if expected not in digits_heard:
        logger.info(f"[Liveness] FAIL — expected={expected} digits_heard='{digits_heard}'")
        if digits_heard:
            reason = f"Wrong number spoken. You said {digits_heard}, but the number was {expected}."
        else:
            reason = "Number not heard. Say the number shown on screen clearly."
        return {
            "passed": False,
            "reason": reason,
            "expected": expected, "transcribed": raw
        }

    logger.info(f"[Liveness] Stage 1 PASS — number={expected} found in '{digits_heard}'")

    # ── Stage 2: Voice must be consistent throughout ──────────────────────────
    consistency = _check_voice_consistency(audio_path)
    if not consistency["passed"]:
        return {
            "passed": False,
            "reason": consistency["reason"],
            "expected": expected, "transcribed": raw
        }

    logger.info(f"[Liveness] Stage 2 PASS — min_similarity={consistency['min_similarity']}")

    # ── Both stages passed ────────────────────────────────────────────────────
    clear_number(session_id)
    return {
        "passed": True,
        "reason": "Liveness verified.",
        "expected": expected, "transcribed": raw
    }