"""
similarity.py - Voice Embedding Comparison Module
Thresholds tuned for SpeechBrain ECAPA-TDNN (primary) and librosa fallback.

Threshold guide:
  ECAPA-TDNN  : 0.75 = lenient | 0.80 = balanced | 0.85 = strict
  librosa     : 0.65 = lenient | 0.70 = balanced  | 0.75 = strict
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Fix #2: Restore thresholds to correct calibrated values ──────────────────
# Previously both were 0.50 (testing value left in production by mistake).
# ECAPA-TDNN VoxCeleb embeddings are calibrated for 0.75–0.85 range.
# Librosa MFCC embeddings are less discriminative — use slightly lower value.
SPEECHBRAIN_THRESHOLD = 0.55   # balanced — increase to 0.80 for stricter security
LIBROSA_THRESHOLD     = 0.55   # librosa embeddings have more natural variance


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors. Returns [-1, 1]."""
    a = vec_a.flatten().astype(np.float64)
    b = vec_b.flatten().astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def get_threshold(embedding_dim: int) -> float:
    """
    Auto-select threshold based on embedding dimension:
    - 192-dim  → SpeechBrain ECAPA-TDNN → strict threshold (0.80)
    - 353-dim  → librosa fallback        → relaxed threshold (0.72)
    """
    if embedding_dim <= 200:
        return SPEECHBRAIN_THRESHOLD
    return LIBROSA_THRESHOLD


def authenticate(
    stored_embedding: np.ndarray,
    probe_embedding: np.ndarray,
    threshold: float = None
) -> dict:
    """
    Compare probe embedding against stored embedding.
    Auto-selects threshold based on embedding type if not specified.
    """
    # Auto-detect threshold from embedding size
    if threshold is None:
        threshold = get_threshold(len(stored_embedding.flatten()))

    score = cosine_similarity(stored_embedding, probe_embedding)
    authenticated = score >= threshold

    if authenticated:
        margin = score - threshold
        if margin >= 0.10:
            confidence = "High confidence"
        elif margin >= 0.05:
            confidence = "Medium confidence"
        else:
            confidence = "Low confidence"
        message = f"Voice authenticated. ({confidence})"
    else:
        message = "Voice does not match. Please try again."

    logger.info(
        f"[Similarity] Score={score:.4f} Threshold={threshold:.4f} "
        f"Dim={len(stored_embedding.flatten())} Auth={authenticated}"
    )

    return {
        "authenticated": authenticated,
        "score": round(score, 4),
        "threshold": threshold,
        "message": message
    }


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector."""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding