"""
database.py - MongoDB database management for Voice Authentication System

Install: pip install pymongo
Set in .env:
  MONGO_URI=mongodb://localhost:27017/voicekey          (local)
  MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/voicekey  (Atlas)
"""

import io
import hashlib
import logging
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os

logger = logging.getLogger(__name__)

# ── Connection ────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/voicekey")

_client: MongoClient = None
_db = None


def _get_db():
    global _client, _db
    if _db is not None:
        return _db
    _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db_name = MONGO_URI.rsplit("/", 1)[-1].split("?")[0] or "voicekey"
    _db = _client[db_name]
    return _db


def init_db():
    """
    Initialize MongoDB collections and indexes.
    Safe to call multiple times.
    """
    db = _get_db()
    db.users.create_index("username_hash", unique=True)
    db.users.create_index("username")
    db.auth_logs.create_index("username_hash")
    db.auth_logs.create_index("timestamp")
    logger.info(f"[DB] MongoDB connected — db='{db.name}'")


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_username(username: str) -> str:
    return hashlib.sha256(username.strip().lower().encode()).hexdigest()


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, embedding)
    return buf.getvalue()


def _bytes_to_embedding(data) -> np.ndarray:
    if isinstance(data, bytes):
        raw = data
    else:
        raw = bytes(data)
    return np.load(io.BytesIO(raw))


# ── User operations ───────────────────────────────────────────────────────────

def user_exists(username: str) -> bool:
    db = _get_db()
    return db.users.find_one({"username_hash": hash_username(username)}) is not None


def save_user(username: str, embedding, model_type: str = "auto", email: str = "") -> bool:
    """
    Save a user's voice profile.

    Fix #1: `embedding` can now be either:
      - A single np.ndarray  (backward compatible — single recording)
      - A list of np.ndarray (multi-sample enrollment — 3 recordings averaged)

    When a list is provided, all embeddings are averaged and L2-normalised
    into a single robust vector. Averaging across multiple recordings cancels
    out session-specific noise, mic placement variation, and random voice
    fluctuation — producing a more stable reference embedding.
    """
    try:
        # ── Fix #1: Handle multi-sample enrollment ────────────────────────────
        if isinstance(embedding, list):
            if len(embedding) == 0:
                raise ValueError("Embedding list is empty")
            # Stack all enrollment embeddings and compute mean
            stacked = np.stack([e.flatten().astype(np.float64) for e in embedding])
            mean_emb = np.mean(stacked, axis=0)
            norm = np.linalg.norm(mean_emb)
            final_embedding = (mean_emb / norm if norm > 0 else mean_emb).astype(np.float32)
            logger.info(f"[DB] Averaged {len(embedding)} enrollment embeddings → dim={len(final_embedding)}")
        else:
            final_embedding = embedding

        if model_type == "auto":
            dim = len(final_embedding.flatten())
            model_type = "speechbrain-ecapa" if dim <= 200 else "librosa-fallback"

        db = _get_db()
        db.users.insert_one({
            "username_hash":      hash_username(username),
            "username":           username.strip(),
            "embedding":          _embedding_to_bytes(final_embedding),
            "embedding_dim":      int(len(final_embedding.flatten())),
            "model_type":         model_type,
            "email":              email.strip().lower(),
            "enrollment_samples": len(embedding) if isinstance(embedding, list) else 1,
            "created_at":         datetime.now(timezone.utc),
        })
        samples = len(embedding) if isinstance(embedding, list) else 1
        logger.info(f"[DB] Registered '{username}' model={model_type} samples={samples}")
        return True

    except DuplicateKeyError:
        logger.warning(f"[DB] save_user: '{username}' already exists")
        return False
    except Exception:
        import traceback
        logger.error(f"[DB] save_user failed:\n{traceback.format_exc()}")
        return False


def get_user_embedding(username: str):
    db = _get_db()
    doc = db.users.find_one({"username_hash": hash_username(username)})
    if doc is None:
        return None
    embedding = _bytes_to_embedding(doc["embedding"])
    logger.info(f"[DB] Loaded embedding for '{username}' dim={doc.get('embedding_dim')} model={doc.get('model_type')}")
    return embedding


def update_user_embedding(username: str, embedding, model_type: str = "auto") -> bool:
    """
    Fix #9: Update an existing user's voice profile (re-enrollment).

    Called from /api/update-voice after the current voice passes authentication.
    Supports single embedding or list of embeddings (averaged, same as save_user).

    Use cases:
      - User's voice has changed (illness, aging, new mic)
      - User wants to improve accuracy after a bad initial enrollment
      - Periodic re-enrollment for security hygiene
    """
    try:
        if isinstance(embedding, list):
            if len(embedding) == 0:
                raise ValueError("Embedding list is empty")
            stacked = np.stack([e.flatten().astype(np.float64) for e in embedding])
            mean_emb = np.mean(stacked, axis=0)
            norm = np.linalg.norm(mean_emb)
            final_embedding = (mean_emb / norm if norm > 0 else mean_emb).astype(np.float32)
            logger.info(f"[DB] Re-enrollment: averaged {len(embedding)} embeddings")
        else:
            final_embedding = embedding

        if model_type == "auto":
            dim = len(final_embedding.flatten())
            model_type = "speechbrain-ecapa" if dim <= 200 else "librosa-fallback"

        db = _get_db()
        result = db.users.update_one(
            {"username_hash": hash_username(username)},
            {"$set": {
                "embedding":          _embedding_to_bytes(final_embedding),
                "embedding_dim":      int(len(final_embedding.flatten())),
                "model_type":         model_type,
                "enrollment_samples": len(embedding) if isinstance(embedding, list) else 1,
                "updated_at":         datetime.now(timezone.utc),
            }}
        )
        if result.matched_count == 0:
            logger.warning(f"[DB] update_user_embedding: '{username}' not found")
            return False
        logger.info(f"[DB] Updated voice profile for '{username}' model={model_type}")
        return True

    except Exception:
        import traceback
        logger.error(f"[DB] update_user_embedding failed:\n{traceback.format_exc()}")
        return False



def get_user_email(username: str) -> str:
    """Return the email address for a registered user, or empty string."""
    db = _get_db()
    doc = db.users.find_one({"username_hash": hash_username(username)}, {"email": 1})
    if doc is None:
        return ""
    return doc.get("email", "")


def delete_user(username: str) -> bool:
    try:
        db = _get_db()
        result = db.users.delete_one({"username_hash": hash_username(username)})
        deleted = result.deleted_count > 0
        if deleted:
            logger.info(f"[DB] Deleted '{username}'")
        else:
            logger.warning(f"[DB] delete_user: '{username}' not found")
        return deleted
    except Exception as e:
        logger.error(f"[DB] delete_user failed: {e}")
        return False


def get_all_embeddings() -> list:
    db = _get_db()
    result = []
    for doc in db.users.find({}, {"username": 1, "embedding": 1, "model_type": 1, "embedding_dim": 1}):
        try:
            result.append({
                "username":      doc["username"],
                "embedding":     _bytes_to_embedding(doc["embedding"]),
                "model_type":    doc.get("model_type", "unknown"),
                "embedding_dim": doc.get("embedding_dim", 0),
            })
        except Exception as e:
            logger.warning(f"[DB] Skipping corrupt embedding for '{doc.get('username')}': {e}")
    return result


def get_all_users() -> list:
    db = _get_db()
    users = []
    for doc in db.users.find(
        {},
        {"username": 1, "model_type": 1, "embedding_dim": 1, "created_at": 1}
    ).sort("username", 1):
        created = doc.get("created_at", "")
        users.append({
            "username":      doc["username"],
            "model_type":    doc.get("model_type", "unknown"),
            "embedding_dim": doc.get("embedding_dim", 0),
            "created_at":    created.isoformat() if isinstance(created, datetime) else str(created),
        })
    return users


# ── Auth logging ──────────────────────────────────────────────────────────────

def log_auth_attempt(username: str, success: bool,
                     similarity_score: float = None, threshold: float = None):
    try:
        db = _get_db()
        db.auth_logs.insert_one({
            "username_hash":    hash_username(username),
            "username":         username,
            "success":          bool(success),
            "similarity_score": similarity_score,
            "threshold":        threshold,
            "timestamp":        datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.error(f"[DB] log_auth_attempt failed: {e}")

def clear_auth_logs() -> int:
    """Delete all auth log entries. Returns number of deleted documents."""
    try:
        db = _get_db()
        result = db.auth_logs.delete_many({})
        logger.info(f"[DB] Cleared {result.deleted_count} auth log entries")
        return result.deleted_count
    except Exception as e:
        logger.error(f"[DB] clear_auth_logs failed: {e}")
        return -1


def get_recent_logins(limit: int = 20) -> list:
    """Return the most recent successful login attempts, newest first."""
    db = _get_db()
    logs = []
    for doc in db.auth_logs.find(
        {"success": True},
        {"username": 1, "similarity_score": 1, "timestamp": 1}
    ).sort("timestamp", -1).limit(limit):
        ts = doc.get("timestamp", "")
        logs.append({
            "username":         doc.get("username", "unknown"),
            "similarity_score": round(doc.get("similarity_score") or 0, 4),
            "timestamp":        ts.isoformat() if isinstance(ts, datetime) else str(ts),
        })
    return logs