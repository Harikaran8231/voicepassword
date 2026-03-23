"""
app.py - Main Flask Application for Voice Authentication System
"""

import os
import traceback
import logging
from flask import Flask, request, jsonify, render_template, session, redirect
from dotenv import load_dotenv

load_dotenv()  # loads .env file from project root

from database import init_db, user_exists, save_user, get_user_embedding, log_auth_attempt, delete_user, get_all_embeddings, get_all_users, get_user_email, get_recent_logins, update_user_embedding
from email_sender import send_login_notification
from voice_processing import extract_embedding, save_audio_temp, validate_audio
from similarity import authenticate, normalize_embedding
from whisper_check import new_session_id, generate_number, verify_number, get_number
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-in-production")
ADMIN_USER = os.getenv("ADMIN_USER", "").strip().lower()
if not ADMIN_USER:
    logger.warning("⚠  ADMIN_USER not set in .env — /admin will be inaccessible.")

os.makedirs("uploads", exist_ok=True)
init_db()
logger.info("Voice Authentication System initialized.")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/admin")
def admin_page():
    """Admin panel — requires fresh voice+number auth every single visit."""
    if not session.get("admin_authenticated"):
        return redirect("/admin/login")
    # Clear immediately — forces re-auth on every visit, no persistent sessions
    session.pop("admin_authenticated", None)
    return render_template("admin.html", admin_user=ADMIN_USER)

@app.route("/admin/login", methods=["GET"])
def admin_login_page():
    return render_template("admin_login.html")

@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    """
    Voice-based admin authentication with liveness check.
    Whisper (number check) + ECAPA-TDNN (identity) run IN PARALLEL.
    Both must pass. Identified user must match ADMIN_USER in .env.
    """
    audio_path = None
    try:
        if not ADMIN_USER:
            return jsonify({"success": False, "message": "ADMIN_USER not configured in .env."}), 403

        session_id = request.form.get("session_id", "").strip()
        audio_file = request.files.get("audio")

        if not session_id:
            return jsonify({"success": False, "message": "Session ID missing. Click Record again."}), 400
        if not audio_file:
            return jsonify({"success": False, "message": "Audio recording is required."}), 400

        audio_data = audio_file.read()
        if not validate_audio(audio_data):
            return jsonify({"success": False, "message": "Audio too short or empty."}), 400

        if get_number(session_id) is None:
            return jsonify({"success": False, "message": "Challenge expired. Click Record again.",
                            "challenge_expired": True}), 400

        if not user_exists(ADMIN_USER):
            return jsonify({"success": False, "message": f"Admin user '{ADMIN_USER}' is not registered yet."}), 404

        # Only load admin embedding — do NOT compare against all users
        # This prevents false matches when admin is also registered as a regular user
        admin_embedding = get_user_embedding(ADMIN_USER)
        if admin_embedding is None:
            return jsonify({"success": False, "message": "Admin voice profile not found."}), 404

        audio_path = save_audio_temp(audio_data, suffix=".wav")
        _audio_path = audio_path

        # Run Whisper + ECAPA-TDNN in parallel
        def run_liveness():
            return verify_number(session_id, _audio_path)

        def run_identity():
            try:
                emb = extract_embedding(_audio_path)
                return {"success": True, "embedding": emb}
            except ValueError as e:
                return {"success": False, "message": str(e).replace("VAD_FAIL: ", "")}
            except Exception as e:
                return {"success": False, "message": f"Voice processing failed: {str(e)}"}

        with ThreadPoolExecutor(max_workers=2) as ex:
            future_liveness = ex.submit(run_liveness)
            future_identity = ex.submit(run_identity)
            liveness_result = future_liveness.result()
            identity_result = future_identity.result()

        audio_path = None
        logger.info(f"[Admin] Liveness passed={liveness_result['passed']} | Identity success={identity_result['success']}")

        if not liveness_result["passed"]:
            return jsonify({
                "success":         False,
                "message":         f"Liveness check failed: {liveness_result['reason']}",
                "liveness_passed": False,
                "expected_number": liveness_result["expected"],
                "transcribed":     liveness_result["transcribed"],
            }), 401

        if not identity_result["success"]:
            return jsonify({"success": False, "message": identity_result["message"]}), 400

        # Compare probe voice directly against admin embedding only
        probe_embedding  = normalize_embedding(identity_result["embedding"])
        stored_embedding = normalize_embedding(admin_embedding)
        result           = authenticate(stored_embedding, probe_embedding)

        logger.info(f"[Admin] Score vs admin '{ADMIN_USER}': {result['score']:.4f} authenticated={result['authenticated']}")

        if result["authenticated"]:
            session["admin_authenticated"] = True
            session.permanent = False
            logger.info(f"[Admin] Access granted to '{ADMIN_USER}'")
            return jsonify({"success": True, "username": ADMIN_USER})
        else:
            logger.warning(f"[Admin] Access denied — score={result['score']:.4f}")
            return jsonify({
                "success": False,
                "message": f"Voice does not match the admin profile. Score: {round(result['score'] * 100)}%. Try again.",
                "score": round(result["score"], 4)
            }), 401

    except Exception as e:
        logger.error(f"[Admin] ERROR:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500
    finally:
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass


@app.route("/api/admin/logout", methods=["POST"])
def api_admin_logout():
    session.pop("admin_authenticated", None)
    return jsonify({"success": True})


# ── Users list ────────────────────────────────────────────────────────────────

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Return list of all registered usernames."""
    try:
        users = get_all_users()
        # Return just usernames as a simple list for the admin panel
        usernames = [u["username"] if isinstance(u, dict) else u for u in users]
        return jsonify({"success": True, "users": usernames})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ── Register ──────────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    """
    Fix #1: Multi-sample enrollment — accepts up to 3 audio recordings.

    The frontend sends audio_1, audio_2, audio_3 (or just audio for
    backward-compatible single-sample registration). All samples are
    processed into embeddings, then averaged into one robust profile.

    Multiple recordings cancel out:
      - Session-specific noise and room acoustics
      - Mic placement variation between recordings
      - Random voice fluctuation (tired vs alert, etc.)

    This is the single highest-impact accuracy improvement.
    """
    audio_paths = []
    try:
        logger.info("[Register] Request received")

        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()

        if not username:
            return jsonify({"success": False, "message": "Username is required."}), 400
        if len(username) < 2 or len(username) > 50:
            return jsonify({"success": False, "message": "Username must be 2-50 characters."}), 400
        if not email or "@" not in email or "." not in email:
            return jsonify({"success": False, "message": "A valid email address is required."}), 400

        if user_exists(username):
            return jsonify({"success": False, "message": "Username already registered."}), 409

        # ── Collect audio files — support audio_1/2/3 or legacy 'audio' key ─
        audio_files = []
        for key in ["audio_1", "audio_2", "audio_3"]:
            f = request.files.get(key)
            if f:
                audio_files.append(f)
        if not audio_files:
            # Backward compatible: single 'audio' field
            f = request.files.get("audio")
            if f:
                audio_files.append(f)

        if not audio_files:
            return jsonify({"success": False, "message": "At least one audio recording is required."}), 400

        logger.info(f"[Register] Username={username} samples={len(audio_files)}")

        # ── Extract embedding from each recording ─────────────────────────────
        embeddings = []
        for i, audio_file in enumerate(audio_files):
            audio_data = audio_file.read()
            if not validate_audio(audio_data):
                return jsonify({"success": False, "message": f"Recording {i+1} is too short or empty."}), 400

            audio_path = save_audio_temp(audio_data, suffix=".wav")
            audio_paths.append(audio_path)

            try:
                emb = extract_embedding(audio_path)
                audio_paths.remove(audio_path)   # extract_embedding deletes it
                audio_path = None
                embeddings.append(normalize_embedding(emb))
                logger.info(f"[Register] Sample {i+1}/{len(audio_files)} extracted dim={len(emb.flatten())}")
            except ValueError as e:
                msg = str(e).replace("VAD_FAIL: ", "")
                return jsonify({"success": False, "message": f"Recording {i+1}: {msg}"}), 400
            except Exception as e:
                logger.error(f"[Register] Embedding {i+1} failed:\n{traceback.format_exc()}")
                return jsonify({"success": False, "message": f"Voice processing failed on recording {i+1}: {str(e)}"}), 500

        # ── Save — database averages multiple embeddings automatically ────────
        dim        = len(embeddings[0].flatten())
        if dim != 192:
            return jsonify({"success": False, "message": f"Enrollment rejected: ECAPA-TDNN was not used (got {dim}-dim embedding). Delete pretrained_models/ and restart the app."}), 500
        model_type = "speechbrain-ecapa"

        # Pass list to save_user — it will average them (Fix #1 in database.py)
        payload = embeddings if len(embeddings) > 1 else embeddings[0]
        if not save_user(username, payload, model_type, email=email):
            return jsonify({"success": False, "message": "Failed to save user."}), 500

        sample_word = f"{len(embeddings)} recordings" if len(embeddings) > 1 else "1 recording"
        logger.info(f"[Register] SUCCESS: '{username}' model={model_type} samples={len(embeddings)}")
        return jsonify({
            "success": True,
            "message": f"Voice registered using {sample_word}! You can now log in, {username}.",
            "model":   model_type,
            "samples": len(embeddings),
        })

    except Exception as e:
        logger.error(f"[Register] ERROR:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500
    finally:
        for p in audio_paths:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass


# ── Challenge — called when Record button is clicked ──────────────────────────

@app.route("/api/challenge", methods=["POST"])
def api_challenge():
    """
    Called the moment user clicks Record.
    Generates a fresh 4-digit number server-side.
    Number unknown until this exact moment — prevents pre-recording.
    """
    session_id = new_session_id()
    number     = generate_number(session_id)
    logger.info(f"[Challenge] number={number} session={session_id[:8]}...")
    return jsonify({"success": True, "session_id": session_id, "number": number})


# ── Login — paragraph + number liveness check (parallel) ──────────────────────

@app.route("/api/login", methods=["POST"])
def api_login():
    """
    Passwordless, username-free voice login with liveness check.
    Whisper (liveness) and ECAPA-TDNN (identity) run IN PARALLEL for speed.
    Both must pass.

    Pure 1-to-N identification — user speaks, system finds who they are.
    No username field required or used.

    To keep 1-to-N accurate as the user base grows, two extra safety checks
    are applied on top of the threshold:

      1. Margin gap check — the best match score must exceed the second-best
         by at least MIN_MARGIN (0.08). If two users score similarly, the
         voice is ambiguous and login is rejected. This prevents a voice that
         sits between two users from slipping through as either one.

      2. Absolute floor — the winner must still clear the configured threshold
         in similarity.py regardless of how far ahead they are.
    """
    # Minimum score gap between 1st and 2nd best match.
    # Prevents ambiguous voice from matching the wrong user.
    MIN_MARGIN = 0.08

    audio_path = None
    try:
        logger.info("[Login] Request received")

        session_id = request.form.get("session_id", "").strip()
        audio_file = request.files.get("audio")

        if not session_id:
            return jsonify({"success": False, "message": "Session ID missing. Click Record again."}), 400
        if not audio_file:
            return jsonify({"success": False, "message": "Audio recording is required."}), 400

        audio_data = audio_file.read()
        if not validate_audio(audio_data):
            return jsonify({"success": False, "message": "Audio is too short or empty."}), 400

        if get_number(session_id) is None:
            return jsonify({"success": False, "message": "Challenge expired. Click Record again.",
                            "challenge_expired": True}), 400

        candidates = get_all_embeddings()
        if not candidates:
            return jsonify({"success": False, "message": "No registered users found."}), 404

        audio_path  = save_audio_temp(audio_data, suffix=".wav")
        _audio_path = audio_path

        # ── Run Whisper + ECAPA-TDNN in parallel ──────────────────────────────
        def run_liveness():
            return verify_number(session_id, _audio_path)

        def run_identity():
            try:
                emb = extract_embedding(_audio_path)
                return {"success": True, "embedding": emb}
            except ValueError as e:
                return {"success": False, "message": str(e).replace("VAD_FAIL: ", "")}
            except Exception as e:
                return {"success": False, "message": f"Voice processing failed: {str(e)}"}

        with ThreadPoolExecutor(max_workers=2) as ex:
            future_liveness = ex.submit(run_liveness)
            future_identity = ex.submit(run_identity)
            liveness_result = future_liveness.result()
            identity_result = future_identity.result()

        audio_path = None
        logger.info(f"[Login] Liveness passed={liveness_result['passed']} | Identity success={identity_result['success']}")

        if not liveness_result["passed"]:
            return jsonify({
                "success":         False,
                "message":         f"Liveness check failed: {liveness_result['reason']}",
                "liveness_passed": False,
                "expected_number": liveness_result["expected"],
                "transcribed":     liveness_result["transcribed"],
            }), 401

        if not identity_result["success"]:
            return jsonify({"success": False, "message": identity_result["message"]}), 400

        # ── Score all candidates, keep top-2 for margin gap check ─────────────
        probe_embedding = normalize_embedding(identity_result["embedding"])
        scored = []
        for user in candidates:
            stored = normalize_embedding(user["embedding"])
            result = authenticate(stored, probe_embedding)
            scored.append((result["score"], user["username"], result))
            logger.info(f"[Login] vs '{user['username']}': score={result['score']:.4f}")

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score,  best_user,  best_result  = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin       = best_score - second_score

        logger.info(
            f"[Login] Best='{best_user}' score={best_score:.4f} "
            f"2nd={second_score:.4f} margin={margin:.4f} "
            f"authenticated={best_result['authenticated']}"
        )

        log_auth_attempt(
            best_user,
            best_result["authenticated"],
            best_score,
            best_result["threshold"]
        )

        # ── Reject if voice is ambiguous between two users ────────────────────
        if best_result["authenticated"] and len(scored) > 1 and margin < MIN_MARGIN:
            logger.warning(
                f"[Login] Ambiguous match — margin={margin:.4f} < MIN_MARGIN={MIN_MARGIN}. "
                f"Top candidates: '{best_user}'={best_score:.4f}, '{scored[1][1]}'={second_score:.4f}"
            )
            return jsonify({
                "success": False,
                "message": "Voice match is ambiguous. Please speak more clearly and try again.",
                "score":   round(best_score, 4),
            }), 401

        if best_result["authenticated"]:
            # Block admin from logging in through user login
            if best_user.lower() == ADMIN_USER:
                logger.warning(f"[Login] Admin '{best_user}' attempted user login — blocked")
                return jsonify({
                    "success": False,
                    "message": "Admin accounts cannot log in here. Please use the Admin panel.",
                }), 403

            dim        = len(probe_embedding.flatten())
            model_type = "speechbrain-ecapa"
            logger.info(f"[Login] SUCCESS: '{best_user}' score={best_score:.4f} margin={margin:.4f}")
            from threading import Thread
            user_email = get_user_email(best_user)
            Thread(target=send_login_notification, args=(user_email, best_user), daemon=True).start()
            return jsonify({
                "success":         True,
                "username":        best_user,
                "message":         best_result["message"],
                "score":           best_result["score"],
                "model":           model_type,
                "liveness_passed": True,
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Voice not recognised. Best match: {round(best_score * 100)}%. Try again.",
                "score":   round(best_score, 4),
            }), 401

    except Exception as e:
        logger.error(f"[Login] ERROR:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500
    finally:
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass


@app.route("/api/recent-logins", methods=["GET"])
def api_recent_logins():
    """Return the most recent successful logins for the admin panel."""
    try:
        logs = get_recent_logins(limit=20)
        return jsonify({"success": True, "logins": logs})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/update-voice", methods=["POST"])
def api_update_voice():
    """
    Fix #9: Voice profile re-enrollment endpoint.

    Two-phase process:
      Phase 1 — Verify current voice (must pass at existing threshold).
                 Prevents an attacker from overwriting someone's profile.
      Phase 2 — Accept 1–3 new recordings, extract and average embeddings,
                 update the stored profile.

    Request form fields:
      username         : the account to update
      current_audio    : single recording to verify identity first
      session_id       : liveness challenge session (from /api/challenge)
      new_audio_1      : new voice recording (required)
      new_audio_2      : new voice recording (optional)
      new_audio_3      : new voice recording (optional)

    Use cases:
      - Voice changed (illness, aging, new microphone)
      - Poor initial enrollment — user wants to re-record
      - Periodic re-enrollment for security hygiene
    """
    audio_paths = []
    try:
        username      = request.form.get("username", "").strip()
        session_id    = request.form.get("session_id", "").strip()
        current_audio = request.files.get("current_audio")

        if not username:
            return jsonify({"success": False, "message": "Username required."}), 400
        if not session_id:
            return jsonify({"success": False, "message": "Session ID missing. Click Record again."}), 400
        if not current_audio:
            return jsonify({"success": False, "message": "Current voice recording required to verify identity."}), 400

        if not user_exists(username):
            return jsonify({"success": False, "message": f"User '{username}' not found."}), 404

        stored_embedding = get_user_embedding(username)
        if stored_embedding is None:
            return jsonify({"success": False, "message": "Voice profile not found."}), 404

        if get_number(session_id) is None:
            return jsonify({"success": False, "message": "Challenge expired. Click Record again.",
                            "challenge_expired": True}), 400

        # ── Phase 1: Verify current voice ────────────────────────────────────
        current_data = current_audio.read()
        if not validate_audio(current_data):
            return jsonify({"success": False, "message": "Current audio is too short or empty."}), 400

        current_path = save_audio_temp(current_data, suffix=".wav")
        audio_paths.append(current_path)
        _current_path = current_path

        def run_liveness():
            return verify_number(session_id, _current_path)

        def run_identity():
            try:
                emb = extract_embedding(_current_path)
                return {"success": True, "embedding": emb}
            except ValueError as e:
                return {"success": False, "message": str(e).replace("VAD_FAIL: ", "")}
            except Exception as e:
                return {"success": False, "message": str(e)}

        with ThreadPoolExecutor(max_workers=2) as ex:
            liveness_result = ex.submit(run_liveness).result()
            identity_result = ex.submit(run_identity).result()

        if current_path in audio_paths:
            audio_paths.remove(current_path)

        if not liveness_result["passed"]:
            return jsonify({"success": False, "message": f"Liveness check failed: {liveness_result['reason']}"}), 401

        if not identity_result["success"]:
            return jsonify({"success": False, "message": identity_result["message"]}), 400

        probe = normalize_embedding(identity_result["embedding"])
        stored = normalize_embedding(stored_embedding)
        verify = authenticate(stored, probe)

        if not verify["authenticated"]:
            logger.warning(f"[UpdateVoice] Identity verify failed for '{username}' score={verify['score']:.4f}")
            return jsonify({
                "success": False,
                "message": f"Current voice does not match. Score: {round(verify['score'] * 100)}%. Cannot update profile.",
                "score": verify["score"]
            }), 401

        logger.info(f"[UpdateVoice] Phase 1 passed for '{username}' score={verify['score']:.4f}")

        # ── Phase 2: Collect new recordings ──────────────────────────────────
        new_files = []
        for key in ["new_audio_1", "new_audio_2", "new_audio_3"]:
            f = request.files.get(key)
            if f:
                new_files.append(f)

        if not new_files:
            return jsonify({"success": False, "message": "At least one new recording is required."}), 400

        new_embeddings = []
        for i, af in enumerate(new_files):
            data = af.read()
            if not validate_audio(data):
                return jsonify({"success": False, "message": f"New recording {i+1} is too short or empty."}), 400
            p = save_audio_temp(data, suffix=".wav")
            audio_paths.append(p)
            try:
                emb = extract_embedding(p)
                if p in audio_paths:
                    audio_paths.remove(p)
                new_embeddings.append(normalize_embedding(emb))
            except ValueError as e:
                return jsonify({"success": False, "message": f"New recording {i+1}: {str(e).replace('VAD_FAIL: ', '')}"}), 400
            except Exception as e:
                return jsonify({"success": False, "message": f"Failed to process new recording {i+1}: {str(e)}"}), 500

        dim        = len(new_embeddings[0].flatten())
        if dim != 192:
            return jsonify({"success": False, "message": f"Voice update rejected: ECAPA-TDNN not active (got {dim}-dim). Restart the app."}), 500
        model_type = "speechbrain-ecapa"
        payload    = new_embeddings if len(new_embeddings) > 1 else new_embeddings[0]

        if not update_user_embedding(username, payload, model_type):
            return jsonify({"success": False, "message": "Failed to update voice profile."}), 500

        sample_word = f"{len(new_embeddings)} recordings" if len(new_embeddings) > 1 else "1 recording"
        logger.info(f"[UpdateVoice] SUCCESS: '{username}' updated with {len(new_embeddings)} sample(s)")
        return jsonify({
            "success": True,
            "message": f"Voice profile updated using {sample_word}.",
            "samples": len(new_embeddings),
            "model":   model_type,
        })

    except Exception as e:
        logger.error(f"[UpdateVoice] ERROR:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500
    finally:
        for p in audio_paths:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.route("/api/check-user/<username>", methods=["GET"])
def api_check_user(username):
    return jsonify({"exists": user_exists(username.strip())})

@app.route("/api/check_username", methods=["GET"])
def api_check_username():
    username = request.args.get("username", "").strip()
    if not username:
        return jsonify({"exists": False})
    return jsonify({"exists": user_exists(username)})

@app.route("/api/delete-user", methods=["POST"])
def api_delete_user():
    try:
        username = request.json.get("username", "").strip()
        if not username:
            return jsonify({"success": False, "message": "Username required."}), 400
        return jsonify({"success": delete_user(username)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ── Call Analysis ─────────────────────────────────────────────────────────────

@app.route("/call-analysis")
def call_analysis_page():
    return render_template("call_analysis.html")


@app.route("/api/analyze-call", methods=["POST"])
def api_analyze_call():
    try:
        return _api_analyze_call_inner()
    except Exception as _top_err:
        import traceback as _tb
        logger.error(f"[AnalyzeCall] TOP-LEVEL ERROR: {_tb.format_exc()}")
        return jsonify({"success": False, "message": f"Server error: {str(_top_err)}"}), 500

def _api_analyze_call_inner():
    """
    Two-stage pipeline:
      Stage 1 — Gemini transcribes the audio and identifies speaker turns.
      Stage 2 — Acoustic sentiment analysis (librosa) analyzes real audio
                features (pitch, energy, ZCR, spectral centroid, tempo)
                for each utterance segment. Emotions come from the AUDIO
                SIGNAL, not from the words spoken.
    """
    import base64
    import json
    import urllib.request
    import urllib.error

    try:
        from emotion_analysis import analyze_utterance_emotions, build_emotion_summary
        EMOTION_MODULE_OK = True
    except Exception as _emo_import_err:
        logger.warning(f"[AnalyzeCall] emotion_analysis import failed: {_emo_import_err}")
        EMOTION_MODULE_OK = False

    audio_file = request.files.get("audio")
    language   = request.form.get("language", "auto").strip()

    if not audio_file:
        return jsonify({"success": False, "message": "No audio file received."}), 400

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return jsonify({"success": False, "message": "GOOGLE_API_KEY not set in .env"}), 500

    # Save audio to temp file for acoustic analysis
    audio_data  = audio_file.read()
    mime_type   = audio_file.content_type or "audio/mpeg"
    audio_b64   = base64.b64encode(audio_data).decode("utf-8")

    # Save to temp file for librosa processing
    import tempfile
    suffix = ".mp3"
    if mime_type == "audio/wav" or mime_type == "audio/x-wav":
        suffix = ".wav"
    elif mime_type == "audio/mp4" or mime_type == "video/mp4":
        suffix = ".mp4"
    elif mime_type == "audio/ogg":
        suffix = ".ogg"
    elif mime_type == "audio/flac":
        suffix = ".flac"
    elif mime_type == "audio/webm" or mime_type == "video/webm":
        suffix = ".webm"
    elif mime_type == "audio/m4a" or mime_type == "audio/x-m4a":
        suffix = ".m4a"

    os.makedirs("uploads", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="uploads")
    tmp.write(audio_data)
    tmp.close()
    audio_temp_path = tmp.name

    lang_hint = "" if language == "auto" else f"The spoken language is: {language}."

    # Full prompt — transcription + emotion analysis together
    # Gemini analyzes BOTH the audio tone AND the words/context
    prompt = (
        "You are an expert call center analyst specializing in transcription and emotional intelligence.\n"
        "Listen carefully to this audio recording. " + lang_hint + "\n\n"
        "Your task:\n"
        "1. Transcribe the ENTIRE conversation, splitting by speaker turns.\n"
        "2. Label speakers as \"Speaker 1\", \"Speaker 2\", etc.\n"
        "3. Estimate the approximate start time in seconds of each utterance.\n"
        "4. For EACH utterance, analyze the VOCAL TONE, PITCH, SPEED, and WORD CHOICE\n"
        "   to determine the TRUE emotional state. Be very specific — do NOT default to neutral.\n"
        "   Listen for: frustration, excitement, sadness, anger, happiness, fear, surprise.\n"
        "   Choose from: happy, sad, angry, excited, neutral, fearful, surprised, disgusted, confused\n"
        "5. Write a short emotion_note explaining WHY you detected that emotion\n"
        "   (e.g. \"raised voice and fast speech\", \"slow dejected tone\", \"enthusiastic and upbeat\")\n\n"
        "IMPORTANT: Vary the emotions — real conversations have emotional shifts.\n"
        "Only use neutral when the speaker is genuinely calm with no emotional signal.\n\n"
        "Return ONLY valid JSON, no markdown, no explanation:\n"
        "{\n"
        '  "duration_seconds": 120,\n'
        '  "speakers": ["Speaker 1", "Speaker 2"],\n'
        '  "utterances": [\n'
        '    {\n'
        '      "speaker": "Speaker 1",\n'
        '      "start_time": 0,\n'
        '      "text": "I have been waiting for 30 minutes!",\n'
        '      "emotion": "angry",\n'
        '      "emotion_confidence": 0.92,\n'
        '      "emotion_note": "raised voice, fast speech, complaint phrasing"\n'
        '    }\n'
        '  ],\n'
        '  "summary": {\n'
        '    "Speaker 1": {"dominant_emotion": "angry", "emotion_breakdown": {"angry": 60, "sad": 25, "neutral": 15}},\n'
        '    "Speaker 2": {"dominant_emotion": "neutral", "emotion_breakdown": {"neutral": 50, "happy": 30, "fearful": 20}}\n'
        '  }\n'
        "}"
    )

    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
    }

    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    try:
        result     = None
        last_error = ""
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                logger.info(f"[AnalyzeCall] Using model: {model_name}")
                break
            except urllib.error.HTTPError as e:
                last_error = e.read().decode("utf-8")
                logger.warning(f"[AnalyzeCall] Model {model_name} failed: {e.code}")
                continue

        if result is None:
            return jsonify({"success": False, "message": f"All Gemini models failed. Last error: {last_error}"}), 500

        raw_text = (
            result.get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text", "")
        )
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return jsonify({"success": False, "message": "Gemini returned unexpected format.", "raw": raw_text}), 500

        # ── Use Gemini emotions + build summary if missing ──────────────────
        utterances = parsed.get("utterances", [])
        speakers   = parsed.get("speakers", [])

        # If Gemini already returned a summary use it, otherwise build from utterances
        if not parsed.get("summary"):
            summary = {}
            for spk in speakers:
                spk_utts = [u for u in utterances if u.get("speaker") == spk]
                if spk_utts:
                    emos = [u.get("emotion", "neutral") for u in spk_utts]
                    dominant = max(set(emos), key=emos.count)
                    counts = {e: emos.count(e) for e in set(emos)}
                    total  = len(emos)
                    summary[spk] = {
                        "dominant_emotion":  dominant,
                        "emotion_breakdown": {e: round(c/total*100) for e, c in counts.items()}
                    }
            parsed["summary"] = summary

        # Run acoustic analysis on top if module is available
        # This enhances Gemini's emotions with real signal data for natural voice
        if EMOTION_MODULE_OK:
            try:
                enriched   = analyze_utterance_emotions(audio_temp_path, utterances, sr=16000)
                # Only override if acoustic analysis found non-neutral emotions
                non_neutral = [u for u in enriched if u.get("emotion") != "neutral"]
                if len(non_neutral) > len(utterances) * 0.3:
                    parsed["utterances"] = enriched
                    parsed["summary"]    = build_emotion_summary(enriched, speakers)
                    logger.info("[AnalyzeCall] Using acoustic emotions (sufficient variance)")
                else:
                    logger.info("[AnalyzeCall] Acoustic returned mostly neutral — keeping Gemini emotions")
            except Exception as e:
                logger.warning(f"[AnalyzeCall] Acoustic analysis skipped: {e}")

        return jsonify({"success": True, "data": parsed})

    except Exception:
        logger.error(f"[AnalyzeCall] ERROR:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": "Analysis failed. Check server logs."}), 500
    finally:
        if os.path.exists(audio_temp_path):
            try:
                os.remove(audio_temp_path)
            except Exception:
                pass


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "message": "Audio file too large (max 16MB)."}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Unhandled 500: {traceback.format_exc()}")
    return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)