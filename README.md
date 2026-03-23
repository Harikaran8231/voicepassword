# 🎙️ VoiceKey — Voice Biometric Authentication

A production-ready voice authentication system using deep neural voice embeddings for secure, passwordless login.

## Features
- 🎤 **Voice Registration** — Enroll voice in 5 seconds
- 🔐 **Voice Login** — Authenticate with cosine similarity matching
- 🧠 **ECAPA-TDNN** — SpeechBrain deep speaker embedding (librosa fallback)
- 🗑️ **Privacy-first** — Raw audio deleted after embedding extraction
- 📊 **Audit Logs** — All auth attempts logged to SQLite
- ⚡ **Real-time Visualizer** — Live waveform and countdown timer

## Project Structure

```
voice_auth/
├── app.py                 # Main Flask application & API routes
├── database.py            # SQLite database management
├── voice_processing.py    # Audio embedding extraction
├── similarity.py          # Cosine similarity + authentication logic
├── requirements.txt       # Python dependencies
├── .env.example           # Environment configuration template
├── uploads/               # Temporary audio (auto-cleaned)
├── pretrained_models/     # SpeechBrain model cache (auto-downloaded)
└── templates/
    ├── index.html         # Home page (Login / Register)
    ├── register.html      # Voice enrollment page
    └── login.html         # Voice login page
```

## Quick Start

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env to set SECRET_KEY and other settings
```

### 3. Run the server
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### 4. Production deployment (Gunicorn)
```bash
gunicorn app:app -b 0.0.0.0:5000 -w 4
```

## Enabling SpeechBrain (High Accuracy)

For production-grade accuracy, install SpeechBrain:
```bash
pip install speechbrain torch torchaudio
```

The system will automatically detect and use SpeechBrain's ECAPA-TDNN model (auto-downloaded on first run, ~100MB).

## Authentication Threshold

Adjust the similarity threshold in `similarity.py`:
```python
DEFAULT_THRESHOLD = 0.80  # 0.75 = permissive | 0.85 = strict
```

| Score Range | Result |
|-------------|--------|
| ≥ 0.85 | High confidence match |
| 0.80–0.85 | Medium confidence match |
| 0.75–0.80 | Low confidence match |
| < 0.75 | Rejected |

## Security Notes

- Raw audio files are deleted immediately after embedding extraction
- Usernames are hashed with SHA-256 in the database
- Embeddings stored as binary BLOBs (not reconstructable to audio)
- Threshold-based authentication prevents replay attacks
- No audio permanently stored on server

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/register` | Registration page |
| GET | `/login` | Login page |
| POST | `/api/register` | Register new user with voice |
| POST | `/api/login` | Authenticate user with voice |
| GET | `/api/check-user/<username>` | Check if username exists |
