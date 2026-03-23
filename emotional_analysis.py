"""
emotion_analysis.py — Acoustic Sentiment Analysis Module

Thresholds calibrated from real recorded speech data:
  Energy  : -31.6 to -23.7 dB  (mean -27.9)
  ZCR     : 0.050 to 0.087     (mean 0.061)
  Centroid: 531 to 1044 Hz     (mean 731)
  Pitch   : 103 to 338 Hz      (mean 147)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def analyze_emotion_from_audio(audio_segment: np.ndarray, sr: int = 16000) -> dict:
    import librosa

    y = audio_segment.astype(np.float32)

    if len(y) < sr * 0.2:
        return {"emotion": "neutral", "confidence": 0.4,
                "emotion_note": "segment too short", "features": {}}

    # Energy
    rms            = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    mean_energy_db = float(20 * np.log10(np.mean(rms) + 1e-9))
    energy_std     = float(np.std(rms))

    # ZCR
    zcr      = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)[0]
    mean_zcr = float(np.mean(zcr))

    # Spectral centroid
    centroid      = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(centroid))

    # Spectral bandwidth
    bandwidth      = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mean_bandwidth = float(np.mean(bandwidth))

    # Pitch
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=2048
        )
        voiced_f0    = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
        mean_pitch   = float(np.mean(voiced_f0))  if len(voiced_f0) > 3 else 0.0
        pitch_std    = float(np.std(voiced_f0))   if len(voiced_f0) > 3 else 0.0
        pitch_range  = float(np.ptp(voiced_f0))   if len(voiced_f0) > 3 else 0.0
        voiced_ratio = float(len(voiced_f0)) / max(len(f0), 1) if f0 is not None else 0.0
    except Exception as e:
        logger.warning(f"[Emotion] Pitch failed: {e}")
        mean_pitch = pitch_std = pitch_range = voiced_ratio = 0.0

    # Tempo
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        tempo = 0.0

    features = {
        "mean_pitch_hz":        round(mean_pitch, 1),
        "pitch_std_hz":         round(pitch_std, 1),
        "pitch_range_hz":       round(pitch_range, 1),
        "voiced_ratio":         round(voiced_ratio, 3),
        "mean_energy_db":       round(mean_energy_db, 1),
        "energy_std":           round(energy_std, 5),
        "mean_zcr":             round(mean_zcr, 4),
        "spectral_centroid_hz": round(mean_centroid, 1),
        "spectral_bandwidth_hz":round(mean_bandwidth, 1),
        "speech_tempo_bpm":     round(tempo, 1),
    }

    emotion, confidence, note = _classify_emotion(features)
    return {"emotion": emotion, "confidence": round(confidence, 3),
            "emotion_note": note, "features": features}


def _classify_emotion(f: dict) -> tuple:
    """
    Calibrated classifier using thresholds derived from real recorded speech.
    Ranges calibrated:
      energy  : mean=-27.9  high>-24.8  low<-31.0
      zcr     : mean=0.061  high>0.072  low<0.053
      centroid: mean=731    high>910    low<552
      pitch   : mean=147    high>191    low<103
      p_range : mean=250    wide>500    narrow<70
    """
    energy   = f["mean_energy_db"]
    e_std    = f["energy_std"]
    zcr      = f["mean_zcr"]
    centroid = f["spectral_centroid_hz"]
    pitch    = f["mean_pitch_hz"]
    p_range  = f["pitch_range_hz"]
    p_std    = f["pitch_std_hz"]
    voiced   = f["voiced_ratio"]
    tempo    = f["speech_tempo_bpm"]

    scores = {
        "angry": 0.0, "happy": 0.0, "excited": 0.0, "sad": 0.0,
        "fearful": 0.0, "surprised": 0.0, "neutral": 0.0,
        "disgusted": 0.0, "confused": 0.0,
    }
    notes = []

    # ── ENERGY ───────────────────────────────────────────────────────────────
    if energy > -24.8:
        scores["angry"]   += 3.0; scores["excited"] += 2.0
        notes.append("high energy")
    elif energy > -26.5:
        scores["happy"]   += 1.5; scores["excited"] += 1.0; scores["angry"] += 0.5
        notes.append("above-avg energy")
    elif energy < -31.0:
        scores["sad"]     += 3.0; scores["fearful"] += 1.5
        notes.append("low energy")
    elif energy < -29.5:
        scores["sad"]     += 1.5; scores["neutral"] += 1.0
    else:
        scores["neutral"] += 1.5

    if e_std > 0.05:
        scores["angry"]   += 1.5; scores["excited"] += 1.0
        notes.append("energy bursts")
    elif e_std < 0.01:
        scores["neutral"] += 1.0; scores["sad"] += 0.5

    # ── ZCR ──────────────────────────────────────────────────────────────────
    if zcr > 0.0717:
        scores["angry"]   += 2.0; scores["fearful"] += 1.0
        notes.append("tense/rough voice")
    elif zcr > 0.065:
        scores["excited"] += 1.0; scores["angry"] += 0.5
    elif zcr < 0.053:
        scores["sad"]     += 1.5; scores["neutral"] += 0.5
        notes.append("smooth voice")

    # ── SPECTRAL CENTROID ─────────────────────────────────────────────────────
    if centroid > 910:
        scores["excited"]   += 2.5; scores["happy"] += 2.0; scores["surprised"] += 1.0
        notes.append("bright/high centroid")
    elif centroid > 800:
        scores["happy"]     += 1.0; scores["excited"] += 0.8
    elif centroid > 650:
        scores["neutral"]   += 1.0
    elif centroid < 552:
        scores["sad"]       += 2.5; scores["disgusted"] += 1.0
        notes.append("dark/low centroid")
    elif centroid < 620:
        scores["sad"]       += 1.5; scores["neutral"] += 0.5

    # ── PITCH ─────────────────────────────────────────────────────────────────
    if pitch > 191:
        scores["excited"]   += 2.5; scores["surprised"] += 2.0; scores["happy"] += 1.5
        notes.append(f"high pitch {pitch:.0f}Hz")
    elif pitch > 160:
        scores["happy"]     += 1.5; scores["excited"] += 1.0; scores["angry"] += 0.8
        notes.append(f"elevated pitch {pitch:.0f}Hz")
    elif pitch > 130:
        scores["neutral"]   += 1.0; scores["angry"] += 0.5
    elif 0 < pitch < 103:
        scores["sad"]       += 2.5; scores["disgusted"] += 1.0
        notes.append(f"very low pitch {pitch:.0f}Hz")
    elif pitch == 0:
        scores["neutral"]   += 1.0

    # ── PITCH RANGE ───────────────────────────────────────────────────────────
    if p_range > 500:
        scores["excited"]   += 3.0; scores["surprised"] += 2.0
        notes.append("very wide pitch range")
    elif p_range > 150:
        scores["excited"]   += 1.5; scores["angry"] += 1.0; scores["happy"] += 0.8
        notes.append("wide pitch range")
    elif p_range < 70 and voiced > 0.3:
        scores["sad"]       += 2.0; scores["neutral"] += 1.0
        notes.append("flat/monotone")

    if p_std > 40:
        scores["excited"]   += 1.0; scores["angry"] += 0.5
    elif p_std < 10 and voiced > 0.3:
        scores["sad"]       += 1.0; scores["neutral"] += 0.5

    # ── TEMPO ─────────────────────────────────────────────────────────────────
    if tempo > 140:
        scores["excited"]   += 1.5; scores["fearful"] += 1.0; scores["angry"] += 0.5
        notes.append("fast speech")
    elif 0 < tempo < 80:
        scores["sad"]       += 1.5; scores["neutral"] += 0.5
        notes.append("slow speech")

    # ── VOICED RATIO ──────────────────────────────────────────────────────────
    if voiced > 0.8:
        scores["excited"]   += 0.5; scores["angry"] += 0.3
    elif voiced < 0.3:
        scores["sad"]       += 1.0; scores["neutral"] += 0.5

    # ── COMPOUND RULES (calibrated) ───────────────────────────────────────────
    # Angry: high energy + tense voice + moderate pitch
    if energy > -24.8 and zcr > 0.065 and pitch > 130:
        scores["angry"]     += 3.0; notes.append("angry compound")

    # Excited: high pitch + wide range + bright centroid
    if pitch > 191 and p_range > 150 and centroid > 800:
        scores["excited"]   += 3.0; notes.append("excited compound")

    # Happy: elevated pitch + bright centroid + above-avg energy
    if pitch > 160 and centroid > 800 and energy > -26.5:
        scores["happy"]     += 2.5; notes.append("happy compound")

    # Sad: low pitch + low energy + dark centroid
    if pitch < 120 and energy < -29.5 and centroid < 650:
        scores["sad"]       += 3.0; notes.append("sad compound")

    # Fearful: high pitch + low energy + fast + tense
    if pitch > 160 and energy < -27.0 and zcr > 0.065:
        scores["fearful"]   += 2.5; notes.append("fearful compound")

    # Surprised: very high pitch + very wide range
    if pitch > 200 and p_range > 400:
        scores["surprised"] += 3.0; notes.append("surprised compound")

    # Neutral baseline
    scores["neutral"] += 0.3

    # Winner
    total = sum(scores.values())
    if total == 0:
        return "neutral", 0.5, "no signal"

    sorted_s     = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_emotion = sorted_s[0][0]
    best_score   = sorted_s[0][1]
    runner_up    = sorted_s[1][1] if len(sorted_s) > 1 else 0
    gap          = best_score - runner_up
    confidence   = min(0.5 + (gap / (total + 1e-9)) * 2.0, 0.99)

    if best_score < 1.0:
        best_emotion = "neutral"
        confidence   = 0.5

    note = ", ".join(notes) if notes else "moderate signal"
    return best_emotion, confidence, note


def analyze_utterance_emotions(audio_path: str, utterances: list, sr: int = 16000) -> list:
    import librosa

    try:
        y_full, _ = librosa.load(audio_path, sr=sr, mono=True)
        total_duration = len(y_full) / sr
        logger.info(f"[EmotionAnalysis] Loaded: {total_duration:.1f}s, {len(utterances)} utterances")
    except Exception as e:
        logger.error(f"[EmotionAnalysis] Failed to load: {e}")
        return utterances

    enriched = []
    for i, utt in enumerate(utterances):
        try:
            start_time = float(utt.get("start_time") or 0)

            if i + 1 < len(utterances):
                next_start = float(utterances[i+1].get("start_time") or (start_time + 5))
                end_time   = min(next_start + 0.5, total_duration)
            else:
                end_time = total_duration

            # Ensure minimum 2s for reliable analysis
            if end_time - start_time < 2.0:
                end_time = min(start_time + 2.0, total_duration)

            start_time = max(0.0, min(start_time, total_duration - 0.5))
            end_time   = max(start_time + 0.5, min(end_time, total_duration))

            segment = y_full[int(start_time * sr):int(end_time * sr)]

            # Widen window if segment too short
            if len(segment) < sr:
                center    = (start_time + end_time) / 2
                seg_start = max(0, int((center - 1.5) * sr))
                seg_end   = min(len(y_full), int((center + 1.5) * sr))
                segment   = y_full[seg_start:seg_end]

            result   = analyze_emotion_from_audio(segment, sr)
            utt_copy = dict(utt)
            utt_copy["emotion"]            = result["emotion"]
            utt_copy["emotion_confidence"] = result["confidence"]
            utt_copy["emotion_note"]       = result["emotion_note"]

            logger.info(
                f"[EmotionAnalysis] Utt {i+1} {utt.get('speaker')} "
                f"{start_time:.1f}-{end_time:.1f}s -> "
                f"{result['emotion']} ({result['confidence']:.2f}) | {result['emotion_note']}"
            )

        except Exception as e:
            logger.warning(f"[EmotionAnalysis] Utt {i+1} failed: {e}")
            utt_copy = dict(utt)
            utt_copy["emotion"]            = utt.get("emotion", "neutral")
            utt_copy["emotion_confidence"] = 0.4
            utt_copy["emotion_note"]       = "analysis failed"

        enriched.append(utt_copy)

    return enriched


def build_emotion_summary(utterances: list, speakers: list) -> dict:
    summary = {}
    for spk in speakers:
        spk_utts = [u for u in utterances if u.get("speaker") == spk]
        if not spk_utts:
            continue
        emotion_counts = {}
        for u in spk_utts:
            emo = u.get("emotion", "neutral")
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        total    = sum(emotion_counts.values())
        dominant = max(emotion_counts, key=emotion_counts.get)
        summary[spk] = {
            "dominant_emotion":  dominant,
            "emotion_breakdown": {
                emo: round(count/total*100)
                for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1])
            }
        }
    return summary