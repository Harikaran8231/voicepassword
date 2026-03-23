"""
email_sender.py - Send login notification emails via SMTP

Set in .env:
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=yourapp@gmail.com
  SMTP_PASS=your-app-password       ← Gmail: use App Password, not account password
  SMTP_FROM=VoiceAccess <yourapp@gmail.com>

Gmail setup:
  1. Enable 2-Factor Authentication on your Google account
  2. Go to Google Account → Security → App Passwords
  3. Generate an app password for "Mail"
  4. Use that 16-character password as SMTP_PASS
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", f"VoiceAccess <{SMTP_USER}>")


def _is_configured() -> bool:
    return bool(SMTP_USER and SMTP_PASS)


def send_login_notification(to_email: str, username: str) -> bool:
    """
    Send a login success notification to the user's registered email.
    Returns True if sent successfully, False otherwise.
    """
    if not _is_configured():
        logger.warning("[Email] SMTP not configured — skipping notification")
        return False

    if not to_email:
        logger.warning(f"[Email] No email for '{username}' — skipping")
        return False

    now = datetime.now(IST).strftime("%d %b %Y at %H:%M IST")

    subject = "VoiceAccess — Successful Login"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d0f1a; margin: 0; padding: 0; }}
    .wrap {{ max-width: 520px; margin: 40px auto; background: #151829; border-radius: 16px;
             border: 1px solid #1e2235; overflow: hidden; }}
    .header {{ background: linear-gradient(135deg, #00b8cc, #00e5ff); padding: 32px 36px; }}
    .header h1 {{ margin: 0; color: #0d0f1a; font-size: 1.4rem; font-weight: 800; }}
    .header p  {{ margin: 6px 0 0; color: #0d0f1a; opacity: 0.75; font-size: 0.85rem; }}
    .body {{ padding: 32px 36px; }}
    .body p {{ color: #c8d0e0; font-size: 0.95rem; line-height: 1.6; margin: 0 0 16px; }}
    .info-box {{ background: #0d0f1a; border: 1px solid #1e2235; border-radius: 10px;
                 padding: 16px 20px; margin: 20px 0; }}
    .info-row {{ display: flex; justify-content: space-between; padding: 6px 0;
                 border-bottom: 1px solid #1e2235; }}
    .info-row:last-child {{ border-bottom: none; }}
    .info-label {{ color: #6b7280; font-size: 0.8rem; font-family: monospace; }}
    .info-value {{ color: #e2e8f0; font-size: 0.8rem; font-weight: 600; }}
    .warning {{ background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);
                border-radius: 10px; padding: 14px 18px; margin-top: 20px; }}
    .warning p {{ color: #fca5a5; font-size: 0.82rem; margin: 0; }}
    .footer {{ padding: 20px 36px; border-top: 1px solid #1e2235; }}
    .footer p {{ color: #374151; font-size: 0.75rem; margin: 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>🔐 Login Successful</h1>
      <p>Your voice was recognised by VoiceAccess</p>
    </div>
    <div class="body">
      <p>Hi <strong style="color:#00e5ff">{username}</strong>,</p>
      <p>Your voice profile was successfully verified and you have been logged in to VoiceAccess.</p>
      <div class="info-box">
        <div class="info-row">
          <span class="info-label">USERNAME: </span>
          <span class="info-value">{username}</span>
        </div>
        <div class="info-row">
          <span class="info-label">TIME:</span>
          <span class="info-value">{now}</span>
        </div>
        <div class="info-row">
          <span class="info-label">METHOD: </span>
          <span class="info-value">Voice Biometric</span>
        </div>
      </div>
      <div class="warning">
        <p>⚠ If this wasn't you, your voice profile may have been compromised.
        Contact your administrator immediately.</p>
      </div>
    </div>
    <div class="footer">
      <p>This is an automated message from VoiceAccess. Do not reply to this email.</p>
    </div>
  </div>
</body>
</html>
"""

    plain = f"""VoiceAccess — Login Successful

Hi {username},

Your voice was recognised and you have been logged in.

Username : {username}
Time     : {now}
Method   : Voice Biometric

If this wasn't you, contact your administrator immediately.

— VoiceAccess
"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_FROM
        msg["To"]      = to_email
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to_email, msg.as_string())

        logger.info(f"[Email] Login notification sent to '{to_email}' for '{username}'")
        return True

    except Exception as e:
        logger.error(f"[Email] Failed to send to '{to_email}': {e}")
        return False