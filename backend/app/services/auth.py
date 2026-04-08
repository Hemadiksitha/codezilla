"""JWT authentication service for NOVA-C.

Users are stored in a local JSON file (users.json).
Passwords are hashed with bcrypt.
Tokens are signed with HS256.
"""

from __future__ import annotations
import json
import os
import uuid
import time
from pathlib import Path

import bcrypt
import jwt

# ─── Config ───────────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "nova-c-hackathon-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 60 * 60 * 24  # 24 hours

USERS_FILE = Path(__file__).parent.parent.parent.parent / "users.json"


# ─── User store helpers ───────────────────────────────────────────────────────

def _load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


# ─── Public API ────────────────────────────────────────────────────────────────

def sign_up(email: str, password: str) -> dict:
    """Register a new user. Returns user info + JWT."""
    email = email.strip().lower()
    users = _load_users()

    if email in users:
        raise ValueError("Email already registered")

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_id = str(uuid.uuid4())[:8]
    users[email] = {"user_id": user_id, "email": email, "password_hash": hashed}
    _save_users(users)

    token = _create_token(user_id, email)
    return {"user_id": user_id, "email": email, "access_token": token}


def sign_in(email: str, password: str) -> dict:
    """Authenticate an existing user. Returns user info + JWT."""
    email = email.strip().lower()
    users = _load_users()

    user = users.get(email)
    if not user:
        raise ValueError("Invalid email or password")

    if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        raise ValueError("Invalid email or password")

    token = _create_token(user["user_id"], email)
    return {"user_id": user["user_id"], "email": email, "access_token": token}


def verify_token(token: str) -> dict | None:
    """Decode and verify a JWT. Returns user payload or None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": payload["sub"], "email": payload["email"]}
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ─── Internal ──────────────────────────────────────────────────────────────────

def _create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
