"""Authentication routes for NOVA-C."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional

from ..services.auth import sign_up, sign_in, verify_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Request / Response Models ────────────────────────────────────────────────

class AuthRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    access_token: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str
    email: str


# ─── Auth dependency ──────────────────────────────────────────────────────────

async def require_auth(authorization: str = Header(...)) -> dict:
    """Extract Bearer token and validate JWT."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")
    token = authorization[7:]
    user = verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/signup", response_model=AuthResponse)
async def signup(req: AuthRequest):
    """Register a new user with email and password."""
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    try:
        result = sign_up(req.email, req.password)
        return AuthResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/login", response_model=AuthResponse)
async def login(req: AuthRequest):
    """Sign in with email and password."""
    try:
        result = sign_in(req.email, req.password)
        return AuthResponse(**result)
    except ValueError:
        raise HTTPException(401, "Invalid email or password")


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(require_auth)):
    """Get the current authenticated user."""
    return UserResponse(**user)
