from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from typing import Mapping


SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32


@dataclass(frozen=True)
class AccessPolicy:
    required: bool
    configured: bool
    token_env: str


def access_policy(
    require_access_token: bool,
    token_env: str,
    environ: Mapping[str, str] | None = None,
) -> AccessPolicy:
    source = os.environ if environ is None else environ
    token = source.get(token_env, "").strip()
    return AccessPolicy(
        required=require_access_token,
        configured=bool(token),
        token_env=token_env,
    )


def expected_access_token(token_env: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return source.get(token_env, "").strip()


def verify_access_token(expected: str, provided: str) -> bool:
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_secret() -> str:
    """Return a 256-bit URL-safe secret."""
    return secrets.token_urlsafe(32)


def _hash_secret(secret: str, salt: bytes | None = None) -> tuple[str, str]:
    if not secret:
        raise ValueError("Secret must not be empty.")
    actual_salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        secret.encode("utf-8"),
        salt=actual_salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
    )
    return actual_salt.hex(), digest.hex()


def _verify_secret(secret: str, salt_hex: str, digest_hex: str) -> bool:
    if not secret or not salt_hex or not digest_hex:
        return False
    try:
        _, candidate = _hash_secret(secret, bytes.fromhex(salt_hex))
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError):
        return False


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    return _hash_secret(password, salt)


def verify_password(password: str, salt_hex: str, digest_hex: str) -> bool:
    return _verify_secret(password, salt_hex, digest_hex)


def hash_session_secret(secret: str, salt: bytes | None = None) -> tuple[str, str]:
    return _hash_secret(secret, salt)


def verify_session_secret(secret: str, salt_hex: str, digest_hex: str) -> bool:
    return _verify_secret(secret, salt_hex, digest_hex)


# Kept for backwards compatibility with databases created before account login.
def hash_story_secret(secret: str, salt: bytes | None = None) -> tuple[str, str]:
    return _hash_secret(secret, salt)


def verify_story_secret(secret: str, salt_hex: str, digest_hex: str) -> bool:
    return _verify_secret(secret, salt_hex, digest_hex)


def build_story_key(story_id: str, secret: str) -> str:
    if not story_id or not secret:
        raise ValueError("Story id and secret are required.")
    return f"{story_id}.{secret}"


def split_story_key(story_key: str) -> tuple[str, str]:
    story_id, separator, secret = story_key.strip().partition(".")
    if not separator or not story_id or not secret:
        raise ValueError("작품키 형식이 올바르지 않아.")
    return story_id, secret


def build_session_token(session_id: str, secret: str) -> str:
    if not session_id or not secret:
        raise ValueError("Session id and secret are required.")
    return f"{session_id}.{secret}"


def split_session_token(token: str) -> tuple[str, str]:
    session_id, separator, secret = token.strip().partition(".")
    if not separator or not session_id or not secret:
        raise ValueError("Invalid session token.")
    return session_id, secret
