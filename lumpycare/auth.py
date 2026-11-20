"""Environment-configured doctor portal authentication."""
from __future__ import annotations

import hmac
import os


def doctor_credentials_configured() -> bool:
    return bool(os.getenv("LUMPYCARE_DOCTOR_USERNAME") and os.getenv("LUMPYCARE_DOCTOR_PASSWORD"))


def verify_doctor_credentials(username: str, password: str) -> bool:
    expected_username = os.getenv("LUMPYCARE_DOCTOR_USERNAME", "")
    expected_password = os.getenv("LUMPYCARE_DOCTOR_PASSWORD", "")
    if not expected_username or not expected_password:
        return False
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(password, expected_password)
