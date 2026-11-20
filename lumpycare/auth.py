"""Local prototype authentication with salted PBKDF2 password hashes."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from lumpycare.database import connect, init_db

ITERATIONS = 310_000


def init_auth_db() -> None:
    init_db()
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','doctor')),
                hospital_id INTEGER,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS).hex()


def create_user(username: str, full_name: str, role: str, password: str, hospital_id: int | None = None) -> int:
    init_auth_db()
    username, full_name = username.strip(), full_name.strip()
    if not username or not full_name:
        raise ValueError("Username and full name are required.")
    if role not in {"admin", "doctor"}:
        raise ValueError("Role must be admin or doctor.")
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if role == "doctor" and hospital_id is None:
        raise ValueError("A doctor account must be assigned to a hospital.")
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    with connect() as con:
        cur = con.execute(
            """INSERT INTO users(username,full_name,role,hospital_id,password_salt,password_hash)
               VALUES(?,?,?,?,?,?)""",
            (username, full_name, role, hospital_id, salt.hex(), password_hash),
        )
        return int(cur.lastrowid)


def authenticate_user(username: str, password: str) -> dict | None:
    init_auth_db()
    with connect() as con:
        row = con.execute(
            "SELECT * FROM users WHERE username=? COLLATE NOCASE AND active=1",
            (username.strip(),),
        ).fetchone()
    if not row:
        # Perform a dummy hash to reduce obvious username timing differences.
        _hash_password(password, bytes(16))
        return None
    candidate = _hash_password(password, bytes.fromhex(row["password_salt"]))
    if not hmac.compare_digest(candidate, row["password_hash"]):
        return None
    return {
        "id": row["id"], "username": row["username"], "full_name": row["full_name"],
        "role": row["role"], "hospital_id": row["hospital_id"],
    }


def list_users() -> list[dict]:
    init_auth_db()
    with connect() as con:
        rows = con.execute(
            """SELECT id,username,full_name,role,hospital_id,active,created_at
               FROM users ORDER BY role,full_name"""
        ).fetchall()
    return [dict(r) for r in rows]


def set_user_active(user_id: int, active: bool) -> None:
    init_auth_db()
    with connect() as con:
        con.execute("UPDATE users SET active=? WHERE id=?", (int(active), user_id))


def bootstrap_environment_users() -> None:
    """Create initial accounts once, only when their environment secrets are supplied."""
    init_auth_db()
    with connect() as con:
        count = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count:
        return
    admin_user = os.getenv("LUMPYCARE_ADMIN_USERNAME", "")
    admin_pass = os.getenv("LUMPYCARE_ADMIN_PASSWORD", "")
    if admin_user and admin_pass:
        create_user(admin_user, os.getenv("LUMPYCARE_ADMIN_NAME", "Hospital Administrator"), "admin", admin_pass)
    doctor_user = os.getenv("LUMPYCARE_DOCTOR_USERNAME", "")
    doctor_pass = os.getenv("LUMPYCARE_DOCTOR_PASSWORD", "")
    doctor_hospital = os.getenv("LUMPYCARE_DOCTOR_HOSPITAL_ID", "")
    if doctor_user and doctor_pass and doctor_hospital.isdigit():
        create_user(doctor_user, os.getenv("LUMPYCARE_DOCTOR_NAME", "Veterinary Doctor"), "doctor", doctor_pass, int(doctor_hospital))
