import pytest

import lumpycare.auth as auth


def test_password_hash_and_role_authentication(monkeypatch, tmp_path):
    import lumpycare.database as database
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "auth_test.db")
    auth.init_auth_db()
    admin_id = auth.create_user("admin", "Hospital Admin", "admin", "very-long-admin-password")
    doctor_id = auth.create_user("doctor", "Dr. Rao", "doctor", "very-long-doctor-password", 7)
    assert admin_id != doctor_id
    admin = auth.authenticate_user("ADMIN", "very-long-admin-password")
    doctor = auth.authenticate_user("doctor", "very-long-doctor-password")
    assert admin["role"] == "admin"
    assert doctor["role"] == "doctor"
    assert doctor["hospital_id"] == 7
    assert auth.authenticate_user("doctor", "incorrect-password") is None
    assert auth.authenticate_user("missing", "incorrect-password") is None
    assert "password_hash" not in auth.list_users()[0]


def test_password_policy_and_doctor_hospital_assignment(monkeypatch, tmp_path):
    import lumpycare.database as database
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "auth_policy.db")
    auth.init_auth_db()
    with pytest.raises(ValueError):
        auth.create_user("short", "Short Password", "admin", "tiny")
    with pytest.raises(ValueError):
        auth.create_user("doctor", "Doctor", "doctor", "long-enough-password")
    with pytest.raises(ValueError):
        auth.create_user("badrole", "Bad Role", "owner", "long-enough-password")


def test_deactivated_user_cannot_login(monkeypatch, tmp_path):
    import lumpycare.database as database
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "auth_active.db")
    user_id = auth.create_user("doctor", "Dr. Rao", "doctor", "long-enough-password", 3)
    assert auth.authenticate_user("doctor", "long-enough-password")
    auth.set_user_active(user_id, False)
    assert auth.authenticate_user("doctor", "long-enough-password") is None
