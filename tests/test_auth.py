import lumpycare.auth as auth


def test_doctor_login_accepts_configured_credentials(monkeypatch):
    monkeypatch.setenv("LUMPYCARE_DOCTOR_USERNAME", "doctor1")
    monkeypatch.setenv("LUMPYCARE_DOCTOR_PASSWORD", "a-long-test-password")
    assert auth.doctor_credentials_configured()
    assert auth.verify_doctor_credentials("doctor1", "a-long-test-password")


def test_doctor_login_rejects_wrong_or_missing_credentials(monkeypatch):
    monkeypatch.setenv("LUMPYCARE_DOCTOR_USERNAME", "doctor1")
    monkeypatch.setenv("LUMPYCARE_DOCTOR_PASSWORD", "a-long-test-password")
    assert not auth.verify_doctor_credentials("doctor1", "wrong")
    assert not auth.verify_doctor_credentials("other", "a-long-test-password")
    monkeypatch.delenv("LUMPYCARE_DOCTOR_PASSWORD")
    assert not auth.doctor_credentials_configured()
    assert not auth.verify_doctor_credentials("doctor1", "a-long-test-password")
