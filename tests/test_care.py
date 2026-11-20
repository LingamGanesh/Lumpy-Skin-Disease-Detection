import lumpycare.database as database
import lumpycare.care as care

def test_hospital_directory_and_appointment_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "care_test.db")
    care.init_care_db()
    care.add_hospital(
        "District Veterinary Hospital", "Dr. Rao", "9876543210",
        "Main Road, Example Town", "Example Town", "Cattle consultation"
    )
    hospitals = care.list_hospitals("Example")
    assert len(hospitals) == 1
    assert hospitals[0]["name"] == "District Veterinary Hospital"

    care.create_appointment(
        hospitals[0]["id"], "Farm Owner", "9876543210", "Cow-17",
        "2026-12-01", "10:00", "Skin nodules and fever"
    )
    appointments = care.list_appointments()
    assert len(appointments) == 1
    assert appointments[0]["status"] == "Requested"
    assert appointments[0]["hospital_name"] == "District Veterinary Hospital"

    care.update_appointment_status(appointments[0]["id"], "Confirmed")
    assert care.list_appointments()[0]["status"] == "Confirmed"

def test_appointment_status_rejects_unknown_value():
    import pytest
    with pytest.raises(ValueError):
        care.update_appointment_status(1, "Approved")
