"""Veterinary guidance, hospital directory and appointment workflow."""
from datetime import datetime, timezone
from urllib.parse import quote_plus
import streamlit as st
from lumpycare.database import connect, init_db

def init_care_db():
    init_db()
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS hospitals (
          id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
          doctor_name TEXT NOT NULL DEFAULT '', phone TEXT NOT NULL DEFAULT '',
          address TEXT NOT NULL, locality TEXT NOT NULL DEFAULT '',
          services TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS appointments (
          id INTEGER PRIMARY KEY AUTOINCREMENT, hospital_id INTEGER NOT NULL,
          owner_name TEXT NOT NULL, phone TEXT NOT NULL, cattle_name TEXT NOT NULL,
          preferred_date TEXT NOT NULL, preferred_time TEXT NOT NULL,
          reason TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'Requested',
          created_at TEXT NOT NULL, FOREIGN KEY(hospital_id) REFERENCES hospitals(id));
        """)

def list_hospitals(locality=""):
    init_care_db()
    with connect() as con:
        if locality.strip():
            rows=con.execute("SELECT * FROM hospitals WHERE locality LIKE ? OR address LIKE ? ORDER BY name",(f"%{locality.strip()}%",f"%{locality.strip()}%")).fetchall()
        else:
            rows=con.execute("SELECT * FROM hospitals ORDER BY name").fetchall()
    return [dict(r) for r in rows]

def add_hospital(name,doctor,phone,address,locality,services):
    init_care_db()
    with connect() as con:
        con.execute("INSERT INTO hospitals(name,doctor_name,phone,address,locality,services,created_at) VALUES(?,?,?,?,?,?,?)",(name.strip(),doctor.strip(),phone.strip(),address.strip(),locality.strip(),services.strip(),datetime.now(timezone.utc).isoformat()))

def create_appointment(hospital_id,owner,phone,cattle,date,time,reason):
    init_care_db()
    with connect() as con:
        con.execute("INSERT INTO appointments(hospital_id,owner_name,phone,cattle_name,preferred_date,preferred_time,reason,created_at) VALUES(?,?,?,?,?,?,?,?)",(hospital_id,owner.strip(),phone.strip(),cattle.strip(),date,time,reason.strip(),datetime.now(timezone.utc).isoformat()))

def list_appointments():
    init_care_db()
    with connect() as con:
        rows=con.execute("SELECT a.*,h.name hospital_name,h.doctor_name FROM appointments a JOIN hospitals h ON h.id=a.hospital_id ORDER BY a.preferred_date,a.preferred_time,a.id DESC").fetchall()
    return [dict(r) for r in rows]

def update_appointment_status(appointment_id,status):
    if status not in {"Requested","Confirmed","Completed","Cancelled"}: raise ValueError("Invalid appointment status")
    with connect() as con: con.execute("UPDATE appointments SET status=? WHERE id=?",(status,appointment_id))

def render_care_guidance():
    st.markdown("### 🩺 Immediate care & first aid")
    st.warning("An image-screening result is not a confirmed diagnosis. Contact a qualified veterinarian promptly, especially if the animal has fever, painful nodules, discharge, weakness or stops eating.")
    st.markdown("**Safe steps while arranging veterinary help**")
    steps=[
      "Separate the affected animal from healthy cattle; do not share feeders, water troughs or equipment.",
      "Keep it in a clean, dry, shaded and well-ventilated area with access to clean water.",
      "Offer its usual feed and water if it can eat and drink comfortably; do not force-feed.",
      "Wear gloves when handling lesions or secretions; wash hands and disinfect equipment.",
      "Reduce flies, mosquitoes and ticks using veterinarian-approved vector-control practices.",
      "Record symptoms, onset date, temperature if safely measurable, and recent animal movements.",
      "Do not squeeze or puncture nodules or apply kerosene, engine oil, acids or unverified home remedies."
    ]
    for step in steps: st.markdown("- "+step)
    st.markdown("### 💊 Veterinary medicines — educational examples only")
    st.info("This app does not prescribe medicines. A veterinarian must examine the animal and decide whether medicine is needed, including product, dose, route, duration and milk/meat withdrawal period.")
    st.markdown("| Category / example | Possible veterinary use |\n|---|---|\n| Veterinary anti-inflammatory / fever medicine | May be considered for pain or fever after checking hydration and other risks. |\n| Antibiotics such as oxytetracycline | Only for a diagnosed or suspected secondary bacterial infection; not a cure for the LSD virus. |\n| Veterinary antiseptic wound-care products | Cleaning selected lesions under veterinary direction. |\n| Fluids and electrolytes | Support for dehydration or poor intake after assessment. |")
    st.caption("Never self-administer antibiotics, steroids, injections, painkillers or human medicines. Follow the veterinarian's withdrawal-period instructions.")

def render_hospital_workspace():
    init_care_db()
    st.subheader("🏥 Veterinary hospital directory")
    st.caption("Register verified clinics. Search uses the locality/address saved here; it is not live GPS-based nearest-hospital detection.")
    with st.form("hospital_add_form"):
        st.markdown("**Register a veterinary hospital / clinic**")
        c1,c2=st.columns(2)
        name=c1.text_input("Hospital / clinic name")
        doctor=c2.text_input("Veterinarian / doctor name")
        phone=c1.text_input("Contact phone")
        locality=c2.text_input("Village / locality / town")
        address=st.text_area("Full address")
        services=st.text_input("Services",value="Veterinary consultation, cattle care")
        submitted=st.form_submit_button("Add hospital",type="primary")
        if submitted:
            if not name.strip() or not address.strip(): st.error("Hospital name and full address are required.")
            else:
                add_hospital(name,doctor,phone,address,locality,services)
                st.success("Hospital added to this local directory.")
                st.rerun()
    st.divider()
    query=st.text_input("Find hospitals by village / locality / town",key="hospital_search")
    hospitals=list_hospitals(query)
    if not hospitals: st.info("No matching hospital is registered yet. Add a verified hospital above.")
    for h in hospitals:
        with st.container(border=True):
            st.markdown("**"+h["name"]+"**")
            st.write("👨‍⚕️ "+(h["doctor_name"] or "Doctor details not provided"))
            st.write("📍 "+h["address"]+" · "+h["locality"])
            st.write("☎️ "+(h["phone"] or "Phone not provided"))
            st.caption(h["services"])
            url="https://www.google.com/maps/search/?api=1&query="+quote_plus(h["name"]+" "+h["address"])
            st.link_button("Open directions in Google Maps",url)

def render_appointment_booking():
    st.subheader("📅 Request a veterinary appointment")
    hospitals=list_hospitals()
    if not hospitals:
        st.info("No hospitals are registered yet. Open Hospitals & Doctors and add a verified clinic first.")
        return
    options={h["name"]+" — "+h["locality"]:h["id"] for h in hospitals}
    with st.form("appointment_request_form"):
        selected=st.selectbox("Choose hospital",list(options))
        c1,c2=st.columns(2)
        owner=c1.text_input("Owner name")
        phone=c2.text_input("Contact phone")
        cattle=c1.text_input("Cattle name / ID")
        date=c2.date_input("Preferred date")
        time=c1.selectbox("Preferred time",["09:00","10:00","11:00","12:00","14:00","15:00","16:00","17:00"])
        reason=st.text_area("Symptoms / reason for visit")
        submitted=st.form_submit_button("Send appointment request",type="primary")
        if submitted:
            if not owner.strip() or not phone.strip() or not cattle.strip(): st.error("Owner name, phone and cattle name/ID are required.")
            else:
                create_appointment(options[selected],owner,phone,cattle,date.isoformat(),time,reason)
                st.success("Request saved. The clinic must contact you to confirm the appointment.")

def render_doctor_portal():
    """Show appointment management only after a doctor signs in."""
    import streamlit as st
    from lumpycare.auth import doctor_credentials_configured, verify_doctor_credentials

    st.subheader("🩺 Doctor portal")
    if not doctor_credentials_configured():
        st.error("Doctor login is not configured yet. Set LUMPYCARE_DOCTOR_USERNAME and LUMPYCARE_DOCTOR_PASSWORD in your environment, then restart Streamlit.")
        st.code('set LUMPYCARE_DOCTOR_USERNAME=doctor1\nset LUMPYCARE_DOCTOR_PASSWORD=choose-a-long-private-password\npython -m streamlit run app.py', language="bat")
        st.info("For a deployed app, configure these as private environment secrets. Do not commit passwords to GitHub.")
        return

    if not st.session_state.get("doctor_authenticated", False):
        st.markdown("Sign in with the clinic's private doctor account to view and manage appointment requests.")
        with st.form("doctor_login_form"):
            username = st.text_input("Doctor username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Doctor login", type="primary", use_container_width=True)
        if submitted:
            if verify_doctor_credentials(username.strip(), password):
                st.session_state["doctor_authenticated"] = True
                st.session_state["doctor_username"] = username.strip()
                st.rerun()
            else:
                st.error("Invalid doctor username or password.")
        return

    head_col, logout_col = st.columns([4, 1])
    with head_col:
        st.success("Signed in as " + st.session_state.get("doctor_username", "Doctor"))
    with logout_col:
        if st.button("Logout", use_container_width=True):
            st.session_state.pop("doctor_authenticated", None)
            st.session_state.pop("doctor_username", None)
            st.rerun()

    st.caption("Appointment requests from this app. Verify patient details directly with the clinic before confirming. Prototype only; not a secure clinical-record system.")
    appointments = list_appointments()
    if not appointments:
        st.info("No appointment requests have been submitted yet.")
        return
    for item in appointments:
        with st.container(border=True):
            st.markdown("**Appointment #" + str(item["id"]) + " · " + item["status"] + "** — " + item["preferred_date"] + " at " + item["preferred_time"])
            st.write("Hospital: " + item["hospital_name"] + " | Doctor: " + (item["doctor_name"] or "Not assigned"))
            st.write("Owner: " + item["owner_name"] + " · Phone: " + item["phone"] + " · Cattle: " + item["cattle_name"])
            if item["reason"]:
                st.caption("Reason: " + item["reason"])
            statuses = ["Requested", "Confirmed", "Completed", "Cancelled"]
            status = st.selectbox("Update status", statuses, index=statuses.index(item["status"]), key="doctor_appt_status_" + str(item["id"]))
            if st.button("Save status", key="doctor_save_appt_" + str(item["id"])):
                update_appointment_status(item["id"], status)
                st.success("Appointment status updated.")
                st.rerun()
