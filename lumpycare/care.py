"""Veterinary guidance, hospital directory and appointment workflow."""
from datetime import datetime, timezone
from urllib.parse import quote_plus
import streamlit as st
from lumpycare.database import connect, init_db
from lumpycare.auth import authenticate_user, create_user, list_users, set_user_active

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

def list_appointments(hospital_id=None):
    init_care_db()
    with connect() as con:
        if hospital_id is None:
            rows=con.execute("SELECT a.*,h.name hospital_name,h.doctor_name FROM appointments a JOIN hospitals h ON h.id=a.hospital_id ORDER BY a.preferred_date,a.preferred_time,a.id DESC").fetchall()
        else:
            rows=con.execute("SELECT a.*,h.name hospital_name,h.doctor_name FROM appointments a JOIN hospitals h ON h.id=a.hospital_id WHERE a.hospital_id=? ORDER BY a.preferred_date,a.preferred_time,a.id DESC",(hospital_id,)).fetchall()
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
    is_admin = st.session_state.get("lc_user", {}).get("role") == "admin"
    if is_admin:
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
    else:
        st.caption("Only a signed-in Hospital Admin can register clinics. You can still search the directory below.")
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

def _render_login(required_role: str):
    user = st.session_state.get("lc_user")
    if user:
        if user.get("role") == required_role:
            return user
        st.error("This account does not have permission to open this workspace.")
        return None
    st.markdown("Sign in with your hospital account. Passwords are stored as salted PBKDF2 hashes.")
    with st.form("lumpycare_login_" + required_role):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        user = authenticate_user(username, password)
        if user and user["role"] == required_role:
            st.session_state["lc_user"] = user
            st.rerun()
        st.error("Invalid credentials or account role.")
    return None


def _render_workspace_header(user):
    left, right = st.columns([4, 1])
    left.success("Signed in: " + user["full_name"])
    if right.button("Logout", key="lumpycare_logout", use_container_width=True):
        st.session_state.pop("lc_user", None)
        st.rerun()


def render_doctor_portal():
    st.subheader("🩺 Doctor workspace")
    user = _render_login("doctor")
    if not user:
        return
    _render_workspace_header(user)
    hospital_id = user.get("hospital_id")
    if hospital_id is None:
        st.error("This doctor account has no hospital assignment. Contact the Hospital Admin.")
        return
    appointments = list_appointments(hospital_id=hospital_id)
    requested = sum(a["status"] == "Requested" for a in appointments)
    confirmed = sum(a["status"] == "Confirmed" for a in appointments)
    completed = sum(a["status"] == "Completed" for a in appointments)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total", len(appointments))
    k2.metric("Pending", requested)
    k3.metric("Confirmed", confirmed)
    k4.metric("Completed", completed)
    st.caption("Only appointments for your assigned hospital are shown. Confirm the appointment with the owner through the clinic's normal contact process.")
    query = st.text_input("Search owner, cattle ID/name or phone", key="doctor_appointment_search").strip().lower()
    status_filter = st.selectbox("Filter status", ["All", "Requested", "Confirmed", "Completed", "Cancelled"], key="doctor_status_filter")
    visible = [a for a in appointments if (status_filter == "All" or a["status"] == status_filter) and (not query or query in " ".join(str(a[k]) for k in ("owner_name","cattle_name","phone","reason")).lower())]
    if not visible:
        st.info("No matching appointment requests.")
    for item in visible:
        with st.container(border=True):
            st.markdown("**Appointment #" + str(item["id"]) + " · " + item["status"] + "** — " + item["preferred_date"] + " at " + item["preferred_time"])
            st.write("Owner: " + item["owner_name"] + " · Phone: " + item["phone"] + " · Cattle: " + item["cattle_name"])
            if item["reason"]:
                st.caption("Reported symptoms: " + item["reason"])
            statuses = ["Requested", "Confirmed", "Completed", "Cancelled"]
            status = st.selectbox("Appointment status", statuses, index=statuses.index(item["status"]), key="doctor_appt_status_" + str(item["id"]))
            if st.button("Save status", key="doctor_save_appt_" + str(item["id"])):
                update_appointment_status(item["id"], status)
                st.success("Appointment status updated.")
                st.rerun()


def render_hospital_admin_portal():
    st.subheader("🏥 Hospital administration")
    user = _render_login("admin")
    if not user:
        return
    _render_workspace_header(user)
    hospitals = list_hospitals()
    appointments = list_appointments()
    pending = sum(a["status"] == "Requested" for a in appointments)
    a,b,d = st.columns(3)
    a.metric("Registered hospitals", len(hospitals))
    b.metric("Total appointment requests", len(appointments))
    d.metric("Pending requests", pending)
    st.markdown("### Hospital directory")
    for h in hospitals:
        st.write("**" + h["name"] + "** · " + h["locality"] + " · " + (h["phone"] or "No phone"))
    st.markdown("### Create a doctor account")
    if not hospitals:
        st.warning("Register a hospital in Hospitals & Doctors before creating doctor accounts.")
    else:
        choices = {f'{h["name"]} — {h["locality"]} (ID {h["id"]})': h["id"] for h in hospitals}
        with st.form("admin_create_doctor"):
            full_name = st.text_input("Doctor full name")
            username = st.text_input("Doctor username")
            password = st.text_input("Temporary password (minimum 12 characters)", type="password")
            hospital_label = st.selectbox("Assign hospital", list(choices))
            submitted = st.form_submit_button("Create doctor account", type="primary")
            if submitted:
                try:
                    create_user(username, full_name, "doctor", password, choices[hospital_label])
                    st.success("Doctor account created. Share credentials securely with the doctor.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    st.markdown("### User access")
    users = list_users()
    if users:
        st.dataframe([{k: u[k] for k in ("id","username","full_name","role","hospital_id","active")} for u in users], hide_index=True, use_container_width=True)
        for account in users:
            if account["id"] != user["id"]:
                label = "Deactivate " if account["active"] else "Reactivate "
                if st.button(label + account["username"], key="toggle_user_" + str(account["id"])):
                    set_user_active(account["id"], not bool(account["active"]))
                    st.rerun()
