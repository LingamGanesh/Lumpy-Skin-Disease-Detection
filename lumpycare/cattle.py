"""Cattle registry UI backed by the shared SQLite database."""
import pandas as pd
import streamlit as st
from lumpycare.database import create_cattle, list_cattle


def render_cattle_profiles():
    st.subheader("🐄 Cattle registry")
    st.caption("Register each animal once, then link screening records to its cattle ID.")
    with st.form("cattle_profile_form"):
        left, right = st.columns(2)
        cattle_id = left.text_input("Unique cattle ID *")
        name = right.text_input("Cattle name *")
        age = left.number_input("Age (years)", min_value=0.0, max_value=40.0, value=2.0, step=0.5)
        breed = right.text_input("Breed")
        village = left.text_input("Village / locality")
        latitude = right.number_input("Latitude (optional)", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f")
        longitude = left.number_input("Longitude (optional)", min_value=-180.0, max_value=180.0, value=0.0, format="%.6f")
        submitted = st.form_submit_button("Save cattle profile", type="primary")
        if submitted:
            if not cattle_id.strip() or not name.strip():
                st.error("Cattle ID and name are required.")
            else:
                try:
                    create_cattle(
                        cattle_id, name, age, breed, village,
                        latitude if latitude != 0 else None,
                        longitude if longitude != 0 else None,
                    )
                    st.success("Cattle profile saved.")
                    st.rerun()
                except Exception as exc:
                    st.error("Could not save this profile. Check that the cattle ID is unique. " + str(exc))

    st.divider()
    query = st.text_input("Search by cattle ID, name, breed or village", key="cattle_search").strip().lower()
    records = list_cattle()
    visible = [r for r in records if not query or query in " ".join(str(r.get(k) or "") for k in ("cattle_id","name","breed","village")).lower()]
    st.metric("Registered cattle", len(records))
    if visible:
        st.dataframe(pd.DataFrame([
            {
                "Cattle ID": r["cattle_id"], "Name": r["name"], "Age (years)": r["age_years"],
                "Breed": r["breed"], "Village": r["village"], "Screenings": r["scan_count"],
                "GPS saved": bool(r["latitude"] is not None and r["longitude"] is not None),
            } for r in visible
        ]), hide_index=True, use_container_width=True)
    else:
        st.info("No cattle profiles match your search.")
