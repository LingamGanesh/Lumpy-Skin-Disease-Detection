from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from lumpycare.model import CLASSES, ModelBundle, predict_rgb, train_model
from lumpycare.ui import inject_theme, render_sidebar, render_topbar
from lumpycare.database import analytics_activity, init_db, save_scan, list_cattle, get_cattle
from lumpycare.dashboard import render_dashboard
from lumpycare.care import render_care_guidance, render_hospital_workspace, render_appointment_booking, render_doctor_portal, render_hospital_admin_portal
from lumpycare.auth import init_auth_db, bootstrap_environment_users
from lumpycare.cattle import render_cattle_profiles

st.set_page_config(
    page_title="LumpyCare | Cattle Skin Screening",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme(st.session_state.get("lc_dark_mode", False))


@st.cache_resource(show_spinner="Preparing the SVM screening model…")
def get_model() -> ModelBundle:
    return train_model()


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def as_rgb(uploaded_file) :
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    return image


page = render_sidebar()
render_topbar(page)

st.markdown(
    '<div class="hero"><div class="eyebrow">AI-assisted cattle skin screening</div>'
    '<h1>🐄 LumpyCare</h1><p>Image-based screening support for early awareness of Lumpy Skin Disease.</p></div>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

try:
    bundle = get_model()
except Exception as exc:
    st.error("The screening model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Provide both labelled folders under SkinDataset/ (Lumpy skin and Normal skin), "
        "or the existing features/features.txt.npy and features/labels.txt.npy arrays."
    )
    st.stop()

metrics = bundle.metrics
init_db()
init_auth_db()
bootstrap_environment_users()

if page == "Dashboard":
    render_dashboard()

elif page == "New Scan":
    st.subheader("Image screening")
    st.markdown(
        '<p class="muted">Use a clear, well-lit close-up. Camera capture requires browser permission and a secure context (HTTPS or localhost).</p>',
        unsafe_allow_html=True,
    )
    upload_tab, camera_tab = st.tabs(["Upload image", "Use camera"])
    image = None
    source_name = ""
    with upload_tab:
        uploaded = st.file_uploader(
            "Choose a cattle skin image", type=["jpg", "jpeg", "png", "webp"], key="upload_image"
        )
        if uploaded is not None:
            try:
                image = as_rgb(uploaded)
                source_name = uploaded.name
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be decoded as an image.")
    with camera_tab:
        camera_file = st.camera_input("Take a photo", key="camera_image")
        if camera_file is not None:
            try:
                image = as_rgb(camera_file)
                source_name = "camera_capture.jpg"
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("The captured image could not be decoded.")

    if image is not None:
        cattle_profiles = list_cattle()
        cattle_choices = {"Not linked to a registered animal": None}
        cattle_choices.update({f"{item['cattle_id']} — {item['name']}": item["cattle_id"] for item in cattle_profiles})
        selected_cattle_label = st.selectbox("Link this screening to a cattle profile (optional)", list(cattle_choices))
        selected_cattle_id = cattle_choices[selected_cattle_label]
        selected_cattle = get_cattle(selected_cattle_id) if selected_cattle_id else None
        preview_col, result_col = st.columns([1.05, 1])
        with preview_col:
            st.image(image, caption="Image ready for screening", use_container_width=True)
        with result_col:
            st.markdown("### Prediction")
            if st.button("Analyse image", type="primary", use_container_width=True):
                try:
                    result = predict_rgb(bundle, np.asarray(image))
                    scanned_at = datetime.now(timezone.utc)
                    scan_id = save_scan(
                        prediction=result["prediction"],
                        confidence=result["confidence"],
                        image_name=source_name,
                        cattle_id=selected_cattle_id,
                        location=(selected_cattle or {}).get("village") or None,
                        latitude=(selected_cattle or {}).get("latitude"),
                        longitude=(selected_cattle or {}).get("longitude"),
                        source="streamlit",
                        scanned_at=scanned_at.isoformat(),
                    )
                    record = {
                        "Scan ID": scan_id,
                        "Time (UTC)": scanned_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Image": source_name,
                        "Prediction": result["prediction"],
                        "Cattle ID": selected_cattle_id or "",
                        "Confidence (%)": round(result["confidence"] * 100, 2),
                    }
                    st.session_state.latest_result = record
                    st.session_state.history.insert(0, record)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            result = st.session_state.latest_result
            if result:
                if result["Prediction"] == CLASSES[0]:
                    st.error("Model result: Lumpy skin")
                    render_care_guidance()
                    st.info("For veterinary help, use Hospitals & Doctors to search the registered directory or request a visit under Appointments.")
                else:
                    st.success("Model result: Normal skin")
                st.metric("Model probability estimate", f"{result['Confidence (%)']:.1f}%")
                st.progress(min(max(result["Confidence (%)"] / 100, 0), 1))
                st.caption("This score is not a calibrated clinical certainty.")
                st.download_button(
                    "Download result CSV",
                    data=pd.DataFrame([result]).to_csv(index=False).encode("utf-8"),
                    file_name="lumpycare_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.info("Upload a photo or take one with your camera to begin.")

elif page == "Care & First Aid":
    render_care_guidance()

elif page == "Hospitals & Doctors":
    render_hospital_workspace()

elif page == "Appointments":
    render_appointment_booking()

elif page == "Doctor Portal":
    render_doctor_portal()

elif page == "Hospital Admin":
    render_hospital_admin_portal()

elif page == "Cattle Profiles":
    render_cattle_profiles()

elif page == "Outbreak Map":
    st.subheader("📍 Field case map")
    records = analytics_activity(limit=1000, include_demo=False)
    cases = [r for r in records if r.get("prediction") == "Lumpy skin" and r.get("latitude") is not None and r.get("longitude") is not None]
    st.caption("Only real, non-demo positive screening records with coordinates saved on the linked cattle profile are plotted. A screening result is not a confirmed outbreak report.")
    if cases:
        map_data = pd.DataFrame([{"latitude": r["latitude"], "longitude": r["longitude"]} for r in cases])
        st.metric("Mapped positive screening records", len(cases))
        st.map(map_data, size=18)
        st.dataframe(pd.DataFrame([{"Scan ID":r["id"],"Cattle ID":r.get("cattle_id"),"Location":r.get("location"),"Scanned at":r["scanned_at"]} for r in cases]), hide_index=True, use_container_width=True)
    else:
        st.info("No positive scans with saved GPS coordinates yet. Add latitude and longitude to cattle profiles, then link new scans to those animals.")

elif page == "Reports":
    st.subheader("📄 Screening reports")
    records = analytics_activity(limit=1000, include_demo=False)
    if records:
        report = pd.DataFrame(records)
        infected = int((report["prediction"] == "Lumpy skin").sum())
        healthy = int((report["prediction"] == "Normal skin").sum())
        m1,m2,m3 = st.columns(3)
        m1.metric("Real screening records", len(report))
        m2.metric("Lumpy skin flags", infected)
        m3.metric("Normal skin flags", healthy)
        st.dataframe(report, hide_index=True, use_container_width=True)
        st.download_button("Download screening report CSV", report.to_csv(index=False).encode("utf-8"), "lumpycare_report.csv", "text/csv")
    else:
        st.info("No real screening records available to report. Synthetic demo records are excluded.")

elif page == "Model Insights":
    st.subheader("Model evaluation")
    st.caption(
        f"Source: {bundle.source}. The classifier is fit only on the training split; "
        "the displayed metrics are calculated on the held-out test split."
    )
    cols = st.columns(3)
    for col, (label, value) in zip(
        cols,
        [
            ("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%"),
            ("Training samples", str(metrics["train_count"])),
            ("Test samples", str(metrics["test_count"])),
        ],
    ):
        with col:
            metric_card(label, value)
    st.write("")
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    heat = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], labels=CLASSES)
    ax.set_yticks([0, 1], labels=CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion matrix")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=13)
    fig.colorbar(heat, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame(metrics["classification_report"]).T.round(3), use_container_width=True)
    st.markdown("#### Class distribution")
    st.dataframe(pd.DataFrame(
        [{"Class": name, "Images": count} for name, count in metrics["class_counts"].items()]
    ), hide_index=True, use_container_width=True)

elif page == "History":
    st.subheader("Saved screening history")
    st.caption("This table reads persistent SQLite records. Synthetic demo records are excluded.")
    records = analytics_activity(limit=200, include_demo=False)
    if records:
        history = pd.DataFrame(records)
        display_columns = ["id", "scanned_at", "image_name", "prediction", "confidence", "cattle_id", "location"]
        history = history[[name for name in display_columns if name in history.columns]]
        history = history.rename(columns={
            "id": "Scan ID", "scanned_at": "Scanned at (UTC)", "image_name": "Image",
            "prediction": "Prediction", "confidence": "Confidence",
            "cattle_id": "Cattle ID", "location": "Location",
        })
        if "Confidence" in history:
            history["Confidence"] = (history["Confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(history, hide_index=True, use_container_width=True)
        export = pd.DataFrame(records)
        st.download_button(
            "Export saved history CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="lumpycare_saved_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No real screening records are saved yet. Run an image screening to create the first record.")


else:
    future_pages = {
        "Settings": ("⚙️", "Workspace settings", "Language, alert thresholds, account roles, and data-retention controls will be introduced in later phases."),
    }
    icon, title, description = future_pages[page]
    st.markdown(
        f'<div class="panel"><div style="font-size:2rem">{icon}</div>'
        f'<h2 style="color:var(--lc-ink)">{title}</h2><p class="muted">{description}</p>'
        '<p class="muted"><b>Implementation status:</b> planned — no demo data is shown as real data.</p></div>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    '<p class="muted">⚕️ <b>Important:</b> LumpyCare is an educational research prototype, not a veterinary diagnostic device. '
    'Image quality, dataset size, class balance and distribution shift can affect predictions. '
    'Contact a qualified veterinarian for examination, diagnosis and treatment decisions.</p>',
    unsafe_allow_html=True,
)
