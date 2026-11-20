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
from lumpycare.database import analytics_activity, init_db, save_scan
from lumpycare.dashboard import render_dashboard

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
                        source="streamlit",
                        scanned_at=scanned_at.isoformat(),
                    )
                    record = {
                        "Scan ID": scan_id,
                        "Time (UTC)": scanned_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "Image": source_name,
                        "Prediction": result["prediction"],
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
        "Cattle Profiles": ("🐄", "Cattle profiles", "Register cattle, record breed and age, and connect every screening to an animal. Persistent profiles arrive with the database phase."),
        "Outbreak Map": ("📍", "Outbreak intelligence", "Location-based case mapping will be enabled after location capture and persistent scan records are added. No location data is currently stored."),
        "Reports": ("📄", "Screening reports", "Downloadable reports and herd summaries will be added after persistent scan storage. Single-scan CSV export remains available under New Scan."),
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
