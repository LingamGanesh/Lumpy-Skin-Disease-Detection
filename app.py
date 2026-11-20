from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from lumpycare.model import CLASSES, ModelBundle, predict_rgb, train_model

st.set_page_config(
    page_title="LumpyCare | Cattle Skin Screening",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#163b30; --muted:#64776d; --line:#e2ebe5; --green:#176b52; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#f5f8f6; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(245,248,246,.94); }
.hero { padding:clamp(1.25rem,3vw,2.3rem); border-radius:22px; color:#fff;
  background:linear-gradient(120deg,#123c32,#176b52 62%,#2b8a65);
  box-shadow:0 12px 32px rgba(18,60,50,.13); margin-bottom:1.1rem; }
.hero .eyebrow { color:#b7e2cc; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; }
.hero h1 { color:#fff; font-family:Manrope,sans-serif; font-size:clamp(1.8rem,4vw,2.55rem); margin:.25rem 0; }
.hero p { color:#e0f1e8; margin:0; }
.metric-card,.panel { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }
.metric-card { min-height:102px; }
.metric-label { color:var(--muted); font-size:.82rem; font-weight:600; }
.metric-value { color:var(--ink); font-family:Manrope,sans-serif; font-size:1.65rem; font-weight:800; margin-top:.25rem; }
.muted { color:var(--muted); font-size:.9rem; }
div[data-testid="stFileUploader"],div[data-testid="stCameraInput"] { background:#fff; border:1px dashed #9bb9a8; border-radius:14px; padding:.65rem; }
.stButton>button,.stDownloadButton>button { border-radius:10px; font-weight:700; min-height:2.7rem; }
@media(max-width:640px) { .hero { border-radius:16px; } .metric-card { min-height:86px; } }
</style>
""", unsafe_allow_html=True)


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


with st.sidebar:
    st.markdown("## 🐄 LumpyCare")
    st.caption("Cattle skin image screening")
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Screen an image", "Model evaluation", "Screening history"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Streamlit · OpenCV · scikit-learn · FastAPI")
    st.caption("Research prototype — veterinary confirmation required.")

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

if page == "Overview":
    st.subheader("Screening workspace")
    cols = st.columns(4)
    cards = [
        ("Held-out test accuracy", f"{metrics['accuracy'] * 100:.1f}%"),
        ("Dataset images", f"{metrics['total_count']:,}"),
        ("Training images", f"{metrics['train_count']:,}"),
        ("Test images", f"{metrics['test_count']:,}"),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
            metric_card(label, value)
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Start a screening")
        st.markdown(
            "1. Choose **Screen an image** in the sidebar.\n"
            "2. Upload a JPG/PNG/WEBP photo or capture one with your device camera.\n"
            "3. Review the predicted class and model probability estimate.\n"
            "4. Export the result or review this session's history."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Model and data")
        st.write(f"**Classifier:** RBF-kernel Support Vector Machine")
        st.write(f"**Training data source:** {bundle.source}")
        st.write(f"**Image transform:** 128 × 128 RGB, six-cluster K-means")
        st.write(f"**Evaluation:** stratified 80/20 hold-out split")
        st.markdown("</div>", unsafe_allow_html=True)
    st.warning(
        "The accuracy shown is an internal hold-out result on this repository's dataset. "
        "It is not evidence of clinical performance or a guarantee for new farms, cameras, or cattle breeds."
    )

elif page == "Screen an image":
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
                    record = {
                        "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
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

elif page == "Model evaluation":
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

elif page == "Screening history":
    st.subheader("Session screening history")
    if st.session_state.history:
        history = pd.DataFrame(st.session_state.history)
        st.dataframe(history, hide_index=True, use_container_width=True)
        st.download_button(
            "Export session history CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="lumpycare_history.csv",
            mime="text/csv",
        )
        if st.button("Clear history", type="secondary"):
            st.session_state.history = []
            st.session_state.latest_result = None
            st.rerun()
    else:
        st.info("No images have been analysed in this session yet.")

st.divider()
st.markdown(
    '<p class="muted">⚕️ <b>Important:</b> LumpyCare is an educational research prototype, not a veterinary diagnostic device. '
    'Image quality, dataset size, class balance and distribution shift can affect predictions. '
    'Contact a qualified veterinarian for examination, diagnosis and treatment decisions.</p>',
    unsafe_allow_html=True,
)
