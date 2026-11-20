from __future__ import annotations

import io
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ---------------------------------------------------------------------
# Lumpy Skin Disease Detection | Streamlit edition
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LumpyCare | Skin Screening",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = ["Lumpy skin", "Normal skin"]
IMAGE_SIZE = (128, 128)
FEATURE_DIR = "features"
FEATURES_FILE = os.path.join(FEATURE_DIR, "features.txt.npy")
LABELS_FILE = os.path.join(FEATURE_DIR, "labels.txt.npy")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f8f6; }
    [data-testid="stHeader"] { background: rgba(245,248,246,.92); }
    .hero {
      padding: 2rem 2.2rem; border-radius: 22px;
      background: linear-gradient(120deg,#123c32 0%,#176b52 60%,#2b8a65 100%);
      color: white; margin-bottom: 1.2rem;
      box-shadow: 0 12px 32px rgba(18,60,50,.14);
    }
    .hero h1 { color:white; font-family:Manrope,sans-serif; font-size:2.25rem; margin:0 0 .35rem 0; }
    .hero p { color:#d8eee4; font-size:1rem; margin:0; }
    .eyebrow { color:#a9ddc4; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; }
    .metric-card {
      background:white; padding:1rem 1.1rem; border:1px solid #e2ebe5;
      border-radius:16px; min-height:105px; box-shadow:0 4px 16px rgba(20,50,35,.035);
    }
    .metric-label { color:#687b70; font-size:.82rem; font-weight:600; }
    .metric-value { color:#153d31; font-family:Manrope,sans-serif; font-size:1.65rem; font-weight:800; margin-top:.25rem; }
    .soft-card { background:white; padding:1.25rem; border:1px solid #e2ebe5; border-radius:18px; }
    .small-muted { color:#6d7f75; font-size:.88rem; }
    .stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:700; }
    div[data-testid="stFileUploader"] { background:white; border:1px dashed #9bb9a8; border-radius:14px; padding:.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def extract_features(rgb_image: np.ndarray) -> np.ndarray:
    """Use the same 128x128 RGB + 6-cluster K-means representation as training."""
    resized = cv2.resize(rgb_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    pixels = np.float32(resized.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    cv2.setRNGSeed(42)
    _compactness, labels, centers = cv2.kmeans(
        pixels, 6, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    segmented = np.uint8(centers)[labels.flatten()]
    return segmented.reshape(-1).astype(np.float32)

@st.cache_resource(show_spinner="Preparing the screening model…")
def train_model():
    """Load the repository's feature arrays; rebuild them from folders if unavailable."""
    if os.path.exists(FEATURES_FILE) and os.path.exists(LABELS_FILE):
        X = np.load(FEATURES_FILE, allow_pickle=False)
        y = np.load(LABELS_FILE, allow_pickle=False)
    else:
        xs, ys = [], []
        for label, class_name in enumerate(CLASSES):
            folder = os.path.join("SkinDataset", class_name)
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                path = os.path.join(folder, filename)
                image = cv2.imread(path)
                if image is None:
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                xs.append(extract_features(rgb))
                ys.append(label)
        if not xs:
            raise FileNotFoundError(
                "Training data was not found. Add features/*.npy or the SkinDataset/Lumpy skin and SkinDataset/Normal skin folders."
            )
        X, y = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).ravel()
    if X.ndim != 2 or X.shape[1] != IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3:
        raise ValueError(f"Feature file has unexpected shape {X.shape}. Expected (*, 49152).")
    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both Lumpy skin and Normal skin classes.")
    if len(X) < 10:
        raise ValueError("At least 10 labelled images are recommended to train and evaluate the model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = SVC(C=12, gamma="scale", kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=CLASSES,
            output_dict=True, zero_division=0
        ),
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "total_count": int(len(X)),
    }
    return model, metrics

def hero():
    st.markdown(
        '<div class="hero"><div class="eyebrow">AI-assisted cattle skin screening</div>'
        '<h1>🐄 LumpyCare</h1><p>Image-based screening support for early awareness of Lumpy Skin Disease.</p></div>',
        unsafe_allow_html=True,
    )

def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## 🐄 LumpyCare")
    st.caption("Lumpy Skin Disease Detection")
    page = st.radio("NAVIGATION", ["Overview", "Image screening", "Model performance", "Prediction history"])
    st.divider()
    st.caption("Built with Streamlit · OpenCV · Scikit-learn")
    st.caption("Screening support only — consult a veterinarian.")

hero()

try:
    model, metrics = train_model()
    model_ready = True
except Exception as exc:
    model_ready = False
    model_error = str(exc)

if not model_ready:
    st.error("The model could not be prepared.")
    st.code(model_error)
    st.info("Check that the feature arrays or the two SkinDataset class folders are present in the repository.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

if page == "Overview":
    st.subheader("Your screening workspace")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Model test accuracy", f"{metrics['accuracy']*100:.1f}%")
    with c2: metric_card("Images in dataset", f"{metrics['total_count']:,}")
    with c3: metric_card("Training images", f"{metrics['train_count']:,}")
    with c4: metric_card("Test images", f"{metrics['test_count']:,}")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("### ✨ How to use LumpyCare")
        st.markdown(
            "1. Open **Image screening** from the sidebar.\n"
            "2. Upload a clear photo of the affected cattle skin area.\n"
            "3. Review the predicted class and model confidence.\n"
            "4. Download the result or review earlier screenings in **Prediction history**."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("### 🛡️ Responsible use")
        st.markdown(
            "This tool classifies images using the project's SVM model. Lighting, camera quality, "
            "image angle and dataset limitations can affect results. A veterinarian must confirm "
            "any suspected disease."
        )
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Image screening":
    st.subheader("📷 Upload a cattle skin image")
    st.markdown('<p class="small-muted">For best results, use a sharp, well-lit, close-up image with the skin area visible.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            rgb = np.asarray(pil_image)
            a, b = st.columns([1, 1])
            with a:
                st.image(pil_image, caption="Uploaded image", use_container_width=True)
            with b:
                st.markdown("### Screening result")
                if st.button("Analyse image", type="primary", use_container_width=True):
                    with st.spinner("Analysing image…"):
                        vector = extract_features(rgb).reshape(1, -1)
                        prediction = int(model.predict(vector)[0])
                        probabilities = model.predict_proba(vector)[0]
                        confidence = float(probabilities[prediction])
                        label = CLASSES[prediction]
                        record = {
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Prediction": label,
                            "Confidence": round(confidence * 100, 2),
                            "File": uploaded.name,
                        }
                        st.session_state.history.insert(0, record)
                        st.session_state.latest_result = record
                if "latest_result" in st.session_state:
                    result = st.session_state.latest_result
                    if result["Prediction"] == "Lumpy skin":
                        st.error("⚠️ Model result: Lumpy skin")
                    else:
                        st.success("✅ Model result: Normal skin")
                    st.metric("Model confidence", f"{result['Confidence']:.1f}%")
                    st.progress(min(max(result["Confidence"] / 100, 0.0), 1.0))
                    st.caption("Confidence is the model's probability estimate, not a clinical certainty.")
                    st.download_button(
                        "⬇️ Download this result",
                        data=pd.DataFrame([result]).to_csv(index=False).encode("utf-8"),
                        file_name="lumpycare_prediction.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as exc:
            st.error(f"Could not read or analyse this image: {exc}")
    else:
        st.info("Upload an image to begin. Your image is processed for this session and is not saved to the repository.")

elif page == "Model performance":
    st.subheader("📊 Model performance")
    st.caption("Metrics are calculated on a held-out test split. The model is trained only on the training split.")
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Test accuracy", f"{metrics['accuracy']*100:.2f}%")
    with c2: metric_card("Training samples", str(metrics["train_count"]))
    with c3: metric_card("Test samples", str(metrics["test_count"]))
    st.write("")
    cm = metrics["matrix"]
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], labels=CLASSES)
    ax.set_yticks([0, 1], labels=CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=13)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    report = pd.DataFrame(metrics["report"]).T
    st.markdown("#### Classification report")
    st.dataframe(report.round(3), use_container_width=True)

elif page == "Prediction history":
    st.subheader("🗂️ Session prediction history")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="lumpycare_prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.pop("latest_result", None)
            st.rerun()
    else:
        st.info("No predictions yet. Go to Image screening and analyse an image to build your history.")

st.divider()
st.markdown(
    '<p class="small-muted">⚕️ <b>Important:</b> This application is an educational image-screening aid, '
    'not a veterinary diagnostic device. Do not make treatment decisions from this prediction alone. '
    'Contact a qualified veterinarian for examination and diagnosis.</p>',
    unsafe_allow_html=True,
)
