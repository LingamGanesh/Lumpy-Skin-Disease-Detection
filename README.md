# 🐄 LumpyCare — Lumpy Skin Disease Detection

A Streamlit web application for AI-assisted image screening of cattle skin images using OpenCV K-means feature extraction and an SVM classifier.

## Features

- Modern Streamlit interface: Overview, Image Screening, Model Performance and Prediction History.
- Upload JPG, JPEG, PNG or WEBP images and preview before analysis.
- SVM prediction with probability estimate and downloadable CSV result.
- Test accuracy, confusion matrix and classification report.
- Session-only prediction history with CSV export.
- Streamlit Cloud compatible dependencies (headless OpenCV; no Tkinter desktop GUI).

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Select **Create app**.
3. Choose repository `LingamGanesh/Lumpy-Skin-Disease-Detection`, branch `main`, and entrypoint `app.py`.
4. Select **Deploy**.

The app expects either:
- `features/features.txt.npy` and `features/labels.txt.npy` (preferred), or
- `SkinDataset/Lumpy skin/` and `SkinDataset/Normal skin/` folders so features can be generated.

## Important notes

- This is an educational prototype. Predictions are not veterinary diagnoses and must not be used alone to make treatment decisions.
- Reported test accuracy is based on a held-out split of the available feature dataset. It is not proof of real-world clinical accuracy.
- The original Tkinter desktop interface has been replaced by Streamlit. The Jupyter notebook may still contain old desktop code; the deployed app uses `app.py`.
