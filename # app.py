# app.py (replace your current file with this)
import os
from io import BytesIO
import pickle
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader

# --------------------------
# Streamlit page config (must be first Streamlit call)
# --------------------------
st.set_page_config(page_title="Resume Classifier", layout="wide")

# --------------------------
# File paths
# --------------------------
MODEL_PATH = "model.pkl"
VEC_PATH = "vectorizer.pkl"

# --------------------------
# Helper to load from disk
# --------------------------
def load_pickle_from_path(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --------------------------
# Load model & vectorizer (from disk or via upload)
# --------------------------
model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    try:
        model = load_pickle_from_path(MODEL_PATH)
        vectorizer = load_pickle_from_path(VEC_PATH)
    except Exception as e:
        st.error(f"Error while loading model files from disk: {e}")
        st.stop()
else:
    st.warning("Model files not found in the app folder.")
    st.info(
        "Either run `train_and_save.py` locally to produce `model.pkl` and `vectorizer.pkl`, "
        "or upload the files here."
    )
    uploaded_model = st.file_uploader("Upload model.pkl (if not present)", type=["pkl"], key="m")
    uploaded_vec = st.file_uploader("Upload vectorizer.pkl (if not present)", type=["pkl"], key="v")

    if uploaded_model is not None and uploaded_vec is not None:
        try:
            # uploaded_model / uploaded_vec are file-like; pickle.load accepts file-like
            model = pickle.load(uploaded_model)
            uploaded_model.seek(0)
            vectorizer = pickle.load(uploaded_vec)
            uploaded_vec.seek(0)
            st.success("Loaded model and vectorizer from uploaded files.")
        except Exception as e:
            st.error(f"Failed to load uploaded .pkl files: {e}")
            st.stop()
    else:
        st.stop()  # stop until user provides files or adds files to repo

# --------------------------
# UI: Title + input
# --------------------------
st.title("📄 Entry-Level Resume Classifier")
resume_input = st.text_area("Paste resume text OR upload a file below:")

uploaded_file = st.file_uploader("Upload resume file", type=["docx", "pdf", "txt"])

# --- Handle uploaded resume file robustly ---
if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".docx"):
            doc = Document(BytesIO(raw_bytes))
            resume_input = "\n".join(p.text for p in doc.paragraphs)
        elif uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(raw_bytes))
            resume_input = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif uploaded_file.name.lower().endswith(".txt"):
            resume_input = raw_bytes.decode("utf-8", errors="ignore")
        else:
            st.warning("Unsupported file type.")
    except Exception as e:
        st.error(f"Failed to read uploaded resume file: {e}")

# --- Prediction ---
if st.button("Predict"):
    if resume_input is None or resume_input.strip() == "":
        st.warning("⚠️ Please enter or upload resume text.")
    else:
        try:
            vec = vectorizer.transform([resume_input])
            pred = model.predict(vec)[0]
            st.success(f"✅ Predicted Category: **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
