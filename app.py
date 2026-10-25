"""Local Streamlit dashboard for the trained intrusion detection model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "intrusion_model.joblib"

st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ ML-Based Network Intrusion Detection")
st.caption("Educational demo using a trained model and labelled public network-flow data.")

if not MODEL_PATH.exists():
    st.warning("No trained model found yet. Follow the README to download a dataset and run the training command first.")
    st.code("python src/train_model.py --data data/network_traffic.csv --target Label")
    st.stop()

saved = joblib.load(MODEL_PATH)
pipeline = saved["pipeline"]
feature_columns = saved["feature_columns"]
model_name = saved.get("model_name", "Trained model")

st.success(f"Loaded model: {model_name}")
st.markdown(
    "Upload a CSV containing the **same feature columns used during training**. "
    "A label column is optional; it is not used for predictions."
)
uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded, low_memory=False)
        data.columns = [str(c).strip() for c in data.columns]
        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            st.error("This CSV does not contain all required training features.")
            st.write("Missing columns:", missing)
            st.info("Use a compatible CSV from the same dataset/schema used to train the model.")
        else:
            X = data[feature_columns].copy()
            predictions = pipeline.predict(X)
            output = data.copy()
            output["Predicted_Label"] = predictions
            st.subheader("Prediction summary")
            counts = pd.Series(predictions).value_counts().rename_axis("Predicted class").reset_index(name="Rows")
            c1, c2 = st.columns(2)
            c1.metric("Rows analyzed", f"{len(output):,}")
            c2.metric("Predicted classes", str(pd.Series(predictions).nunique()))
            st.dataframe(counts, use_container_width=True)
            st.subheader("Sample predictions")
            st.dataframe(output.head(100), use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="intrusion_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")

with st.expander("Model and safety notes"):
    st.write("This demo analyzes uploaded CSV rows only. It does not capture live traffic, block connections, or guarantee that a prediction is a real attack.")
    st.write("Predictions depend on the training dataset and may produce false positives or false negatives.")
