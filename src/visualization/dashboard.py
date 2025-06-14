import streamlit as st
from streamlit_autorefresh import st_autorefresh  # type: ignore
import os
import json
from pathlib import Path
import glob
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Set paths
HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="Geo-ML Dashboard",
    page_icon="🌍",
    layout="wide",
)

# Sidebar navigation
page = st.sidebar.selectbox("Page", ["Streaming", "Batch Metrics"])

@st.cache_data
def load_batch():
    try:
        df = pd.read_csv(f"{OUTPUTS}/predictions.csv")
        return df
    except FileNotFoundError:
        st.error("Batch predictions CSV not found. Please run inference first.")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_stream():
    try:
        df = pd.read_csv(f"{OUTPUTS}/streaming_predictions.csv")
        return df
    except FileNotFoundError:
        st.error("Streaming predictions CSV not found. Please run streaming inference first.")
        return pd.DataFrame()

@st.cache_data
def load_class_names():
    try:
        with open(f"{OUTPUTS}/class_names.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Class names JSON not found. Please ensure it exists.")
        return pd.DataFrame()
    
if page == "Batch Metrics":
    st.title("Batch Test‐Set Evaluation")
    df = load_batch()
    class_names = load_class_names()

    # Overall Accuracy
    if not df.empty:
        acc = (df['true_label'] == df['pred_label']).mean()
        st.metric("Overall Accuracy", f"{acc:.2%}")
    else:
        st.warning("No batch data available.")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    if not df.empty:
        conf_matrix = pd.crosstab(df.true_label, df.pred_label, normalize='index')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap='Blues', ax=ax,
                    xticklabels=class_names['class_name'], yticklabels=class_names['class_name'])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

    else:
        st.warning("No batch data available.")
    
    # Per-class confidence bar chart
    st.subheader("Average Confidence by True Label")
    if not df.empty:
        conf = df.groupby("true_label").confidence.mean().reindex(class_names)
        st.bar_chart(conf)
    else:
        st.warning("No batch data available.")
    
elif page == "Streaming":
    # Auto-refresh every 60 seconds
    count = st_autorefresh(interval=60 * 1000, key="data_refresh")

    st.title("Real-time Streaming Inference")
    df = load_stream()

    st.markdown(
        f"**Total Images Processed:** {len(df):,} \n"
    )

    # Display latest image
    all_images = glob.glob("data/processed/*.png") + glob.glob("data/processed/*.jpg")
    if all_images:
        latest = max(all_images, key=os.path.getmtime)
        # st.subheader("Last Image Processed")
        # st.image(latest, caption=os.path.basename(latest), use_container_width=False)
    else:
        st.info("No processed images yet.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Last Image Processed")
        st.image(latest, caption=os.path.basename(latest), use_container_width=True)
    with col2:
        st.subheader("Last Prediction")
        st.markdown(
        f"{df.iloc[-1]['pred_label']} \n"
        f" (confidence = {df.iloc[-1]['confidence']:.2f})"
    )

    # Live Predictions Table with running accuracy
    st.subheader("Recent Predictions")
    st.dataframe(df.tail(10))
    running_acc = (df['true_label'] == df['pred_label']).mean()
    st.markdown(f"<h3 style='font-size:14px; color:#00FF00;'>Overall Accuracy: {running_acc:.2%}</h1>",
            unsafe_allow_html=True)

    # Streaming Class Counts over Time
    st.subheader("Cumulative Predictions by Class")
    counts = (
        df.pred_label
        .value_counts()
        .reindex(load_class_names(), fill_value=0)
    )
    st.bar_chart(counts)

    # Confidence trend
    st.subheader("Confidence Over Time")
    df["timestamp"] = pd.to_datetime(df["filepath"].apply(
        lambda p: os.path.getmtime(p)), unit='s')
    df = df.sort_values("timestamp")
    st.line_chart(df.set_index("timestamp")["confidence"])

    
