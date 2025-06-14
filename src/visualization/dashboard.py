import streamlit as st
from streamlit_autorefresh import st_autorefresh  # type: ignore
import os
import json
import subprocess
from pathlib import Path
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

def start_process(key, cmd, success_msg, error_msg):
    """Launch `cmd` once and store the Popen handle under st.session_state[key]."""
    if key in st.session_state:
        st.warning(f"{success_msg} is already running.")
        return

    try:
        # Make stdout/stderr go to a log file, or DEVNULL if you prefer silent
        log_path = os.path.join("logs", f"{key}.log")
        os.makedirs("logs", exist_ok=True)
        log_file = open(log_path, "a")

        # Launch the subprocess
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.getcwd(),  # ensure we’re in project root
        )
        st.session_state[key] = proc
        st.success(success_msg)
    except Exception as e:
        st.error(f"{error_msg}: {e}")

st.sidebar.header("🚀 Simulated Real-Time Inference")

# 1) explanatory text
st.sidebar.markdown(
    """
    **How it works**  
    1. **Injector**: copies one held-out test image into `data/raw/` every _N_ seconds.  
    2. **Watcher**: immediately moves that image to `data/processed/`.  
    3. **Inference**: runs your model on it, appends to `streaming_predictions.csv`.  
    4. **Dashboard**: auto-refreshes every 5 s to show latest results.  
    """
)

# 2) interactive delay picker
delay = st.sidebar.number_input(
    "Injector delay (seconds)",
    min_value=1,
    max_value=60,
    value=10,
    step=1
)

# Button to start the watcher
if st.sidebar.button("▶️ Start Watcher"):
    start_process(
        key       = "watcher_proc",
        cmd       = ["python", "-m", "src.ingest.watcher"],
        success_msg = "Watcher",
        error_msg   = "Failed to start watcher",
    )

# Button to start the injector
if st.sidebar.button("▶️ Start Injector"):
    start_process(
        key       = "injector_proc",
        cmd       = [
            "python", "-m", "src.utils.injector",
            "--src_dir", "data/test",
            "--dst_dir", "data/raw",
            "--delay",   str(delay),
            "--seed",    "42"
        ],
        success_msg = "Injector",
        error_msg   = "Failed to start injector",
    )

# Button to stop both
if st.sidebar.button("⏹ Stop All"):
    for key in ["watcher_proc","injector_proc"]:
        proc = st.session_state.get(key)
        if proc:
            proc.terminate()
            st.session_state.pop(key)
    st.warning("Stopped watcher and injector.")

@st.cache_data
def load_batch():
    try:
        df = pd.read_csv(f"{OUTPUTS}/predictions.csv")
        return df
    except FileNotFoundError:
        st.error("Batch predictions CSV not found. Please run inference first.")
        return pd.DataFrame()

@st.cache_data(ttl=10)
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
    count = st_autorefresh(interval=int(delay * 1000), key="data_refresh")

    st.title("Real-time Streaming Inference")
    df = load_stream()

    st.markdown(
        f"**Total Images Processed:** {len(df):,} \n"
    )

    # Display latest image
    if not df.empty:
        last_row = df.iloc[-1]
        last_path = last_row.filepath
        last_pred = last_row.pred_label
        last_conf = last_row.confidence
    else:
        last_path = None
        last_pred = None
        last_conf = None

    if last_path:
        st.subheader("Last Inference")
        col1, col2 = st.columns([1, 1])
        with col1:
            # force width or height as you like
            st.image(last_path, caption=os.path.basename(last_path), width=300)
        with col2:
            st.markdown("### Prediction")
            st.markdown(f"**{last_pred}**  \nconfidence: {last_conf:.4f}")
    else:
        st.info("Waiting for first streaming inference…")

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

    
