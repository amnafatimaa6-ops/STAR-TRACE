
import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

# GitHub ZIP uploaded from your STARTRACE project

GITHUB_ZIP_URL = (
    "https://raw.githubusercontent.com/"
    "amnafatimaa6-ops/STAR-TRACE/main/"
    "STARTRACE_deployment_assets.zip"
)

st.set_page_config(
    page_title="STARTRACE | AI Astronomy Lab",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 STARTRACE")
st.caption("AI-assisted stellar light-curve analysis")

@st.cache_resource
def load_assets(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "STARTRACE-Streamlit"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    names = archive.namelist()

    def find_file(filename):
        matches = [n for n in names if n.endswith("/" + filename)]
        if not matches:
            raise FileNotFoundError(f"{filename} missing from ZIP")
        return archive.read(matches[0])

    models = {}
    for filename in (
        "startrace_denoising_v2.keras",
        "startrace_autoencoder.keras",
    ):
        models[filename] = tf.keras.models.load_model(
            io.BytesIO(find_file(filename)),
            compile=False,
        )

    tables = {}
    for filename in (
        "startrace_v2_final_evaluation_summary.csv",
        "startrace_v2_model_comparison.csv",
        "startrace_morphology_features.csv",
        "startrace_v2_morphology_clusters.csv",
    ):
        try:
            tables[filename] = pd.read_csv(
                io.BytesIO(find_file(filename))
            )
        except FileNotFoundError:
            pass

    return models, tables, names


st.sidebar.header("STARTRACE controls")
if st.sidebar.button("Load GitHub models and results"):
    st.cache_resource.clear()

try:
    with st.spinner("Loading STARTRACE assets from GitHub..."):
        models, tables, asset_names = load_assets(GITHUB_ZIP_URL)
    st.sidebar.success(f"{len(models)} models loaded")
except Exception as exc:
    st.error(
        "Could not load the GitHub ZIP. Check the repository URL, "
        "ZIP filename, repository visibility, and model files."
    )
    st.code(str(exc))
    st.stop()

model_choice = st.sidebar.selectbox(
    "Select model",
    ["Denoising v2", "Original autoencoder"],
)
model_file = (
    "startrace_denoising_v2.keras"
    if model_choice == "Denoising v2"
    else "startrace_autoencoder.keras"
)
model = models[model_file]

st.subheader("Existing research results")
for filename, table in tables.items():
    with st.expander(filename.replace("_", " ").replace(".csv", "")):
        st.dataframe(table, use_container_width=True)

st.divider()
st.subheader("Analyse a light curve")
uploaded = st.file_uploader(
    "Upload a CSV containing time and flux measurements",
    type=["csv"],
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_columns:
        st.error("The CSV must contain numeric columns.")
        st.stop()

    time_col = st.selectbox("Time column", ["None"] + numeric_columns)
    flux_candidates = [
        col for col in numeric_columns if col != time_col
    ]
    if not flux_candidates:
        st.error("Choose a numeric flux column.")
        st.stop()

    flux_col = st.selectbox("Flux column", flux_candidates)
    flux = pd.to_numeric(df[flux_col], errors="coerce").to_numpy()
    valid = np.isfinite(flux)
    flux = flux[valid]

    if time_col != "None":
        time = pd.to_numeric(
            df.loc[valid, time_col], errors="coerce"
        ).to_numpy()
    else:
        time = np.arange(len(flux))

    if len(flux) < 256:
        st.error("At least 256 valid flux measurements are required.")
        st.stop()

    # Provisional normalization; replace with the exact training
    # preprocessing before treating scores as scientifically comparable.
    median = np.median(flux)
    scale = np.percentile(flux, 95) - np.percentile(flux, 5)
    if not np.isfinite(scale) or scale <= 1e-12:
        st.error("Flux variation is too small to normalize.")
        st.stop()

    signal = (flux - median) / scale
    stride = 128
    windows = np.stack([
        signal[i:i + 256]
        for i in range(0, len(signal) - 255, stride)
    ]).astype("float32")

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) != 3 or input_shape[1:] != (256, 1):
        st.error(f"Unexpected model input shape: {input_shape}")
        st.stop()

    predictions = model.predict(windows[..., None], verbose=0)
    predictions = np.asarray(predictions).reshape(-1, 256)
    mse = np.mean((windows - predictions) ** 2, axis=1)

    st.subheader("Reconstruction results")
    a, b = st.columns(2)
    a.metric("Observations", f"{len(flux):,}")
    b.metric("Windows", f"{len(windows):,}")

    selected = st.slider("Inspect window", 0, len(windows) - 1, 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(windows[selected], label="Input")
    ax.plot(predictions[selected], label="Reconstruction")
    ax.set_xlabel("Window sample")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mse)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Reconstruction MSE")
    st.pyplot(fig)
    plt.close(fig)

    results = pd.DataFrame({
        "window_index": np.arange(len(mse)),
        "reconstruction_mse": mse,
    })
    st.download_button(
        "Download analysis CSV",
        results.to_csv(index=False).encode("utf-8"),
        "startrace_results.csv",
        "text/csv",
    )

    st.warning(
        "Exploratory output only. The upload preprocessing is provisional "
        "and has not yet been validated against the training pipeline. "
        "Reconstruction error is not a confirmed event detection."
    )

st.caption("STARTRACE research prototype | Not a validated discovery system")
