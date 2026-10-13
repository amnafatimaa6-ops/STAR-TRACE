
from __future__ import annotations

import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# STAR-TRACE | MISSION CONTROL
# ============================================================

st.set_page_config(
    page_title="STAR-TRACE | Mission Control",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSET_ZIP_URL = (
    "https://raw.githubusercontent.com/"
    "amnafatimaa6-ops/STAR-TRACE/main/"
    "STARTRACE_deployment_assets.zip"
)

MODEL_FILES = {
    "Denoising model": "startrace_denoising_v2.keras",
    "Autoencoder": "startrace_autoencoder.keras",
}

CSV_FILES = [
    "startrace_v2_final_evaluation_summary.csv",
    "startrace_v2_model_comparison.csv",
    "startrace_morphology_features.csv",
    "startrace_v2_morphology_clusters.csv",
]

WINDOW_SIZE = 256


# ============================================================
# VISUAL SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --bg: #050914;
        --panel: #0a1424;
        --panel2: #0d1b2d;
        --line: #19344a;
        --cyan: #53e6ff;
        --blue: #5598ff;
        --text: #e4f3ff;
        --muted: #88a3b9;
        --green: #6fffc2;
        --amber: #ffc66d;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(5,9,20,.94);
    }

    [data-testid="stSidebar"] {
        background: #07111f;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    h1, h2, h3 {
        letter-spacing: -.035em;
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--cyan);
        font-size: .72rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .5rem;
    }

    .hero {
        border: 1px solid #20445d;
        border-radius: 18px;
        padding: 27px 30px;
        background:
            radial-gradient(ellipse at 88% 10%, #10384b 0%, transparent 40%),
            linear-gradient(125deg, #0b192a, #07101e 70%);
        margin-bottom: 20px;
        overflow: hidden;
    }

    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.2rem);
        margin: 0;
        font-weight: 800;
        line-height: 1.05;
    }

    .hero p {
        color: var(--muted);
        max-width: 780px;
        margin: 12px 0 0;
        line-height: 1.65;
    }

    .hero-tag {
        display: inline-block;
        border: 1px solid #23617a;
        background: #0b2939;
        color: var(--cyan);
        padding: 5px 9px;
        border-radius: 6px;
        font: 500 .68rem 'IBM Plex Mono', monospace;
        letter-spacing: .06em;
        margin-bottom: 16px;
    }

    .section-label {
        color: var(--cyan);
        font: 500 .72rem 'IBM Plex Mono', monospace;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 18px 0 8px;
    }

    .metric-card {
        background: linear-gradient(145deg, #0d1c2e, #091321);
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 17px;
        min-height: 105px;
    }

    .metric-label {
        color: var(--muted);
        font: .69rem 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .09em;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: var(--muted);
        font-size: .76rem;
        margin-top: 4px;
    }

    .panel {
        background: #091524;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 17px 19px;
        margin: 10px 0;
    }

    .panel-title {
        color: var(--text);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .panel-copy {
        color: var(--muted);
        font-size: .87rem;
        line-height: 1.6;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--green);
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #6fffc255;
    }

    .stButton > button {
        background: #0d2a3b;
        border: 1px solid #28617b;
        color: var(--cyan);
        border-radius: 8px;
        font-weight: 650;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #12394c;
        border-color: var(--cyan);
        color: white;
    }

    .stDownloadButton > button {
        background: #0a1928;
        border: 1px solid var(--line);
        color: var(--text);
        border-radius: 8px;
    }

    div[data-testid="stMetric"] {
        background: #0a1726;
        border: 1px solid var(--line);
        padding: 12px 15px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--cyan);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
    }

    hr {
        border-color: var(--line);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD THE ORIGINAL DEPLOYMENT ZIP
# ============================================================

def download_archive(url: str) -> bytes:
    """Download the original asset archive and validate it."""
    last_error = None

    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "STAR-TRACE-Streamlit/1.0"},
            )

            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()

            if not zipfile.is_zipfile(io.BytesIO(content)):
                raise ValueError(
                    "Downloaded content is not a valid ZIP archive."
                )

            return content

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not retrieve deployment ZIP: {last_error}"
    )


def find_archive_member(names, target_name):
    """Locate an archive file even if it is inside a subfolder."""
    for name in names:
        if Path(name).name.lower() == target_name.lower():
            return name
    return None


@st.cache_resource(show_spinner="Loading STAR-TRACE research assets...")
def load_assets(url: str):
    archive_bytes = download_archive(url)

    models = {}
    tables = {}
    model_info = {}
    asset_errors = []

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()

        for label, filename in MODEL_FILES.items():
            member = find_archive_member(names, filename)

            if member is None:
                asset_errors.append(f"Missing model: {filename}")
                continue

            try:
                model_bytes = archive.read(member)

                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, filename)

                    with open(model_path, "wb") as file:
                        file.write(model_bytes)

                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                    )

                models[label] = model
                model_info[label] = {
                    "filename": filename,
                    "parameters": model.count_params(),
                    "input_shape": str(model.input_shape),
                    "output_shape": str(model.output_shape),
                    "archive_member": member,
                }

            except Exception as exc:
                asset_errors.append(
                    f"{filename}: {type(exc).__name__}: {exc}"
                )

        for filename in CSV_FILES:
            member = find_archive_member(names, filename)

            if member is None:
                asset_errors.append(f"Missing CSV: {filename}")
                continue

            try:
                tables[filename] = pd.read_csv(
                    io.BytesIO(archive.read(member))
                )
            except Exception as exc:
                asset_errors.append(
                    f"{filename}: {type(exc).__name__}: {exc}"
                )

    return {
        "models": models,
        "tables": tables,
        "model_info": model_info,
        "archive_entries": len(names),
        "asset_errors": asset_errors,
    }


# ============================================================
# SYNTHETIC DEMONSTRATION SIGNAL
# ============================================================

def make_demo_curve(n: int = 1800, seed: int = 17) -> pd.DataFrame:
    """
    Generate a reproducible illustrative signal.
    This is not real telescope data or a verified discovery.
    """
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0, 18, n)

    variability = (
        0.0030 * np.sin(2 * np.pi * time_axis / 5.8)
        + 0.0015 * np.sin(2 * np.pi * time_axis / 2.4)
    )

    noise = rng.normal(0, 0.0015, n)
    flux = 1.0 + variability + noise

    for center in (4.5, 9.0, 13.5):
        distance = np.abs(time_axis - center)

        flux[distance < 0.06] -= 0.012

        shoulder = (distance >= 0.06) & (distance < 0.12)
        flux[shoulder] -= 0.003

    return pd.DataFrame({
        "time": time_axis,
        "flux": flux,
    })


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_flux(values, normalization):
    """
    Normalize input and return a function that reverses the normalization.

    IMPORTANT: choose the normalization used in model training if known.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if normalization == "Z-score":
        center = float(np.mean(values))
        scale = float(np.std(values))

        if scale < 1e-8:
            scale = 1.0

        prepared = (values - center) / scale
        restore = lambda result: result * scale + center

    elif normalization == "Min-max":
        low = float(np.min(values))
        high = float(np.max(values))
        scale = high - low

        if scale < 1e-8:
            scale = 1.0

        prepared = (values - low) / scale
        restore = lambda result: result * scale + low

    else:
        prepared = values.copy()
        restore = lambda result: result

    return prepared.astype(np.float32), restore


def predict_window(model, window):
    """Run a model on one fixed-length window."""
    x = np.asarray(window, dtype=np.float32).reshape(
        1, WINDOW_SIZE, 1
    )

    prediction = model.predict(x, verbose=0)
    prediction = np.asarray(prediction)

    if prediction.ndim == 3:
        prediction = prediction[0, :, 0]
    elif prediction.ndim == 2:
        prediction = prediction[0]
    else:
        prediction = prediction.reshape(-1)

    if len(prediction) != WINDOW_SIZE:
        raise ValueError(
            f"Expected {WINDOW_SIZE} output samples, got "
            f"{len(prediction)}. The model may not be compatible "
            "with this inference workflow."
        )

    if not np.all(np.isfinite(prediction)):
        raise ValueError("Model output contains NaN or infinite values.")

    return prediction.astype(np.float32)


def denoise_light_curve(model, values, stride=128):
    """
    Predict overlapping 256-point windows and average their overlaps.
    This creates a reconstruction, not a classification.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if len(values) < 8:
        raise ValueError("At least 8 valid samples are required.")

    if not 1 <= stride <= WINDOW_SIZE:
        raise ValueError("Stride must be between 1 and 256.")

    n = len(values)

    if n < WINDOW_SIZE:
        total_padding = WINDOW_SIZE - n
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        padded = np.pad(
            values,
            (left_padding, right_padding),
            mode="edge",
        )

        prediction = predict_window(model, padded)

        return prediction[
            left_padding:left_padding + n
        ]

    starts = list(range(0, n - WINDOW_SIZE + 1, stride))
    final_start = n - WINDOW_SIZE

    if starts[-1] != final_start:
        starts.append(final_start)

    accumulated = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    for start in starts:
        end = start + WINDOW_SIZE
        prediction = predict_window(model, values[start:end])

        accumulated[start:end] += prediction
        counts[start:end] += 1

    return (
        accumulated / np.maximum(counts, 1)
    ).astype(np.float32)


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def heading(kicker, title, description=None):
    st.markdown(
        f'<div class="eyebrow">{kicker}</div><h2>{title}</h2>',
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<p style="color:#88a3b9;line-height:1.6">'
            f'{description}</p>',
            unsafe_allow_html=True,
        )


def plot_light_curve(
    x,
    y,
    label,
    color="#53e6ff",
    title="Light curve",
    x_label="Time",
    y_label="Flux",
):
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#091524")
    ax.set_facecolor("#091524")

    ax.plot(
        x, y,
        color=color,
        linewidth=0.85,
        label=label,
    )

    ax.set_title(title, color="#e4f3ff", loc="left", pad=12)
    ax.set_xlabel(x_label, color="#88a3b9")
    ax.set_ylabel(y_label, color="#88a3b9")
    ax.tick_params(colors="#88a3b9")
    ax.grid(True, color="#1b3448", linewidth=0.55, alpha=0.8)

    for spine in ax.spines.values():
        spine.set_color("#19344a")

    ax.legend(
        facecolor="#0d1b2d",
        edgecolor="#19344a",
        labelcolor="#e4f3ff",
    )

    fig.tight_layout()
    return fig


def signal_signature(time_values, flux_values):
    """
    Identify the current input so a previous result is not shown
    for a different CSV or configuration.
    """
    return (
        len(time_values),
        float(time_values[0]),
        float(time_values[-1]),
        float(np.mean(flux_values)),
        float(np.std(flux_values)),
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 0 20px">
            <div style="font-size:1.8rem">🔭</div>
            <div style="font-size:1.25rem;font-weight:800;
                        letter-spacing:.05em">
                STAR-TRACE
            </div>
            <div class="eyebrow" style="margin-top:5px">
                STELLAR SIGNAL INTELLIGENCE
            </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Mission Overview",
            "Signal Laboratory",
            "Model Intelligence",
            "Research Archive",
            "System Diagnostics",
        ],
        key="startrace_page",
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="section-label">Operating mode</div>
        <div class="panel-copy">
            <span class="status-dot"></span>
            Local analysis interface
        </div>
        <br>
        <div class="panel-copy">
            No live telescope feed is connected. Model inference runs
            only when requested in the analysis workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("↻ Reload research assets", use_container_width=True):
        load_assets.clear()
        st.rerun()


# ============================================================
# LOAD ASSETS
# ============================================================

assets = None
asset_error = None

try:
    assets = load_assets(ASSET_ZIP_URL)
except Exception as exc:
    asset_error = f"{type(exc).__name__}: {exc}"


# ============================================================
# TOP BANNER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">
            MISSION CONTROL / ASTROPHYSICAL AI
        </div>
        <h1>
            STAR-TRACE<span style="color:#53e6ff">.</span>
        </h1>
        <p>
            An AI-assisted environment for exploring stellar light curves,
            testing reconstruction models, and inspecting research datasets.
            Start with a synthetic demonstration or bring your own data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if asset_error:
    st.error(
        "Research assets could not be loaded. Check the GitHub repository, "
        "deployment ZIP URL, and archive availability."
    )

    with st.expander("Technical details"):
        st.code(asset_error)

    st.info(
        "The interface remains available, but model inference and "
        "research archive browsing require the original assets."
    )


# ============================================================
# PAGE 1: MISSION OVERVIEW
# ============================================================

if page == "Mission Overview":
    heading(
        "01 / Mission status",
        "Welcome to the observatory",
        "Explore the workspace and its bundled research assets. "
        "The default signal is synthetic, not an astronomical observation.",
    )

    model_count = len(assets["models"]) if assets else 0
    table_count = len(assets["tables"]) if assets else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Keras models", str(model_count), "Loaded from ZIP")

    with c2:
        metric_card("Research tables", str(table_count), "CSV assets loaded")

    with c3:
        metric_card("Demo source", "SYNTHETIC", "Generated locally")

    with c4:
        metric_card("Live telemetry", "OFFLINE", "No telescope connection")

    st.markdown(
        '<div class="section-label">'
        'Signal preview / synthetic demonstration'
        '</div>',
        unsafe_allow_html=True,
    )

    demo = make_demo_curve()

    fig = plot_light_curve(
        demo["time"],
        demo["flux"],
        "Synthetic demonstration",
        title="Illustrative stellar-like variability",
        x_label="Synthetic time (arbitrary units)",
        y_label="Synthetic flux (arbitrary units)",
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.warning(
        "This curve is generated for demonstration purposes. "
        "Its dips are illustrative and do not represent confirmed "
        "exoplanets or any other astronomical discovery."
    )

    st.markdown(
        '<div class="section-label">Mission modules</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    01 — Signal Laboratory
                </div>
                <div class="panel-copy">
                    Explore the synthetic signal or upload a CSV.
                    Select time and flux columns, run the denoising
                    model, inspect residuals, and export results.
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">
                    02 — Model Intelligence
                </div>
                <div class="panel-copy">
                    Inspect Keras models, tensor shapes, and
                    parameter counts. Model metadata alone does
                    not establish scientific validity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    03 — Research Archive
                </div>
                <div class="panel-copy">
                    Browse the four included CSV files, inspect
                    columns, and download research tables.
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">
                    04 — System Diagnostics
                </div>
                <div class="panel-copy">
                    Inspect software versions, loaded assets,
                    model compatibility, and TensorFlow devices.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if st.button("Open Signal Laboratory →", use_container_width=True):
        st.session_state["startrace_page"] = "Signal Laboratory"
        st.rerun()


# ============================================================
# PAGE 2: SIGNAL LABORATORY
# ============================================================

elif page == "Signal Laboratory":
    heading(
        "02 / Signal processing",
        "Signal Laboratory",
        "Choose a demonstration or upload a CSV containing numeric time "
        "and flux columns. The model produces a reconstruction, not "
        "an event classification.",
    )

    source = st.radio(
        "SIGNAL SOURCE",
        ["Synthetic demonstration", "Upload CSV"],
        horizontal=True,
        key="signal_source",
    )

    if source == "Synthetic demonstration":
        data = make_demo_curve()
        time_col = "time"
        flux_col = "flux"

        st.info(
            "Synthetic demo active. This is locally generated data, "
            "not a real star or telescope observation."
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV light curve",
            type=["csv"],
            key="light_curve_upload",
            help="Choose a CSV containing numeric time and flux columns.",
        )

        if uploaded_file is None:
            st.info(
                "Upload a CSV to open the interactive analysis workspace."
            )
            st.stop()

        try:
            data = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()

        if data.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        numeric_cols = [
            column
            for column in data.columns
            if pd.api.types.is_numeric_dtype(data[column])
        ]

        if not numeric_cols:
            st.error("No numeric columns were found.")
            st.stop()

        default_time = (
            numeric_cols.index("time")
            if "time" in numeric_cols else 0
        )

        default_flux = (
            numeric_cols.index("flux")
            if "flux" in numeric_cols
            else min(1, len(numeric_cols) - 1)
        )

        col_a, col_b = st.columns(2)

        with col_a:
            time_col = st.selectbox(
                "Time column",
                numeric_cols,
                index=default_time,
            )

        with col_b:
            flux_col = st.selectbox(
                "Flux column",
                numeric_cols,
                index=default_flux,
            )

        if time_col == flux_col:
            st.error("Choose different time and flux columns.")
            st.stop()

    working = data[[time_col, flux_col]].copy()
    working.columns = ["time", "flux"]

    working["time"] = pd.to_numeric(
        working["time"], errors="coerce"
    )
    working["flux"] = pd.to_numeric(
        working["flux"], errors="coerce"
    )

    working = (
        working
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(working) < 8:
        st.error("At least 8 valid time/flux samples are required.")
        st.stop()

    if len(working) > 500_000:
        st.error(
            "This interface currently limits a single analysis to "
            "500,000 rows. Reduce the CSV size and retry."
        )
        st.stop()

    time_values = working["time"].to_numpy(dtype=np.float64)
    flux_values = working["flux"].to_numpy(dtype=np.float32)

    # Fixed: create a signature for the current input.
    current_signature = signal_signature(time_values, flux_values)

    # If the user changes the signal, invalidate the old result.
    if (
        st.session_state.get("startrace_result_signature")
        != current_signature
    ):
        st.session_state.pop("startrace_last_result", None)
        st.session_state.pop("startrace_last_metrics", None)
        st.session_state.pop("startrace_result_signature", None)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Valid samples", f"{len(working):,}", "After cleaning")

    with c2:
        metric_card(
            "Flux mean",
            f"{np.mean(flux_values):.6g}",
            "Input series",
        )

    with c3:
        metric_card(
            "Flux standard deviation",
            f"{np.std(flux_values):.5g}",
            "Input series",
        )

    with c4:
        metric_card(
            "Time span",
            f"{time_values[-1] - time_values[0]:.5g}",
            "Units as supplied",
        )

    st.markdown(
        '<div class="section-label">01 / Input signal</div>',
        unsafe_allow_html=True,
    )

    fig = plot_light_curve(
        time_values,
        flux_values,
        "Input flux",
        color="#5598ff",
        title="Input light curve",
        x_label="Time (uploaded units)",
        y_label="Flux (uploaded units)",
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Inspect cleaned input rows"):
        st.dataframe(working.head(100), use_container_width=True)

    st.markdown(
        '<div class="section-label">02 / Inference configuration</div>',
        unsafe_allow_html=True,
    )

    denoiser_available = (
        assets is not None
        and "Denoising model" in assets["models"]
    )

    if not denoiser_available:
        st.error(
            "The denoising model is unavailable. Check System Diagnostics."
        )

    else:
        left, right = st.columns(2)

        with left:
            normalization = st.selectbox(
                "Input normalization",
                ["Z-score", "Min-max", "Raw values"],
                index=0,
                key="inference_normalization",
                help=(
                    "Use the original training normalization if known. "
                    "The app cannot infer training preprocessing "
                    "automatically from the model file."
                ),
            )

        with right:
            stride = st.slider(
                "Window stride",
                min_value=32,
                max_value=256,
                value=128,
                step=32,
                key="inference_stride",
                help=(
                    "Smaller strides increase overlap between "
                    "the 256-point windows."
                ),
            )

        st.caption(
            "Window length: 256 samples. Unknown training preprocessing "
            "can make model output scientifically misleading."
        )

        run_inference = st.button(
            "▶ Run denoising inference",
            type="primary",
            use_container_width=True,
        )

        if run_inference:
            try:
                model = assets["models"]["Denoising model"]

                prepared, restore = prepare_flux(
                    flux_values,
                    normalization,
                )

                with st.spinner("Running model inference..."):
                    predicted_normalized = denoise_light_curve(
                        model,
                        prepared,
                        stride=stride,
                    )

                predicted = restore(predicted_normalized)
                residuals = flux_values - predicted

                rmse = float(
                    np.sqrt(np.mean(residuals ** 2))
                )
                mae = float(
                    np.mean(np.abs(residuals))
                )

                input_std = float(np.std(flux_values))
                relative_rmse = (
                    rmse / input_std
                    if input_std > 1e-12
                    else float("nan")
                )

                result = pd.DataFrame({
                    "time": time_values,
                    "input_flux": flux_values,
                    "model_reconstruction": predicted,
                    "residual": residuals,
                })

                # Fixed: store both results and the exact input signature.
                st.session_state["startrace_last_result"] = result
                st.session_state["startrace_last_metrics"] = {
                    "rmse": rmse,
                    "mae": mae,
                    "relative_rmse": relative_rmse,
                    "normalization": normalization,
                    "stride": stride,
                }
                st.session_state["startrace_result_signature"] = (
                    current_signature
                )

                st.success("Inference completed.")

            except Exception as exc:
                st.error(
                    f"Inference failed: {type(exc).__name__}: {exc}"
                )

    # Only show the output if it belongs to the current input.
    result = st.session_state.get("startrace_last_result")
    result_signature = st.session_state.get(
        "startrace_result_signature"
    )

    if (
        result is not None
        and result_signature == current_signature
    ):
        st.markdown(
            '<div class="section-label">03 / Model reconstruction</div>',
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(11, 4))
        fig.patch.set_facecolor("#091524")
        ax.set_facecolor("#091524")

        ax.plot(
            result["time"],
            result["input_flux"],
            color="#5598ff",
            linewidth=0.8,
            alpha=0.75,
            label="Input flux",
        )

        ax.plot(
            result["time"],
            result["model_reconstruction"],
            color="#53e6ff",
            linewidth=1.15,
            label="Model reconstruction",
        )

        ax.set_title(
            "Input vs model reconstruction",
            color="#e4f3ff",
            loc="left",
        )
        ax.set_xlabel("Time", color="#88a3b9")
        ax.set_ylabel("Flux", color="#88a3b9")
        ax.tick_params(colors="#88a3b9")
        ax.grid(True, color="#1b3448", linewidth=0.55)

        for spine in ax.spines.values():
            spine.set_color("#19344a")

        ax.legend(
            facecolor="#0d1b2d",
            edgecolor="#19344a",
            labelcolor="#e4f3ff",
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        metrics = st.session_state.get(
            "startrace_last_metrics", {}
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "RMSE",
                f"{metrics.get('rmse', float('nan')):.6g}",
            )

        with m2:
            st.metric(
                "MAE",
                f"{metrics.get('mae', float('nan')):.6g}",
            )

        with m3:
            st.metric(
                "RMSE / input std",
                f"{metrics.get('relative_rmse', float('nan')):.5g}",
            )

        st.markdown(
            '<div class="section-label">04 / Residual inspection</div>',
            unsafe_allow_html=True,
        )

        fig = plot_light_curve(
            result["time"],
            result["residual"],
            "Input − reconstruction",
            color="#ffc66d",
            title="Residual series",
            x_label="Time",
            y_label="Residual flux",
        )

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.warning(
            "Residuals can contain noise, genuine astrophysical variability, "
            "model error, or preprocessing effects. They are not confirmed "
            "event detections."
        )

        st.download_button(
            "↓ Download analysis CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="star_trace_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("View output table"):
            st.dataframe(
                result.head(500),
                use_container_width=True,
            )


# ============================================================
# PAGE 3: MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":
    heading(
        "03 / Model registry",
        "Model Intelligence",
        "Inspect the included Keras artifacts. Architecture metadata "
        "does not independently validate scientific performance.",
    )

    if not assets or not assets["models"]:
        st.error("No Keras models are currently available.")

    else:
        model_name = st.selectbox(
            "Select model",
            list(assets["models"].keys()),
        )

        model = assets["models"][model_name]
        info = assets["model_info"][model_name]

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Total parameters",
                f"{info['parameters']:,}",
                "Keras model",
            )

        with c2:
            metric_card(
                "Input shape",
                info["input_shape"],
                "Model metadata",
            )

        with c3:
            metric_card(
                "Output shape",
                info["output_shape"],
                "Model metadata",
            )

        st.markdown(
            '<div class="section-label">Artifact details</div>',
            unsafe_allow_html=True,
        )

        st.code(
            f"Model: {model_name}\n"
            f"File: {info['filename']}\n"
            f"Archive path: {info['archive_member']}\n"
            f"Input shape: {info['input_shape']}\n"
            f"Output shape: {info['output_shape']}\n"
            f"Parameters: {info['parameters']:,}",
            language="text",
        )

        with st.expander("Keras model summary"):
            lines = []
            model.summary(print_fn=lines.append)
            st.code("\n".join(lines), language="text")

        if model_name == "Autoencoder":
            st.info(
                "The autoencoder is available for inspection. Its input "
                "preprocessing and output interpretation may differ from "
                "those of the denoising model."
            )

        st.markdown(
            '<div class="section-label">Interpretation limits</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            - Parameter count does not measure scientific accuracy.
            - Output depends on training data and preprocessing.
            - A reconstruction is not automatically an event classification.
            - Independent validation is required before making discovery claims.
            """
        )


# ============================================================
# PAGE 4: RESEARCH ARCHIVE
# ============================================================

elif page == "Research Archive":
    heading(
        "04 / Research data",
        "Research Archive",
        "Browse and export the four CSV files bundled with STAR-TRACE.",
    )

    tables = assets["tables"] if assets else {}

    if not tables:
        st.error("No research CSVs could be loaded.")

    else:
        selected = st.selectbox(
            "Select research table",
            list(tables.keys()),
        )

        table = tables[selected]

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Rows", f"{len(table):,}", "CSV data")

        with c2:
            metric_card(
                "Columns",
                str(len(table.columns)),
                "Available fields",
            )

        with c3:
            metric_card(
                "Missing cells",
                f"{int(table.isna().sum().sum()):,}",
                "Entire table",
            )

        st.markdown(
            '<div class="section-label">Table preview</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            table.head(500),
            use_container_width=True,
        )

        st.download_button(
            "↓ Download selected CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=selected,
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Column types and missing values"):
            details = pd.DataFrame({
                "column": table.columns,
                "dtype": [
                    str(table[col].dtype)
                    for col in table.columns
                ],
                "missing": [
                    int(table[col].isna().sum())
                    for col in table.columns
                ],
                "unique_values": [
                    int(table[col].nunique(dropna=True))
                    for col in table.columns
                ],
            })

            st.dataframe(details, use_container_width=True)

        with st.expander("Numeric summary"):
            numeric = table.select_dtypes(include=np.number)

            if numeric.empty:
                st.info("This table has no numeric columns.")
            else:
                st.dataframe(
                    numeric.describe().T,
                    use_container_width=True,
                )

        st.markdown(
            '<div class="section-label">Included research assets</div>',
            unsafe_allow_html=True,
        )

        for filename in CSV_FILES:
            status = "Loaded" if filename in tables else "Not found"
            st.markdown(f"- `{filename}` — **{status}**")


# ============================================================
# PAGE 5: SYSTEM DIAGNOSTICS
# ============================================================

elif page == "System Diagnostics":
    heading(
        "05 / System health",
        "System Diagnostics",
        "Runtime and asset checks. These are not telescope telemetry "
        "or independent scientific validation.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Python",
            os.sys.version.split()[0],
            "Runtime",
        )

    with c2:
        metric_card(
            "Streamlit",
            st.__version__,
            "Interface framework",
        )

    with c3:
        metric_card(
            "TensorFlow",
            tf.__version__,
            "Model runtime",
        )

    st.markdown(
        '<div class="section-label">Deployment archive</div>',
        unsafe_allow_html=True,
    )

    if assets:
        st.success(
            f"Archive loaded: {assets['archive_entries']} entries."
        )

        for label, filename in MODEL_FILES.items():
            status = "Loaded" if label in assets["models"] else "Unavailable"
            st.markdown(f"- `{filename}` — **{status}**")

        for filename in CSV_FILES:
            status = "Loaded" if filename in assets["tables"] else "Unavailable"
            st.markdown(f"- `{filename}` — **{status}**")

        if assets["asset_errors"]:
            with st.expander("Asset warnings"):
                for message in assets["asset_errors"]:
                    st.warning(message)

        with st.expander("Model input/output compatibility"):
            for label, info in assets["model_info"].items():
                st.markdown(f"**{label}**")
                st.code(
                    f"Input: {info['input_shape']}\n"
                    f"Output: {info['output_shape']}\n"
                    f"Parameters: {info['parameters']:,}",
                    language="text",
                )

    st.markdown(
        '<div class="section-label">Compute devices</div>',
        unsafe_allow_html=True,
    )

    try:
        devices = tf.config.list_physical_devices()

        if devices:
            for device in devices:
                st.code(
                    f"{device.device_type}: {device.name}"
                )
        else:
            st.info("TensorFlow did not report a physical device.")

    except Exception as exc:
        st.warning(f"Could not inspect devices: {exc}")

    st.markdown(
        '<div class="section-label">Data integrity boundaries</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">
                No live observatory connection
            </div>
            <div class="panel-copy">
                STAR-TRACE does not connect to a telescope or retrieve
                live astronomical telemetry. The demonstration is generated
                locally, while uploaded data remains the selected input.
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">
                No automatic discovery claims
            </div>
            <div class="panel-copy">
                The interface does not independently verify exoplanets,
                transits, stellar events, or other discoveries. Scientific
                interpretation requires suitable labels, validation,
                and independent confirmation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#617d95;
        font-size:.76rem;
        font-family:'IBM Plex Mono',monospace;
        line-height:1.8
    ">
        STAR-TRACE / MISSION CONTROL<br>
        AI-ASSISTED LIGHT-CURVE ANALYSIS · NO LIVE TELEMETRY CONNECTED
    </div>
    """,
    unsafe_allow_html=True,
)
