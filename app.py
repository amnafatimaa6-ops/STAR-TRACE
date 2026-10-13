
import io
import os
import tempfile
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    lk = None
    LIGHTKURVE_AVAILABLE = False


# ============================================================
# STAR-TRACE | Mission Control
# ============================================================

st.set_page_config(
    page_title="STAR-TRACE | Mission Control",
    page_icon="✦",
    layout="wide",
)

ASSET_ZIP_URL = (
    "https://raw.githubusercontent.com/"
    "amnafatimaa6-ops/STAR-TRACE/main/"
    "STARTRACE_deployment_assets.zip"
)

WINDOW_SIZE = 256
WINDOW_STRIDE = 128

# -------------------- MISSION CONTROL CSS --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%,
        rgba(20,80,115,.22), transparent 38%),
        linear-gradient(180deg, #07111c, #091522 50%, #07101a);
    color: #e6f0f7;
}

[data-testid="stSidebar"] {
    background: #09131f;
    border-right: 1px solid #203449;
}

h1, h2, h3 {
    color: #eaf7ff !important;
}

p, label, li {
    color: #c0d0de;
}

div[data-testid="stMetric"] {
    background: rgba(14,31,47,.8);
    border: 1px solid #244158;
    padding: 14px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #b9f3ff;
    font-family: 'IBM Plex Mono', monospace;
}

.stButton button, .stDownloadButton button {
    border: 1px solid #3a819b;
    border-radius: 9px;
    background: #102b3c;
    color: #dff8ff;
}

.mission-tag {
    font: 600 11px 'IBM Plex Mono', monospace;
    color: #7ee8fa;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.mission-panel {
    background: rgba(11,28,43,.75);
    border: 1px solid #234157;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 8px 0 16px 0;
}

code {
    color: #a8edff !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- ASSET LOADER --------------------

def download_assets():
    cached = st.session_state.get("_startrace_assets_dir")

    if cached and Path(cached).exists():
        return Path(cached)

    try:
        request = urllib.request.Request(
            ASSET_ZIP_URL,
            headers={"User-Agent": "STAR-TRACE-Streamlit/1.0"},
        )

        with urllib.request.urlopen(request, timeout=35) as response:
            archive_bytes = response.read()

        root = Path(tempfile.mkdtemp(prefix="startrace_"))

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            archive.extractall(root)

        st.session_state["_startrace_assets_dir"] = str(root)
        return root

    except Exception as exc:
        st.warning(
            "Could not download the deployment asset ZIP. "
            "CSV uploads and the synthetic demo remain available. "
            f"Details: {exc}"
        )
        return None


@st.cache_resource(show_spinner="Loading STAR-TRACE models...")
def load_assets(root_string):
    import tensorflow as tf

    root = Path(root_string)
    models = {}
    tables = {}
    errors = []

    for path in root.rglob("*.keras"):
        try:
            models[path.name] = tf.keras.models.load_model(
                str(path), compile=False
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    for path in root.rglob("*.csv"):
        try:
            tables[path.name] = pd.read_csv(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return models, tables, errors


# -------------------- SYNTHETIC DEMO --------------------

def make_synthetic_signal(n=1800, seed=23):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 12, n)

    flux = (
        1.0
        + 0.012 * np.sin(2 * np.pi * time / 8.5)
        + 0.018 * np.sin(2 * np.pi * time * 1.6)
        + rng.normal(0, 0.012, n)
    )

    # Simulated transit-like dips for demonstration only.
    for center, depth, width in [
        (3.1, 0.07, 0.055),
        (7.0, 0.065, 0.06),
        (10.9, 0.075, 0.05),
    ]:
        flux -= depth * np.exp(
            -0.5 * ((time - center) / width) ** 2
        )

    return pd.DataFrame({
        "time": time,
        "flux": flux,
    })


# -------------------- CSV PROCESSING --------------------

def read_lightcurve_csv(file):
    df = pd.read_csv(file)

    if df.empty:
        raise ValueError("The CSV contains no observations.")

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    time_candidates = [
        "time", "btjd", "bjd", "jd", "timestamp", "t"
    ]

    flux_candidates = [
        "pdcsap_flux", "sap_flux", "flux",
        "normalized_flux", "normalised_flux",
        "relative_flux", "brightness",
    ]

    time_col = next(
        (lookup[c] for c in time_candidates if c in lookup),
        None,
    )

    flux_col = next(
        (lookup[c] for c in flux_candidates if c in lookup),
        None,
    )

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if flux_col is None:
        if not numeric_cols:
            raise ValueError("No numeric flux column was found.")
        flux_col = numeric_cols[-1]

    if time_col is None:
        other_cols = [
            c for c in numeric_cols if c != flux_col
        ]

        if other_cols:
            time_values = df[other_cols[0]].to_numpy()
            time_label = str(other_cols[0])
        else:
            time_values = np.arange(len(df))
            time_label = "Sample"
    else:
        time_values = df[time_col].to_numpy()
        time_label = str(time_col)

    clean = pd.DataFrame({
        "time": pd.to_numeric(
            pd.Series(time_values), errors="coerce"
        ),
        "flux": pd.to_numeric(
            df[flux_col], errors="coerce"
        ),
    })

    clean = clean.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    clean = (
        clean.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    if len(clean) < WINDOW_SIZE:
        raise ValueError(
            f"Need at least {WINDOW_SIZE} valid observations. "
            f"Found {len(clean)}."
        )

    clean.attrs["time_label"] = time_label
    clean.attrs["flux_label"] = str(flux_col)

    return clean


# -------------------- REAL TESS DATA --------------------

def fetch_tess_lightcurve(target):
    if not LIGHTKURVE_AVAILABLE:
        raise RuntimeError(
            "Live TESS retrieval requires lightkurve. "
            "Add it to requirements.txt and redeploy, "
            "or upload a CSV light curve instead."
        )

    results = lk.search_lightcurve(
        target, mission="TESS"
    )

    if len(results) == 0:
        raise RuntimeError(
            f"No TESS light-curve products found for {target}."
        )

    lightcurve = None
    errors = []

    for i in range(min(len(results), 5)):
        try:
            lightcurve = results[i].download()
            if lightcurve is not None:
                break
        except Exception as exc:
            errors.append(str(exc))

    if lightcurve is None:
        raise RuntimeError(
            "TESS download failed: " + " | ".join(errors[-2:])
        )

    try:
        flux = lightcurve.PDCSAP_FLUX
        flux_name = "PDCSAP_FLUX"
    except Exception:
        flux = lightcurve.SAP_FLUX
        flux_name = "SAP_FLUX"

    df = pd.DataFrame({
        "time": np.asarray(lightcurve.time.value, dtype=float),
        "flux": np.asarray(flux.value, dtype=float),
    })

    df = df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    df = df[df["flux"] > 0]
    df = (
        df.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    if len(df) < WINDOW_SIZE:
        raise RuntimeError(
            f"Only {len(df)} valid observations found; "
            f"at least {WINDOW_SIZE} are required."
        )

    df.attrs["time_label"] = "TESS time (BTJD)"
    df.attrs["flux_label"] = flux_name

    return df, f"Real TESS observation: {target} ({flux_name})"


# -------------------- MODEL PREPROCESSING --------------------

def normalize_flux(flux, mode):
    flux = np.asarray(flux, dtype=np.float32)

    if mode == "Z-score":
        center = float(np.mean(flux))
        scale = float(np.std(flux))
        scale = scale if scale > 1e-8 else 1.0

        return (
            (flux - center) / scale,
            lambda y: np.asarray(y) * scale + center,
        )

    if mode == "Robust":
        center = float(np.median(flux))
        low, high = np.percentile(flux, [5, 95])
        scale = float(high - low)
        scale = scale if scale > 1e-8 else 1.0

        return (
            (flux - center) / scale,
            lambda y: np.asarray(y) * scale + center,
        )

    if mode == "Min-max":
        low = float(np.min(flux))
        scale = float(np.max(flux) - low)
        scale = scale if scale > 1e-8 else 1.0

        return (
            (flux - low) / scale,
            lambda y: np.asarray(y) * scale + low,
        )

    return flux.copy(), lambda y: np.asarray(y)


def reconstruct_signal(model, signal):
    """Sliding-window inference for compatible sequence models."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    n = len(signal)

    if n < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} samples are required."
        )

    starts = list(
        range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE)
    )

    last_start = n - WINDOW_SIZE

    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    windows = np.stack([
        signal[s:s + WINDOW_SIZE] for s in starts
    ])[..., None]

    predictions = np.asarray(
        model.predict(windows, verbose=0)
    )

    if predictions.ndim == 2:
        predictions = predictions[..., None]

    if (
        predictions.ndim != 3
        or predictions.shape[1] != WINDOW_SIZE
    ):
        raise ValueError(
            f"Model output shape {predictions.shape} is "
            f"incompatible with {WINDOW_SIZE}-sample windows."
        )

    total = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    for start, prediction in zip(starts, predictions):
        chunk = prediction.reshape(-1)

        total[start:start + WINDOW_SIZE] += chunk
        counts[start:start + WINDOW_SIZE] += 1

    counts[counts == 0] = 1

    return (total / counts).astype(np.float32)


# -------------------- AI PREDICTION PLOTS --------------------

def plot_prediction(time, observed, predicted, title, time_label):
    residual = observed - predicted

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    fig.patch.set_facecolor("#0a1521")

    for ax in axes:
        ax.set_facecolor("#0c1b29")
        ax.tick_params(colors="#a9c1d1")

        for spine in ax.spines.values():
            spine.set_color("#2b465b")

        ax.grid(True, alpha=0.18, color="#8fa8bb")

        ax.xaxis.label.set_color("#c8dbe8")
        ax.yaxis.label.set_color("#c8dbe8")

    axes[0].plot(
        time, observed,
        color="#78b7ff",
        linewidth=1,
        label="Observed flux",
    )

    axes[0].plot(
        time, predicted,
        color="#ffca75",
        linewidth=1.4,
        label="AI reconstruction",
    )

    axes[0].set_ylabel("Flux")
    axes[0].set_title(
        title, color="#e8f5ff", loc="left"
    )

    axes[0].legend(
        facecolor="#102436",
        edgecolor="#2b465b",
        labelcolor="#dcebf5",
    )

    axes[1].plot(
        time, residual,
        color="#8ce6d0",
        linewidth=0.9,
        label="Observed − reconstructed",
    )

    axes[1].axhline(
        0, color="white", linewidth=0.8, alpha=0.5
    )

    axes[1].set_xlabel(time_label)
    axes[1].set_ylabel("Residual")

    axes[1].legend(
        facecolor="#102436",
        edgecolor="#2b465b",
        labelcolor="#dcebf5",
    )

    fig.tight_layout()

    return fig, residual


# -------------------- LOAD EXISTING ASSETS --------------------

asset_root = download_assets()

models = {}
research_tables = {}
load_errors = []

if asset_root:
    try:
        models, research_tables, load_errors = load_assets(
            str(asset_root)
        )
    except Exception as exc:
        st.warning(f"Asset loading error: {exc}")

model_names = list(models.keys())

denoiser_name = next(
    (
        name for name in model_names
        if "denois" in name.lower()
    ),
    model_names[0] if model_names else None,
)


# -------------------- NAVIGATION FIX --------------------

NAVIGATION_OPTIONS = [
    "Mission Overview",
    "Signal Laboratory",
    "Model Intelligence",
    "Research Archive",
    "System Diagnostics",
]

# Set navigation state BEFORE creating the radio widget.
if st.session_state.pop(
    "navigate_to_signal_lab", False
):
    st.session_state["startrace_page"] = "Signal Laboratory"

if st.session_state.pop(
    "navigate_to_archive", False
):
    st.session_state["startrace_page"] = "Research Archive"


with st.sidebar:
    st.markdown(
        '<div class="mission-tag">ASTROPHYSICS · AI SYSTEM</div>',
        unsafe_allow_html=True,
    )

    st.markdown("# ✦ STAR-TRACE")
    st.caption("AI-assisted stellar light-curve analysis")

    page = st.radio(
        "NAVIGATION",
        NAVIGATION_OPTIONS,
        key="startrace_page",
    )

    st.divider()
    st.markdown("**MISSION STATUS**")
    st.caption(
        "● Asset bundle: "
        + ("loaded" if asset_root else "unavailable")
    )
    st.caption(f"● Keras models: {len(models)}")
    st.caption(f"● Research tables: {len(research_tables)}")
    st.caption(
        "● Live TESS: "
        + ("available" if LIGHTKURVE_AVAILABLE else "CSV upload")
    )


st.markdown(
    '<div class="mission-tag">MISSION CONTROL / STELLAR SIGNAL INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.title("STAR-TRACE")

st.markdown(
    "Analyze stellar brightness time series and compare "
    "the observations against the AI model's reconstruction."
)


# ============================================================
# PAGE 1: MISSION OVERVIEW
# ============================================================

if page == "Mission Overview":

    st.markdown(
        '<div class="mission-panel">',
        unsafe_allow_html=True,
    )

    st.markdown("### Mission brief")

    st.write(
        "STAR-TRACE is an AI-assisted light-curve analysis "
        "interface. Explore a synthetic demo, upload real "
        "observations, or retrieve public TESS data."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Keras models", len(models))
    c2.metric("Research tables", len(research_tables))
    c3.metric("Window size", WINDOW_SIZE)
    c4.metric(
        "TESS retrieval",
        "Available" if LIGHTKURVE_AVAILABLE else "CSV mode",
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Signal Laboratory")
        st.write(
            "Load observations, run AI inference, and visualize "
            "observed flux, reconstructed flux, and residuals."
        )

        if st.button(
            "Open Signal Laboratory →",
            use_container_width=True,
        ):
            st.session_state["navigate_to_signal_lab"] = True
            st.rerun()

    with right:
        st.markdown("### Research Archive")
        st.write(
            "Inspect model evaluation, model comparisons, "
            "morphology features, and clustering results."
        )

        if st.button(
            "Open Research Archive →",
            use_container_width=True,
        ):
            st.session_state["navigate_to_archive"] = True
            st.rerun()

    st.info(
        "The synthetic demo is not telescope data. "
        "Model outputs are not confirmed astronomical discoveries."
    )


# ============================================================
# PAGE 2: SIGNAL LABORATORY
# ============================================================

elif page == "Signal Laboratory":

    st.header("Signal Laboratory")

    st.write(
        "Choose a source and run the loaded model to see "
        "what it reconstructs from the observed light curve."
    )

    source_mode = st.radio(
        "DATA SOURCE",
        [
            "Synthetic demo",
            "Upload CSV",
            "Fetch real TESS observation",
        ],
        horizontal=True,
        key="signal_data_source",
    )

    data = None
    data_label = ""
    time_label = "Time"
    source_error = None

    # -------- Synthetic mode --------

    if source_mode == "Synthetic demo":
        data = make_synthetic_signal()
        data_label = "Synthetic demonstration"
        time_label = "Demo time (arbitrary units)"

        st.caption(
            "Synthetic demonstration only — not a real observation."
        )

    # -------- CSV mode --------

    elif source_mode == "Upload CSV":

        uploaded = st.file_uploader(
            "Upload a stellar light-curve CSV",
            type=["csv"],
            help=(
                "Include time and flux columns. Common flux "
                "names: flux, PDCSAP_FLUX, SAP_FLUX."
            ),
        )

        if uploaded is not None:
            try:
                data = read_lightcurve_csv(uploaded)
                data_label = f"CSV: {uploaded.name}"
                time_label = data.attrs.get(
                    "time_label", "Time"
                )

                st.success(
                    f"Loaded {len(data):,} valid observations."
                )

                st.caption(
                    "Flux column: "
                    + data.attrs.get("flux_label", "flux")
                )

            except Exception as exc:
                source_error = str(exc)

    # -------- Real TESS mode --------

    elif source_mode == "Fetch real TESS observation":

        target = st.text_input(
            "TESS target",
            value="TIC 307210830",
            help=(
                "Enter a TIC ID or a resolvable target name. "
                "Archive availability varies by target."
            ),
        )

        if st.button(
            "Retrieve TESS light curve",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Searching public TESS observations..."
                ):
                    fetched, label = fetch_tess_lightcurve(
                        target.strip()
                    )

                st.session_state["_startrace_tess_data"] = fetched
                st.session_state["_startrace_tess_label"] = label

            except Exception as exc:
                source_error = str(exc)

        if st.session_state.get("_startrace_tess_data") is not None:
            data = st.session_state["_startrace_tess_data"]

            data_label = st.session_state.get(
                "_startrace_tess_label",
                "Real TESS observation",
            )

            time_label = data.attrs.get(
                "time_label", "TESS time"
            )

    if source_error:
        st.error(source_error)

    # -------- Preview and inference --------

    if data is not None:

        time_label = data.attrs.get(
            "time_label", time_label
        )

        with st.expander(
            "Preview observations",
            expanded=False,
        ):
            st.dataframe(
                data.head(25),
                use_container_width=True,
            )

        st.markdown("### AI model")

        if not denoiser_name:
            st.error(
                "No compatible Keras model was loaded. "
                "Check the deployment ZIP and model files."
            )

        else:
            model = models[denoiser_name]

            try:
                st.caption(
                    f"Loaded model: {denoiser_name} · "
                    f"Input: {model.input_shape} · "
                    f"Output: {model.output_shape}"
                )
            except Exception:
                st.caption(f"Loaded model: {denoiser_name}")

            normalization = st.selectbox(
                "Input normalization",
                [
                    "Z-score",
                    "Robust",
                    "Min-max",
                    "Raw",
                ],
                help=(
                    "The correct choice must match the model's "
                    "training preprocessing. If unknown, test "
                    "carefully and do not treat the output as validated."
                ),
            )

            point_limit = st.slider(
                "Maximum observations to process",
                min_value=512,
                max_value=6000,
                value=min(
                    3000,
                    max(512, (len(data) // 128) * 128),
                ),
                step=128,
            )

            if st.button(
                "Run AI reconstruction",
                type="primary",
                use_container_width=True,
            ):

                try:
                    with st.spinner(
                        "Running model inference..."
                    ):
                        selected = data.iloc[
                            :min(len(data), point_limit)
                        ].copy()

                        observed = selected[
                            "flux"
                        ].to_numpy(dtype=np.float32)

                        time_values = selected[
                            "time"
                        ].to_numpy()

                        normalized, inverse = normalize_flux(
                            observed,
                            normalization,
                        )

                        reconstructed_norm = reconstruct_signal(
                            model,
                            normalized,
                        )

                        reconstructed = inverse(
                            reconstructed_norm
                        )

                        residual = observed - reconstructed

                        rmse = float(
                            np.sqrt(np.mean(residual ** 2))
                        )

                        mae = float(
                            np.mean(np.abs(residual))
                        )

                        st.session_state[
                            "_startrace_last_result"
                        ] = {
                            "time": time_values,
                            "observed": observed,
                            "reconstructed": reconstructed,
                            "residual": residual,
                            "source": data_label,
                            "time_label": time_label,
                            "normalization": normalization,
                            "n": len(observed),
                            "rmse": rmse,
                            "mae": mae,
                        }

                except Exception as exc:
                    st.error(
                        f"AI inference failed: {exc}"
                    )

            result = st.session_state.get(
                "_startrace_last_result"
            )

            if result and result["source"] == data_label:

                st.markdown("### What is the AI predicting?")

                k1, k2, k3, k4 = st.columns(4)

                k1.metric(
                    "Observations",
                    f"{result['n']:,}",
                )

                k2.metric(
                    "Reconstruction RMSE",
                    f"{result['rmse']:.5g}",
                )

                k3.metric(
                    "Mean absolute error",
                    f"{result['mae']:.5g}",
                )

                k4.metric(
                    "Normalization",
                    result["normalization"],
                )

                fig, residual = plot_prediction(
                    result["time"],
                    result["observed"],
                    result["reconstructed"],
                    f"{result['source']} | Observed vs AI",
                    result["time_label"],
                )

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

                export = pd.DataFrame({
                    "time": result["time"],
                    "observed_flux": result["observed"],
                    "ai_reconstructed_flux": result["reconstructed"],
                    "residual": result["residual"],
                })

                st.download_button(
                    "Download AI results as CSV",
                    data=export.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="startrace_ai_reconstruction.csv",
                    mime="text/csv",
                )

                st.markdown("### How to read this plot")

                st.markdown("""
                - **Blue — Observed flux:** the brightness measurements from your data.
                - **Yellow — AI reconstruction:** the signal produced by the loaded model.
                - **Green — Residual:** observed flux minus reconstructed flux.
                - **RMSE / MAE:** numerical differences between observations and reconstruction.
                """)

                st.warning(
                    "This is signal reconstruction, not future-time "
                    "forecasting. A residual or dip does not establish "
                    "a planet, transit, flare, or other discovery. "
                    "Confirm the model's training normalization and "
                    "validate against labelled observations before "
                    "making scientific claims."
                )


# ============================================================
# PAGE 3: MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.header("Model Intelligence")

    if not models:
        st.warning("No Keras models are available.")
    else:
        for name, model in models.items():

            with st.expander(name, expanded=False):

                try:
                    st.write("Input shape:", model.input_shape)
                    st.write("Output shape:", model.output_shape)
                    st.write("Layers:", len(model.layers))
                    st.write(
                        "Parameters:",
                        f"{model.count_params():,}",
                    )

                    lines = []

                    model.summary(
                        print_fn=lambda line: lines.append(line)
                    )

                    st.code("\n".join(lines[:45]))

                except Exception as exc:
                    st.caption(
                        f"Could not display model details: {exc}"
                    )

    if load_errors:
        with st.expander("Model loading warnings"):
            for error in load_errors:
                st.write(error)

    st.markdown("### Inference protocol")

    st.write(
        f"Sliding windows: {WINDOW_SIZE} samples; "
        f"stride: {WINDOW_STRIDE} samples. Overlapping "
        "reconstructions are averaged."
    )

    st.info(
        "The preprocessing used in training must be verified. "
        "Changing normalization can substantially change predictions."
    )


# ============================================================
# PAGE 4: RESEARCH ARCHIVE
# ============================================================

elif page == "Research Archive":

    st.header("Research Archive")

    if not research_tables:
        st.warning(
            "No research tables were loaded. Check the ZIP contents."
        )

    for filename, table in research_tables.items():

        with st.expander(
            f"{filename} · {len(table):,} rows",
            expanded=False,
        ):

            st.dataframe(
                table,
                use_container_width=True,
            )

            st.download_button(
                f"Download {filename}",
                data=table.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=filename,
                mime="text/csv",
                key=f"download_{filename}",
            )


# ============================================================
# PAGE 5: SYSTEM DIAGNOSTICS
# ============================================================

elif page == "System Diagnostics":

    st.header("System Diagnostics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Asset bundle",
        "Loaded" if asset_root else "Unavailable",
    )

    c2.metric("Models loaded", len(models))
    c3.metric("Research tables", len(research_tables))

    st.markdown("### Runtime")

    st.write("Python:", os.sys.version.split()[0])
    st.write("NumPy:", np.__version__)
    st.write("Pandas:", pd.__version__)
    st.write("Live TESS support:", LIGHTKURVE_AVAILABLE)

    if asset_root:
        st.write("Asset directory:", str(asset_root))

    if model_names:
        st.write("Models:", model_names)

    if research_tables:
        st.write("Tables:", list(research_tables.keys()))

    if load_errors:
        st.markdown("### Warnings")
        for error in load_errors:
            st.write(error)


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "STAR-TRACE · AI-assisted stellar light-curve analysis · "
    "Model outputs require independent scientific validation."
)
