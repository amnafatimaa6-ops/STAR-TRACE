
import io
import time
import zipfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# STAR-TRACE | AI-ASSISTED STELLAR LIGHT-CURVE ANALYSIS
# ============================================================

st.set_page_config(
    page_title="STAR-TRACE | Stellar Analysis",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


REPOSITORY_URL = (
    "https://raw.githubusercontent.com/"
    "amnafatimaa6-ops/STAR-TRACE/main/"
    "STARTRACE_deployment_assets.zip"
)

MODEL_FILES = [
    "startrace_denoising_v2.keras",
    "startrace_autoencoder.keras",
]

CSV_FILES = [
    "startrace_v2_final_evaluation_summary.csv",
    "startrace_v2_model_comparison.csv",
    "startrace_morphology_features.csv",
    "startrace_v2_morphology_clusters.csv",
]


# ============================================================
# DOWNLOAD GITHUB ARCHIVE
# ============================================================

def download_archive(url, attempts=3):
    last_error = None

    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "STAR-TRACE-Streamlit/1.0",
                    "Accept": "application/vnd.github+json, "
                              "application/zip, */*",
                },
            )

            with urllib.request.urlopen(
                request, timeout=120
            ) as response:
                archive_bytes = response.read()

            if not archive_bytes:
                raise ValueError("GitHub returned an empty archive.")

            if not zipfile.is_zipfile(io.BytesIO(archive_bytes)):
                preview = archive_bytes[:200].decode(
                    "utf-8", errors="replace"
                )
                raise ValueError(
                    "The downloaded response is not a valid ZIP. "
                    f"Response preview: {preview}"
                )

            return archive_bytes

        except Exception as exc:
            last_error = exc

            if attempt < attempts - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Could not download the GitHub archive after "
        f"{attempts} attempts. Last error: {last_error}"
    ) from last_error


# ============================================================
# LOAD MODELS AND CSV FILES
# ============================================================

@st.cache_resource(show_spinner=False)
def load_assets(url):
    archive_bytes = download_archive(url)

    models = {}
    tables = {}
    temp_dir = tempfile.mkdtemp(prefix="startrace_")

    try:
        with zipfile.ZipFile(
            io.BytesIO(archive_bytes), "r"
        ) as archive:

            names = archive.namelist()

            # Match files by basename, including files at the
            # repository root or inside nested directories.
            def find_file(filename):
                matches = [
                    name for name in names
                    if Path(name).name == filename
                    and not name.endswith("/")
                ]

                if not matches:
                    return None

                if len(matches) > 1:
                    st.warning(
                        f"Multiple copies of {filename} were found. "
                        "Using the first match."
                    )

                return matches[0]

            missing_models = [
                filename
                for filename in MODEL_FILES
                if find_file(filename) is None
            ]

            if missing_models:
                raise FileNotFoundError(
                    "Required model files are missing from the "
                    "GitHub ZIP: "
                    + ", ".join(missing_models)
                    + ". Check the exact filenames and ensure "
                    "the files are committed to the main branch."
                )

            # Load required TensorFlow models.
            for filename in MODEL_FILES:
                member = find_file(filename)

                destination = Path(temp_dir) / filename
                destination.write_bytes(archive.read(member))

                models[filename] = tf.keras.models.load_model(
                    str(destination),
                    compile=False,
                )

            # Load optional CSV tables.
            for filename in CSV_FILES:
                member = find_file(filename)

                if member is None:
                    continue

                try:
                    with archive.open(member) as file:
                        tables[filename] = pd.read_csv(file)
                except Exception as exc:
                    st.warning(
                        f"Could not read {filename}: {exc}"
                    )

        return models, tables, names

    except Exception:
        # Do not silently hide model or archive errors.
        raise


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🔭 STAR-TRACE")
    st.caption("Stellar Signal Intelligence")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Model Explorer",
            "Research Data",
            "System Diagnostics",
        ],
    )

    st.divider()

    st.caption("Project repository")
    st.markdown(
        "[View STAR-TRACE on GitHub]"
        "(https://github.com/amnafatimaa6-ops/STAR-TRACE)"
    )

    if st.button("🔄 Reload assets", use_container_width=True):
        load_assets.clear()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("STAR-TRACE")
st.markdown(
    "### AI-assisted stellar light-curve analysis"
)

st.write(
    "A research interface for exploring trained neural networks, "
    "model evaluation tables, and stellar morphology data."
)

st.divider()


# ============================================================
# LOAD ASSETS AND DISPLAY ERRORS
# ============================================================

try:
    with st.spinner(
        "Downloading research assets and loading TensorFlow models..."
    ):
        models, tables, archive_names = load_assets(REPOSITORY_URL)

except Exception as exc:
    st.error("STAR-TRACE could not load its research assets.")

    st.markdown("#### Error details")
    st.exception(exc)

    st.markdown("#### Troubleshooting checklist")
    st.markdown(
        """
        - Confirm the GitHub repository is public.
        - Confirm both `.keras` files exist in the `main` branch ZIP.
        - Check that the filenames match the names in `app.py`.
        - Check the Streamlit deployment logs for the complete traceback.
        - Verify that the TensorFlow version supports the saved models.
        """
    )

    st.link_button(
        "Open GitHub repository",
        "https://github.com/amnafatimaa6-ops/STAR-TRACE",
    )

    st.stop()


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.subheader("Mission overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models loaded", len(models))

    with col2:
        st.metric("Research tables", len(tables))

    with col3:
        st.metric("Archive entries", len(archive_names))

    st.success("Research assets loaded successfully.")

    st.markdown("#### Available capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Neural network models**

            - Denoising model
            - Autoencoder model
            - Model input/output inspection
            """
        )

    with col2:
        st.markdown(
            """
            **Research data**

            - Evaluation summaries
            - Model comparison tables
            - Morphology features
            - Morphology cluster assignments
            """
        )

    st.info(
        "This deployment currently provides model inspection and "
        "research-data exploration. It does not yet perform "
        "light-curve inference or confirm astronomical discoveries."
    )


# ============================================================
# MODEL EXPLORER
# ============================================================

elif page == "Model Explorer":

    st.subheader("Trained model explorer")

    model_names = list(models.keys())

    selected_model = st.selectbox(
        "Select a trained model",
        model_names,
    )

    model = models[selected_model]

    st.markdown(f"**Model:** `{selected_model}`")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input shape**")
        st.code(str(model.input_shape))

    with col2:
        st.markdown("**Output shape**")
        st.code(str(model.output_shape))

    st.markdown("**Number of parameters**")
    st.metric(
        "Total parameters",
        f"{model.count_params():,}",
    )

    if st.checkbox("Show model architecture"):
        model.summary(print_fn=lambda line: None)

        summary_lines = []
        model.summary(print_fn=summary_lines.append)

        st.code("\n".join(summary_lines), language="text")

    st.caption(
        "Inspecting a model does not evaluate its scientific accuracy."
    )


# ============================================================
# RESEARCH DATA
# ============================================================

elif page == "Research Data":

    st.subheader("Research data explorer")

    if not tables:
        st.warning(
            "No optional CSV tables were found in the repository ZIP."
        )

    else:
        selected_table = st.selectbox(
            "Select a research table",
            list(tables.keys()),
        )

        dataframe = tables[selected_table]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", f"{len(dataframe):,}")

        with col2:
            st.metric("Columns", len(dataframe.columns))

        st.dataframe(
            dataframe,
            use_container_width=True,
            height=450,
        )

        csv_data = dataframe.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download displayed table as CSV",
            data=csv_data,
            file_name=selected_table,
            mime="text/csv",
        )

        if st.checkbox("Show descriptive statistics"):
            st.dataframe(
                dataframe.describe(include="all").T,
                use_container_width=True,
            )


# ============================================================
# SYSTEM DIAGNOSTICS
# ============================================================

elif page == "System Diagnostics":

    st.subheader("System diagnostics")

    st.markdown("#### Runtime")

    st.write("TensorFlow version:", tf.__version__)
    st.write("Pandas version:", pd.__version__)
    st.write("Streamlit version:", st.__version__)

    st.markdown("#### Available devices")

    devices = tf.config.list_physical_devices()

    if devices:
        for device in devices:
            st.write(f"- {device.device_type}: {device.name}")
    else:
        st.write("No TensorFlow devices detected.")

    st.markdown("#### Archive files")

    st.write(f"Total archive entries: {len(archive_names)}")

    with st.expander("View archive file listing"):
        st.code("\n".join(archive_names), language="text")

    st.markdown("#### Loaded assets")

    st.write("Models:", list(models.keys()))
    st.write("Tables:", list(tables.keys()))

    st.caption(
        "A CPU-only runtime is acceptable for model inspection. "
        "Inference speed depends on model size and input dimensions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "STAR-TRACE | Experimental scientific AI | "
    "Model predictions require independent scientific validation."
)
