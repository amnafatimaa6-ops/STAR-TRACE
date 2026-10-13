
import io
import time
import zipfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# STAR-TRACE: STELLAR SIGNAL INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="STAR-TRACE | Stellar Signal Intelligence",
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

WINDOW_SIZE = 256


# ============================================================
# DOWNLOAD RESEARCH ASSETS
# ============================================================

def download_archive(url, attempts=3):
    last_error = None

    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "STAR-TRACE-Streamlit/1.0"
                },
            )

            with urllib.request.urlopen(
                request, timeout=120
            ) as response:
                data = response.read()

            if not zipfile.is_zipfile(io.BytesIO(data)):
                raise ValueError(
                    "Downloaded file is not a valid ZIP archive."
                )

            return data

        except Exception as exc:
            last_error = exc

            if attempt < attempts - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Asset download failed: {last_error}"
    ) from last_error


@st.cache_resource(show_spinner=False)
def load_assets(url):
    archive_data = download_archive(url)

    models = {}
    tables = {}
    temporary_directory = tempfile.mkdtemp(
        prefix="startrace_"
    )

    with zipfile.ZipFile(
        io.BytesIO(archive_data), "r"
    ) as archive:

        names = archive.namelist()

        def find_file(filename):
            matches = [
                name for name in names
                if Path(name).name == filename
                and not name.endswith("/")
            ]

            return matches[0] if matches else None

        for filename in MODEL_FILES:
            member = find_file(filename)

            if member is None:
                raise FileNotFoundError(
                    f"Missing model file: {filename}"
                )

            destination = (
                Path(temporary_directory) / filename
            )
            destination.write_bytes(
                archive.read(member)
            )

            models[filename] = tf.keras.models.load_model(
                str(destination),
                compile=False,
            )

        for filename in CSV_FILES:
            member = find_file(filename)

            if member is not None:
                try:
                    tables[filename] = pd.read_csv(
                        io.BytesIO(archive.read(member))
                    )
                except Exception:
                    pass

    return models, tables, names


# ============================================================
# LIGHT-CURVE PREPROCESSING
# ============================================================

def prepare_flux(values, method):
    """Normalise flux and retain the inverse transformation."""

    values = np.asarray(values, dtype=np.float32)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Flux contains missing or infinite values. "
            "Clean these before analysis."
        )

    if method == "Standard score (z-score)":
        centre = float(np.mean(values))
        scale = float(np.std(values))

        if scale < 1e-12:
            scale = 1.0

        prepared = (values - centre) / scale

        def restore(output):
            return output * scale + centre

    elif method == "Min-max [0, 1]":
        minimum = float(np.min(values))
        scale = float(np.max(values) - minimum)

        if scale < 1e-12:
            scale = 1.0

        prepared = (values - minimum) / scale

        def restore(output):
            return output * scale + minimum

    else:
        prepared = values.copy()

        def restore(output):
            return output

    return prepared.astype(np.float32), restore


# ============================================================
# OVERLAPPING 256-POINT INFERENCE
# ============================================================

def denoise_light_curve(
    model,
    flux,
    window_size=256,
    stride=128,
    batch_size=64,
):
    """
    Run inference on overlapping windows and average
    overlapping predictions.

    Short series are edge-padded and cropped back to length.
    Long series receive a final window at the end if required.
    """

    flux = np.asarray(flux, dtype=np.float32)
    original_length = len(flux)

    if original_length == 0:
        raise ValueError("The light curve is empty.")

    if stride < 1 or stride > window_size:
        raise ValueError(
            "Stride must be between 1 and 256."
        )

    if original_length < window_size:
        padded = np.pad(
            flux,
            (0, window_size - original_length),
            mode="edge",
        )

        windows = padded.reshape(
            1, window_size, 1
        )

        prediction = model.predict(
            windows, verbose=0
        )

        prediction = np.asarray(prediction)

        if (
            prediction.ndim != 3
            or prediction.shape[1] != window_size
            or prediction.shape[2] != 1
        ):
            raise ValueError(
                "The denoising model output shape is "
                f"{prediction.shape}; expected "
                "(batch, 256, 1)."
            )

        return prediction[0, :original_length, 0]

    starts = list(
        range(
            0,
            original_length - window_size + 1,
            stride,
        )
    )

    final_start = original_length - window_size

    if starts[-1] != final_start:
        starts.append(final_start)

    windows = np.stack(
        [
            flux[start:start + window_size]
            for start in starts
        ]
    ).astype(np.float32)

    windows = windows[..., np.newaxis]

    predictions = model.predict(
        windows,
        batch_size=batch_size,
        verbose=0,
    )

    predictions = np.asarray(predictions)

    if (
        predictions.ndim != 3
        or predictions.shape[1] != window_size
        or predictions.shape[2] != 1
    ):
        raise ValueError(
            "Unexpected denoising output shape: "
            f"{predictions.shape}. Expected "
            "(batch, 256, 1)."
        )

    reconstructed = np.zeros(
        original_length, dtype=np.float64
    )
    weights = np.zeros(
        original_length, dtype=np.float64
    )

    for start, prediction in zip(
        starts, predictions[:, :, 0]
    ):
        reconstructed[
            start:start + window_size
        ] += prediction

        weights[
            start:start + window_size
        ] += 1.0

    reconstructed /= np.maximum(weights, 1.0)

    return reconstructed.astype(np.float32)


# ============================================================
# CSV UTILITIES
# ============================================================

def read_uploaded_csv(uploaded_file):
    uploaded_file.seek(0)

    dataframe = pd.read_csv(uploaded_file)

    if dataframe.empty:
        raise ValueError("The uploaded CSV is empty.")

    if len(dataframe.columns) < 1:
        raise ValueError("No columns were found.")

    return dataframe


def guess_time_column(columns):
    candidates = [
        "time",
        "TIME",
        "btjd",
        "BTJD",
        "bjd",
        "BJD",
        "timestamp",
        "cadence_time",
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return "(Use row number)"


def guess_flux_column(columns):
    candidates = [
        "PDCSAP_FLUX",
        "SAP_FLUX",
        "flux",
        "FLUX",
        "relative_flux",
        "normalized_flux",
        "y",
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return columns[0]


# ============================================================
# SIDEBAR AND HEADER
# ============================================================

with st.sidebar:
    st.title("🔭 STAR-TRACE")
    st.caption("Stellar Signal Intelligence")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Light-Curve Analysis",
            "Model Explorer",
            "Research Data",
            "System Diagnostics",
        ],
    )

    st.divider()

    st.markdown(
        "[GitHub repository]"
        "(https://github.com/amnafatimaa6-ops/STAR-TRACE)"
    )

st.title("STAR-TRACE")
st.caption(
    "AI-assisted stellar light-curve analysis"
)

st.divider()


# ============================================================
# LOAD MODELS
# ============================================================

try:
    with st.spinner(
        "Loading STAR-TRACE models and research data..."
    ):
        models, tables, archive_names = load_assets(
            REPOSITORY_URL
        )

except Exception as exc:
    st.error("Could not load STAR-TRACE assets.")
    st.exception(exc)

    st.info(
        "Check the deployment ZIP, model filenames, "
        "and TensorFlow compatibility."
    )

    st.stop()


denoising_model = models[
    "startrace_denoising_v2.keras"
]


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Mission overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Trained models", len(models))
    col2.metric("Research tables", len(tables))
    col3.metric("Archive entries", len(archive_names))

    st.success("All available research assets loaded.")

    st.markdown("### What STAR-TRACE can do")

    st.markdown(
        """
        - Upload and inspect a CSV light curve.
        - Compare raw flux with a neural-network reconstruction.
        - Inspect residuals and numerical summary statistics.
        - Explore trained model architectures.
        - Browse evaluation and morphology tables.
        """
    )

    st.warning(
        "Denoising is not event detection. A reconstruction "
        "may suppress real astrophysical signals, so always "
        "inspect the raw data and validate the model."
    )

    st.info(
        "Start with Light-Curve Analysis in the sidebar."
    )


# ============================================================
# LIGHT-CURVE ANALYSIS
# ============================================================

elif page == "Light-Curve Analysis":

    st.subheader("Interactive light-curve laboratory")

    st.write(
        "Upload a CSV containing numerical flux measurements. "
        "A time column is optional."
    )

    uploaded_file = st.file_uploader(
        "Upload light-curve CSV",
        type=["csv"],
        help=(
            "Expected: one row per observation and at least "
            "one numerical flux column."
        ),
    )

    if uploaded_file is None:
        st.markdown(
            """
            **Suggested workflow**

            1. Export a light curve to CSV.
            2. Upload it here.
            3. Select the time and flux columns.
            4. Check the raw signal.
            5. Run the denoising model and inspect the residuals.
            """
        )

    else:
        try:
            dataframe = read_uploaded_csv(uploaded_file)

        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        numeric_columns = [
            column for column in dataframe.columns
            if pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if not numeric_columns:
            st.error(
                "No numerical columns were found in the CSV."
            )
            st.stop()

        st.markdown("#### Select data columns")

        col1, col2 = st.columns(2)

        time_options = [
            "(Use row number)"
        ] + list(dataframe.columns)

        default_time = guess_time_column(
            dataframe.columns
        )

        with col1:
            time_column = st.selectbox(
                "Time column",
                time_options,
                index=time_options.index(default_time),
            )

        default_flux = guess_flux_column(
            numeric_columns
        )

        with col2:
            flux_column = st.selectbox(
                "Flux column",
                numeric_columns,
                index=numeric_columns.index(default_flux),
            )

        st.caption(
            "Check that you selected the intended flux series. "
            "Some files contain several flux measurements."
        )

        if time_column == "(Use row number)":
            time_values = np.arange(len(dataframe))
        else:
            time_values = dataframe[time_column].to_numpy()

        raw_flux_series = pd.to_numeric(
            dataframe[flux_column],
            errors="coerce",
        )

        valid_mask = (
            raw_flux_series.notna().to_numpy()
            & np.isfinite(
                raw_flux_series.fillna(0).to_numpy(
                    dtype=np.float64
                )
            )
        )

        if time_column != "(Use row number)":
            numeric_time = pd.to_numeric(
                pd.Series(time_values),
                errors="coerce",
            )

            valid_mask &= numeric_time.notna().to_numpy()

            time_values = numeric_time.to_numpy()

        clean_flux = raw_flux_series.to_numpy(
            dtype=np.float64
        )[valid_mask]

        clean_time = np.asarray(time_values)[valid_mask]

        if len(clean_flux) < 2:
            st.error(
                "At least two valid flux observations are required."
            )
            st.stop()

        if len(clean_flux) > 2_000_000:
            st.error(
                "This upload exceeds two million observations. "
                "Use a smaller segment for this deployment."
            )
            st.stop()

        if not np.all(np.isfinite(clean_flux)):
            st.error(
                "Flux contains infinite values. Clean the data first."
            )
            st.stop()

        if len(clean_flux) != len(dataframe):
            st.warning(
                f"Excluded {len(dataframe) - len(clean_flux):,} "
                "rows with missing or invalid selected values."
            )

        st.markdown("#### Raw signal")

        st.metric(
            "Valid observations",
            f"{len(clean_flux):,}",
        )

        raw_plot = pd.DataFrame(
            {
                "Observation": np.arange(len(clean_flux)),
                "Raw flux": clean_flux,
            }
        )

        if len(clean_flux) > 100_000:
            display_stride = int(
                np.ceil(len(clean_flux) / 100_000)
            )
            raw_plot = raw_plot.iloc[::display_stride]

        st.line_chart(
            raw_plot,
            x="Observation",
            y="Raw flux",
            height=300,
        )

        st.markdown("#### Inference settings")

        norm_method = st.selectbox(
            "Input normalisation",
            [
                "Standard score (z-score)",
                "Min-max [0, 1]",
                "Raw flux",
            ],
            index=0,
            help=(
                "Use the preprocessing applied during training "
                "if it is known. The default is only a starting "
                "point, not a claim about training preprocessing."
            ),
        )

        stride = st.slider(
            "Window stride",
            min_value=32,
            max_value=256,
            value=128,
            step=32,
            help=(
                "The model processes 256 observations per window. "
                "Smaller strides increase overlap and computation."
            ),
        )

        st.caption(
            f"Window size: {WINDOW_SIZE} observations. "
            "Overlapping predictions are averaged."
        )

        run_analysis = st.button(
            "Run denoising analysis",
            type="primary",
            use_container_width=True,
        )

        if run_analysis:
            try:
                with st.spinner(
                    "Preprocessing flux and running inference..."
                ):
                    prepared_flux, restore_scale = prepare_flux(
                        clean_flux,
                        norm_method,
                    )

                    reconstructed_normalised = (
                        denoise_light_curve(
                            denoising_model,
                            prepared_flux,
                            window_size=WINDOW_SIZE,
                            stride=stride,
                        )
                    )

                    reconstructed_flux = restore_scale(
                        reconstructed_normalised
                    )

                    reconstructed_flux = np.asarray(
                        reconstructed_flux,
                        dtype=np.float64,
                    )

                    residuals = (
                        clean_flux - reconstructed_flux
                    )

                    if not np.all(
                        np.isfinite(reconstructed_flux)
                    ):
                        raise ValueError(
                            "The model returned non-finite values. "
                            "Check the input normalisation and "
                            "model compatibility."
                        )

                    rmse = float(
                        np.sqrt(np.mean(residuals ** 2))
                    )

                    mae = float(
                        np.mean(np.abs(residuals))
                    )

                    flux_std = float(np.std(clean_flux))

                    relative_rmse = (
                        rmse / flux_std
                        if flux_std > 1e-12
                        else np.nan
                    )

                    results = pd.DataFrame(
                        {
                            "time": clean_time,
                            "raw_flux": clean_flux,
                            "denoised_flux": reconstructed_flux,
                            "residual": residuals,
                        }
                    )

                st.success("Inference completed.")

                st.markdown("#### Raw versus denoised")

                st.line_chart(
                    results,
                    x="time",
                    y=["raw_flux", "denoised_flux"],
                    height=400,
                )

                st.caption(
                    "The horizontal axis uses the selected time "
                    "values, or observation indices if no time "
                    "column was selected."
                )

                st.markdown("#### Residual signal")

                st.line_chart(
                    results,
                    x="time",
                    y="residual",
                    height=250,
                )

                st.caption(
                    "Residual = raw flux − reconstructed flux. "
                    "Residuals may contain noise, real events, "
                    "or model reconstruction error."
                )

                st.markdown("#### Numerical summary")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Residual RMSE",
                    f"{rmse:.6g}",
                )

                c2.metric(
                    "Residual MAE",
                    f"{mae:.6g}",
                )

                c3.metric(
                    "RMSE / raw standard deviation",
                    (
                        f"{relative_rmse:.4f}"
                        if np.isfinite(relative_rmse)
                        else "Undefined"
                    ),
                )

                st.dataframe(
                    results.head(20),
                    use_container_width=True,
                )

                st.download_button(
                    "Download analysis CSV",
                    data=results.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="startrace_denoising_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.warning(
                    "These are reconstruction diagnostics, not "
                    "proof of improved signal quality. Without "
                    "a trusted clean reference signal, residual "
                    "RMSE and MAE do not measure denoising accuracy. "
                    "Inspect the raw curve for transits, flares, "
                    "gaps, and other potentially real events."
                )

            except Exception as exc:
                st.error("Denoising analysis failed.")
                st.exception(exc)


# ============================================================
# MODEL EXPLORER
# ============================================================

elif page == "Model Explorer":

    st.subheader("Trained model explorer")

    selected_name = st.selectbox(
        "Select model",
        list(models.keys()),
    )

    model = models[selected_name]

    col1, col2 = st.columns(2)

    col1.metric(
        "Total parameters",
        f"{model.count_params():,}",
    )

    col2.write("**Input shape**")
    col2.code(str(model.input_shape))

    st.write("**Output shape**")
    st.code(str(model.output_shape))

    if st.checkbox("Show architecture"):
        lines = []
        model.summary(print_fn=lines.append)

        st.code(
            "\n".join(lines),
            language="text",
        )

    st.caption(
        "Model architecture alone does not establish scientific validity."
    )


# ============================================================
# RESEARCH DATA
# ============================================================

elif page == "Research Data":

    st.subheader("Research data explorer")

    if not tables:
        st.warning("No CSV tables are available.")

    else:
        table_name = st.selectbox(
            "Select research table",
            list(tables.keys()),
        )

        table = tables[table_name]

        col1, col2 = st.columns(2)

        col1.metric("Rows", f"{len(table):,}")
        col2.metric("Columns", len(table.columns))

        st.dataframe(
            table,
            use_container_width=True,
            height=450,
        )

        st.download_button(
            "Download CSV",
            data=table.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=table_name,
            mime="text/csv",
        )

        if st.checkbox("Show descriptive statistics"):
            st.dataframe(
                table.describe(
                    include="all"
                ).T,
                use_container_width=True,
            )


# ============================================================
# SYSTEM DIAGNOSTICS
# ============================================================

elif page == "System Diagnostics":

    st.subheader("System diagnostics")

    st.write("TensorFlow:", tf.__version__)
    st.write("Pandas:", pd.__version__)
    st.write("Streamlit:", st.__version__)

    st.markdown("#### TensorFlow devices")

    for device in tf.config.list_physical_devices():
        st.write(
            f"{device.device_type}: {device.name}"
        )

    st.markdown("#### Loaded models")

    for name, model in models.items():
        st.write(
            f"**{name}** — input {model.input_shape}; "
            f"output {model.output_shape}; "
            f"{model.count_params():,} parameters"
        )

    st.markdown("#### Available research tables")

    for name, table in tables.items():
        st.write(
            f"**{name}** — {len(table):,} rows, "
            f"{len(table.columns)} columns"
        )

    with st.expander("View archive contents"):
        st.code(
            "\n".join(archive_names),
            language="text",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "STAR-TRACE | Experimental scientific AI | "
    "Predictions require independent scientific validation."
)
