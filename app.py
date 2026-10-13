
import io
import zipfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import tensorflow as tf
import streamlit as st


@st.cache_resource
def load_assets(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "STARTRACE-Streamlit"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()

        def find_file(filename):
            matches = [
                name for name in names
                if name.endswith("/" + filename)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"{filename} not found inside GitHub ZIP"
                )
            return matches[0]

        # Keep extracted model files available while Streamlit runs.
        temp_dir = tempfile.mkdtemp(prefix="startrace_")
        models = {}

        for filename in (
            "startrace_denoising_v2.keras",
            "startrace_autoencoder.keras",
        ):
            member = find_file(filename)
            destination = Path(temp_dir) / filename
            destination.write_bytes(archive.read(member))

            models[filename] = tf.keras.models.load_model(
                str(destination),
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
                member = find_file(filename)
                tables[filename] = pd.read_csv(
                    io.BytesIO(archive.read(member))
                )
            except FileNotFoundError:
                pass

    return models, tables, names
