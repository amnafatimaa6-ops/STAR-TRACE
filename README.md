# STAR-TRACE
# ⭐ STARTRACE — Learning Stellar Signal Morphology from TESS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amnafatimaa6-ops/STAR-TRACE/blob/main/Untitled8.ipynb)

An exploratory research pilot that uses real NASA TESS light curves to study whether convolutional autoencoders can learn useful representations of stellar signal shape.

**Research question:** Can a convolutional denoising model learn useful representations of existing TESS light-curve windows, and do the resulting groups capture measurable signal properties?

> The results are exploratory. No independently verified labels or confirmed discoveries were established.

## Pilot targets

| Target | Context | TESS product |
|---|---|---|
| HD 209458 | Known transiting-planet host | Sector 56, SPOC, 20 s |
| TOI-700 | Known planetary system | Sector 1, SPOC, 2 min |
| Algol | Eclipsing binary | Sector 18, SPOC, 2 min |
| RR Lyrae | Pulsating variable star | Sector 14, SPOC, 2 min |

## What the notebook does

**Part 1 — Data acquisition and catalogue cross-matching**
- Sets up the project (seed 42, folders, `config.json`) and downloads one TESS SPOC light curve per target with `lightkurve`.
- Retrieves NASA Exoplanet Archive catalogues: 3,687 transiting-planet hosts and 3,393 planet records with transit ephemerides.
- Cross-matches them with the observations. TOI-700 matches four planets, and Algol and RR Lyrae match none.
- HD 209458's catalogue entry lacks a transit mid-time, so 72 predicted transits are retrieved from NASA's transit service.

**Part 2 — Cleaning, windowing, first model, audits**
- Cleans and normalizes the curves, then builds **1,158 windows × 256 samples** (stride 128, robust per-target scaling).
- Trains a v1 autoencoder on HD 209458, TOI-700 and Algol, holding RR Lyrae out. An interactive inspector shows each window's reconstruction error.
- Audits Algol's low-flux dips (33 exploratory candidates), RR Lyrae's abnormal flux scale, and the dataset, and evaluates v1 against a zero baseline with noise and missing-data tests.

**Part 3 — Denoising model v2, evaluation, clustering**
- Trains a masked-denoising 1D convolutional autoencoder (v2) with a time-based train/validation split and a gap.
- Compares v2 with v1 on clean and masked-noisy input, including held-out RR Lyrae.
- Clusters 32-dimensional learned embeddings (K-means, k = 4), compares them with handcrafted-feature clusters, checks seed stability, and saves a final report.

## Results

| Experiment | Result |
|---|---|
| Dataset | 1,158 windows (HD 209458 843, RR Lyrae 127, TOI-700 113, Algol 75) |
| v1 reconstruction MAE | Algol 0.140 · HD 209458 0.575 · RR Lyrae (held out) 0.105 · TOI-700 0.668 |
| v1 vs zero baseline (MAE improvement) | Algol 81.2% · HD 209458 28.6% · RR Lyrae 87.0% · TOI-700 15.8% |
| v2, masked + noisy, temporal validation | MAE 0.149, correlation 0.916 (v1: 0.576, 0.431) |
| v2, masked + noisy, held-out RR Lyrae | MAE 0.109, correlation 0.960 (v1: 0.201, 0.912) |
| Clustering silhouette | 0.311 learned vs 0.416 handcrafted (ARI 0.001, NMI 0.021) |

These are reconstruction and clustering metrics, not evidence of new astronomical discoveries.

## Run it

Open `Untitled8.ipynb` in Google Colab and run the cells in order. Internet access is needed for the TESS archive and the NASA Exoplanet Archive.

Libraries used: `lightkurve`, `tensorflow`/`keras`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `astropy`, `ipywidgets`, `requests`.

Files are written under `/content/STARTRACE`:

```
STARTRACE/
├── data/raw/         TESS FITS files, catalogue and ephemeris CSVs, predicted transits
├── data/processed/   cleaned light curves, X_windows.npy, window_metadata.csv
├── results/          audit and evaluation CSVs, figures, research reports
├── startrace_autoencoder.keras
└── startrace_denoising_v2.keras
```

## Limitations

1. Only four stars are represented.
2. Adjacent windows overlap, so windows are not independent samples.
3. RR Lyrae's corrected PDCSAP flux has not been propagated to the saved window array used in these experiments.
4. Autocorrelation at a few fixed lags is not a validated period measurement.
5. No independently verified labels or confirmed discoveries were established.
6. Cluster stability across random initializations does not establish reproducibility across datasets.

## Next steps

1. Verify target metadata and event labels against independent catalogues.
2. Define reproducible event windows and negative examples.
3. Build a labelled dataset with target-level train/validation/test separation.
4. Compare simple baselines with a 1D CNN if the dataset supports it.
5. Report precision, recall, F1, PR-AUC and held-out-target performance.

## Data sources

TESS light curves via `lightkurve` (NASA MAST); planet catalogues and transit ephemerides from the NASA Exoplanet Archive.
