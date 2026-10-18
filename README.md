# 🔭 STARTRACE
## Learning Stellar Signal Morphology from TESS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amnafatimaa6-ops/STAR-TRACE/blob/main/notebook_star_trace.ipynb)

**STARTRACE is an exploratory research project where I use real NASA TESS observations to explore the intersection of astrophysics and machine learning.**

I wanted to find out whether convolutional autoencoders could learn useful patterns from stellar light curves, reconstruct noisy signals, and help group different types of stellar behaviour.

Rather than relying entirely on simulated data, I built a pipeline around real observations from NASA's Transiting Exoplanet Survey Satellite (TESS). The project covers data acquisition, preprocessing, neural network training, denoising, and unsupervised clustering.

**Research question:** Can a convolutional denoising model learn useful representations of existing TESS light-curve windows, and do the resulting groups capture measurable signal properties?

This is an exploratory research pilot. I have not established independently verified labels or confirmed any new astronomical discoveries, and I have kept those limitations explicit throughout the project.



## 🎬 Demo

![STAR-TRACE Demo](demo.gif)

## 📸 Screenshots

### Screenshot 1
![STAR-TRACE Screenshot 1](Screenshot1.png)

### Screenshot 2
![STAR-TRACE Screenshot 2](Screenshot2.png)

### Screenshot 3
![STAR-TRACE Screenshot 3](Screenshot3.png)

---

## The stars I worked with

I selected four targets representing different astrophysical contexts so I could explore how the pipeline handled different kinds of stellar signals.

| Target | Astrophysical context | TESS observations |
|---|---|---|
| HD 209458 | Known transiting-exoplanet host | Sector 56, SPOC, 20-second cadence |
| TOI-700 | Known planetary system | Sector 1, SPOC, 2-minute cadence |
| Algol | Eclipsing binary | Sector 18, SPOC, 2-minute cadence |
| RR Lyrae | Pulsating variable star | Sector 14, SPOC, 2-minute cadence |

## 🛠️ How I built STARTRACE

I divided the notebook into three parts, moving from collecting the observations to training models and investigating what they learned.

### Part 1 — Data acquisition and catalogue cross-matching

I started by setting up the project structure, fixing the random seed at 42, and organising the raw data, processed data, results, and figures. I used `lightkurve` to download one TESS SPOC light curve for each target and saved the original observations as FITS files.

I then queried the NASA Exoplanet Archive to check the targets against confirmed transiting-planet hosts and available transit ephemerides.

- Retrieved catalogue records for 3,687 confirmed transiting-planet host stars.
- Collected 3,393 planet records with usable transit ephemerides.
- Compared predicted transit times with the actual observation ranges.
- Investigated why HD 209458 did not match through the initial catalogue query and retrieved 72 predicted transit rows from NASA's transit service.

This stage helped me establish the data pipeline and investigate catalogue issues before moving into model development.

### Part 2 — Cleaning, windowing, and the first autoencoder

Once the observations were collected, I cleaned the light curves, removed flagged observations and robustly identified outliers. I then divided the data into fixed-length windows of 256 samples, using a stride of 128 and skipping windows that crossed large observational gaps.

The result was a dataset of **1,158 windows**.

I trained my first convolutional autoencoder, v1, on HD 209458, TOI-700, and Algol while holding RR Lyrae out from training.

I also built an interactive light-curve inspector to examine individual reconstructions and their reconstruction errors. Alongside this, I audited Algol's low-flux candidates, investigated the unusual flux scale in RR Lyrae, and checked the saved dataset for quality issues.

The first model gave me a baseline for understanding how reconstruction performance varied across the targets.

### Part 3 — Denoising, evaluation, and clustering

For the second model, v2, I developed a masked-denoising 1D convolutional autoencoder using dilated Conv1D layers.

During training, I added Gaussian noise and masked segments of the input signals so the model had to reconstruct the original windows from incomplete or corrupted inputs.

I evaluated v2 against v1 using both temporally held-out validation windows and the held-out RR Lyrae data. I then extracted 32-dimensional representations from the trained model and applied K-means clustering with four clusters.

To see how those representations compared with a more traditional approach, I also calculated nine handcrafted signal features and clustered those separately.

Finally, I checked the stability of the learned clusters across five random seeds and saved the evaluation results, plots, and final research report.

---

## Results so far

The experiments produced some encouraging reconstruction results, but they also raised questions about how useful the learned representations are for astrophysical analysis.

### Dataset and first model

| Experiment | Result |
|---|---|
| Total dataset | 1,158 windows × 256 samples |
| HD 209458 | 843 windows |
| RR Lyrae | 127 windows |
| TOI-700 | 113 windows |
| Algol | 75 windows |

The first autoencoder provided a baseline against which I could compare the denoising model.

| Target | v1 reconstruction MAE | Improvement over zero baseline |
|---|---:|---:|
| Algol | 0.140 | 81.2% |
| HD 209458 | 0.575 | 28.6% |
| RR Lyrae (held out) | 0.105 | 87.0% |
| TOI-700 | 0.668 | 15.8% |

The results varied considerably across targets, which reinforced the importance of evaluating each star separately rather than relying only on an overall score.

### Comparing v1 and v2

| Evaluation | Model | MAE | Correlation |
|---|---|---:|---:|
| Temporal validation, masked + noisy | v1 | 0.576 | 0.431 |
| Temporal validation, masked + noisy | v2 | 0.149 | 0.916 |
| Held-out RR Lyrae, masked + noisy | v1 | 0.201 | 0.912 |
| Held-out RR Lyrae, masked + noisy | v2 | 0.109 | 0.960 |

Under the masked-and-noisy evaluation conditions, v2 achieved lower reconstruction errors and higher correlations than v1 on both evaluation splits.

These results suggest that the denoising approach was more effective at reconstructing the evaluated signals. However, reconstruction quality alone does not demonstrate that the model has learned scientifically meaningful features or will generalise to a larger population of stars.

### What happened with clustering?

I compared K-means clustering on the learned embeddings with clustering on nine handcrafted signal features.

| Measurement | Learned embeddings | Handcrafted features |
|---|---:|---:|
| Silhouette score | 0.311 | 0.416 |

Agreement between the two clusterings was very low:

- **ARI:** 0.001
- **NMI:** 0.021

The learned clusters were identical across five random initialisations, with an ARI of 1.0 between those runs.

This was an interesting result because the handcrafted features achieved better separation according to the silhouette score, while the two approaches organised the windows into substantially different groups.

The stable clustering is encouraging, but it does not prove that the clusters represent genuine astrophysical categories. That still needs to be tested against verified labels and independent observations.

**These are reconstruction and clustering results, not evidence of new astronomical discoveries.**

---

## Project structure

The notebook writes its outputs under `/content/STARTRACE` in Google Colab.

```text
STARTRACE/
├── data/
│   ├── raw/
│   │   ├── TESS FITS files
│   │   ├── catalogue and ephemeris CSVs
│   │   └── predicted transit data
│   └── processed/
│       ├── cleaned light curves
│       ├── X_windows.npy
│       └── window_metadata.csv
├── results/
│   ├── audit and evaluation CSVs
│   ├── figures
│   └── research reports
├── startrace_autoencoder.keras
└── startrace_denoising_v2.keras
```

##  Run the notebook

You can open the notebook directly in Google Colab using the badge at the top of this README.

1. Open `notebook_star_trace.ipynb`.
2. Run the cells in order.
3. Allow internet access for downloading the TESS observations and querying the NASA Exoplanet Archive.
4. Review the generated datasets, model evaluations, plots, and research reports.

**Libraries used:** `lightkurve`, `tensorflow` / `keras`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `astropy`, `ipywidgets`, and `requests`.

##  Limitations

I wanted to document the limitations alongside the results rather than make the project appear more conclusive than it is.

1. **Small sample:** The pilot includes only four stars, which is insufficient to establish general performance across stellar populations.
2. **Overlapping windows:** Adjacent windows share observations and cannot be treated as independent samples.
3. **RR Lyrae processing:** The corrected PDCSAP flux was retrieved but was not propagated into the saved window array used in these experiments.
4. **Period estimation:** Autocorrelation at a few fixed lags is exploratory and does not constitute a validated period measurement.
5. **Unverified labels:** No independently verified event labels or confirmed discoveries were established.
6. **Clustering validation:** Stability across random initialisations does not establish reproducibility across independent datasets or astrophysical validity.

Reconstruction error is not an event probability, and cluster separation is not proof of astrophysical meaning.

##  What I want to explore next

STARTRACE is an ongoing research project, and there are several things I would like to improve before drawing stronger conclusions.

- Verify target metadata and event labels against independent astronomical catalogues.
- Define reproducible event windows and appropriate negative examples.
- Expand the dataset and use target-level training, validation, and test splits.
- Compare simple baselines with a 1D CNN if the expanded dataset supports a meaningful comparison.
- Evaluate performance using precision, recall, F1-score, PR-AUC, and genuinely held-out targets.

My priority is to establish whether the learned patterns remain useful on stars the model has never encountered, rather than simply improving reconstruction metrics.

## Data sources

- **TESS observations:** NASA's Mikulski Archive for Space Telescopes (MAST), accessed using `lightkurve`.
- **Planet catalogue and transit ephemerides:** NASA Exoplanet Archive.
- **Additional transit information:** NASA's transit service and the ephemeris sources investigated during the HD 209458 audit.

---

## Why I built this project

I am interested in combining astrophysics with artificial intelligence, particularly in how computational methods can help us study signals from space.

STARTRACE gave me a way to explore that interest through a practical research problem: working with real astronomical observations, building a time-series pipeline, training neural networks, and investigating what their outputs actually tell us.

One thing this project has taught me is that getting a model to work is only part of the process. Understanding the data, questioning the results, comparing against simpler methods, and recognising the limitations are just as important.

I have not solved the problem of stellar signal classification, and I am not presenting STARTRACE as a discovery system. What I have built is an exploratory foundation that I can improve through better data, stronger validation, and further experiments.

**My goal is to learn from real astronomical observations, test on genuinely unseen stars, and make claims only when the evidence supports them.**


