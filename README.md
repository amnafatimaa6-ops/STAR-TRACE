# 🔭 STARTRACE

### Learning Stellar Signal Morphology from TESS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amnafatimaa6-ops/STAR-TRACE/blob/main/notebook_star_trace.ipynb)

---

## 🌟 What is this project? (in simple words)

Stars are not perfectly steady. Their brightness changes a tiny bit over time. A planet passing in front of a star blocks a little of its light. Two stars circling each other take turns hiding one another. Some stars swell and shrink on their own. A graph of a star's brightness over time is called a **light curve**, a bit like a heartbeat graph for a star.

NASA's TESS space telescope records light curves, and the data is public. Real data is noisy and has gaps, so it is hard to work with.

In this project I asked a simple question: **can a small neural network (a type of AI) learn what real star signals look like, well enough to rebuild a signal when part of it is noisy or missing? And can what it learned help sort signals into groups?**

What I found, in short:

- **Rebuilding signals worked.** My second model rebuilt noisy and partly hidden signals much better than my first model.
- **Grouping is not proven.** The groups made from the AI's features were not cleaner than groups made from simple hand-calculated measurements, and I cannot say they match real types of stars.
- **No discoveries.** I am not claiming to have found any planet, eclipse or other event. This is a learning project built on real data, and I have tried to be clear about what it does and does not show.

I also built a small web dashboard, so anyone can try the trained model on a demo signal, a CSV file or a TESS target (see the dashboard section below).

### At a glance

| | |
|---|---|
| **Data** | Real NASA TESS light curves for 4 stars |
| **Dataset** | 1,158 windows, each with 256 brightness measurements |
| **Models** | Model v1 (an autoencoder, 1,749 parameters) and model v2 (a denoising network, 9,649 parameters) |
| **Dashboard** | Streamlit app with 5 pages |
| **Status** | Exploratory research project. No independently verified labels and no confirmed discoveries |

---

## 🖥️ The STAR-TRACE dashboard

Besides the notebook, I turned the two trained models into a web app (`app.py`, built with Streamlit). When it starts, it downloads `STARTRACE_deployment_assets.zip` from this repository, which holds the two trained models and my result tables.

### Demo

![STAR-TRACE dashboard demo](VIsuals/demo.gif)

The GIF shows the **Signal Laboratory** page. It runs the built-in synthetic demo signal through the model, switches the input normalisation from Z-score to Min-max and compares the error values, and then scrolls down to the plot, the residual panel and the CSV download button.

### Screenshots

**1. Mission Overview** shows the sidebar menu and mission status, four summary tiles (2 Keras models, 24 research tables, window size 256, TESS retrieval available) and buttons that open the Signal Laboratory and the Research Archive. It also has a note that the synthetic demo is not telescope data.

![Mission Overview](VIsuals/Screenshot1.png)

**2. Model Intelligence: denoising model v2** shows its input and output shape `(None, 256, 1)`, 6 layers, 9,649 parameters and the layer-by-layer summary.

![Model v2 details](VIsuals/Screenshot2.png)

**3. Model Intelligence: first autoencoder (v1)** shows the same details for the first model: 7 layers and 1,749 parameters.

![Model v1 details](VIsuals/Screenshot3.png)

### What the dashboard has

**Sidebar:** a navigation menu with the five pages below, plus a "Mission status" box showing whether the asset bundle loaded, how many Keras models and research tables were found, and whether live TESS download is available.

**1. Mission Overview**
- A short mission brief and the four summary tiles.
- Buttons to jump to the Signal Laboratory or the Research Archive.
- A reminder that the synthetic demo is not real data and model outputs are not confirmed discoveries.

**2. Signal Laboratory** (the main page). You choose one of three data sources:
- **Synthetic demo:** a made-up signal (1,800 points with slow waves, noise and three transit-like dips). It is for demonstration only and is not a real observation.
- **Upload CSV:** your own light curve. The app looks for common column names (for example `time` / `BTJD` and `flux` / `PDCSAP_FLUX` / `SAP_FLUX`) and needs at least 256 valid rows.
- **Fetch real TESS observation:** type a TIC ID or target name. The app searches TESS with `lightkurve`, takes the first product that downloads (out of the first five results) and uses PDCSAP flux, or SAP flux if PDCSAP is not available. It may not be the same sector or cadence I used in the notebook.

Then you can:
- Preview the first 25 rows of the data.
- See which model is loaded. The app automatically uses the denoising model (v2).
- Pick an input normalisation (**Z-score, Robust, Min-max or Raw**) and the maximum number of observations to process (512 to 6,000).
- Click **Run AI reconstruction**. The app cuts the signal into 256-sample windows with a stride of 128, runs the model on each window and averages the overlapping parts.
- Read four result tiles (observations, reconstruction RMSE, mean absolute error, normalisation used).
- View a plot with the **observed flux** (blue) and the **AI reconstruction** (yellow), and a second panel with the **residual** (observed minus reconstructed, green).
- Download the results as a CSV (time, observed flux, reconstructed flux, residual).
- Read a short "How to read this plot" guide and a warning that this is reconstruction, not forecasting.

**3. Model Intelligence:** for each loaded model, the input and output shapes, number of layers, number of parameters and the layer summary. It also explains the sliding-window method (256 samples, stride 128, overlaps averaged).

**4. Research Archive:** all 24 result tables from my notebook (for example the model comparison, morphology features, cluster results and the cleaned light curves). Each one opens in its own panel with its row count, a scrollable table and a download button.

**5. System Diagnostics:** whether the asset bundle loaded, the number of models and tables, the Python, NumPy and Pandas versions, whether live TESS support is available, and any loading warnings.

### Things to know when using it

- It **reconstructs** the signal you give it. It does not predict the future, and a residual or a dip does not mean a planet or any other event. The app says this too.
- **Normalisation matters.** None of the four options exactly matches how I scaled the windows in the notebook (per-star median and MAD scaling, clipped to ±8). In the GIF, the same demo signal gives an RMSE of 0.00079 with Z-score and 0.0025 with Min-max.
- The error values in the dashboard are in the flux units of the data you load. They cannot be compared with the MAE numbers in the notebook, which are in scaled units.
- The models were trained on 256-sample windows from only four stars, so results on other stars are not validated.

### Run the dashboard yourself

```
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` lists `streamlit==1.65.0`, `tensorflow==2.21.0`, `pandas`, `numpy`, `matplotlib` and `lightkurve`. The app needs internet access to download the asset ZIP.

---

## ⭐ The stars I worked with

I picked four stars with different behaviour so I could see how the pipeline handled different kinds of signals.

| Target | What it is | TESS data used |
| --- | --- | --- |
| HD 209458 | Known transiting-exoplanet host | Sector 56, SPOC, 20-second cadence |
| TOI-700 | Known planetary system | Sector 1, SPOC, 2-minute cadence |
| Algol | Eclipsing binary | Sector 18, SPOC, 2-minute cadence |
| RR Lyrae | Pulsating variable star | Sector 14, SPOC, 2-minute cadence |

---

## 🛠️ How I built it

The notebook has three parts.

### Part 1: Getting the data and checking it against catalogues

- Set up the folders, fixed the random seed at 42, and used `lightkurve` to download one TESS SPOC light curve for each star.
- Queried the NASA Exoplanet Archive: **3,687** confirmed transiting-planet host stars, and **3,393** planet records with usable transit timing (from 2,634 host stars). I converted catalogue times from BJD to BTJD so they match TESS times.
- **TOI-700:** four planets have predicted transits inside my observation window (b: 3, c: 1, e: 1, d: 1).
- **Algol and RR Lyrae:** no confirmed transiting planet in the catalogue, which fits the kind of stars they are.
- **HD 209458:** my first catalogue query gave its orbital period (3.524749 days) but no transit mid-time or duration. I retrieved **72** predicted-transit rows from NASA's transit service, which come from several published timing solutions (Kokori et al. 2022 and 2023, Rosenthal et al. 2021 and ExoFOP).
- I only checked that predicted transit times fall inside the observed time range. I have **not** yet checked the light curves at those times.

### Part 2: Cleaning, windows, the first model and data checks

- **Cleaning:** kept points with finite flux and quality flag 0, removed points more than 5 robust standard deviations (median and MAD) from the median, and divided by the median. Points kept: HD 209458 110,897; TOI-700 18,279; Algol 13,247; RR Lyrae 18,292.
- **Windows:** cut each light curve into windows of 256 samples with a stride of 128 (neighbouring windows overlap by half), skipped windows that cross large gaps (more than 5× the cadence), scaled each star with its own median and MAD, and clipped to ±8. This gave **1,158 windows**.
- **Model v1:** a convolutional autoencoder. Three layers squeeze each window down to 32 steps × 4 channels and three layers expand it back. I trained it for 12 epochs (final training MAE 0.582, validation MAE 0.414). RR Lyrae was kept out of training on purpose. *See the note in the limitations about how the validation split worked.*
- **Light-curve inspector:** a small widget to pick a star and window, compare the original with the reconstruction, and see the reconstruction error and where that window ranks among the same star's windows. Its text box answers a few simple questions using fixed rules (it is not a language model).
- **Algol audit:** I marked 33 low-flux runs (below the 10th-percentile flux). They are exploratory, not confirmed eclipses, and some are just pieces of one broad dip.
- **RR Lyrae audit:** after scaling, its flux ranged from −2.83 to 7.03 and 5,441 values were negative. In the original file, about 30% of the flux values were negative. I looked at the file header and columns and re-downloaded the official SPOC Sector 14 product, but I have **not** found the cause yet (see limitations).
- **Dataset check:** all window values are finite and the metadata has no missing cells. The catalogue label for every star is still marked unverified.

### Part 3: Denoising model v2, evaluation and clustering

- **Split by time:** for HD 209458, TOI-700 and Algol, the first 75% of windows (by time) went to training and the last 20% to validation, with a gap in between (1.25, 1.28 and 1.57 days). That gave 772 training windows and 207 validation windows. All 127 RR Lyrae windows stayed held out.
- **Model v2:** five 1D convolution layers (some dilated) that keep all 256 samples, 9,649 parameters. During training, each window was damaged with Gaussian noise (standard deviation 0.08) and one hidden section of 16 to 48 samples, and the model had to rebuild the clean window. Training ran for 30 epochs (early stopping was on) and the final validation MAE was 0.146.
- **Learned features:** 32 numbers per window, taken from the last hidden layer of v2 and averaged over time. I grouped them with K-means (4 groups).
- **Comparison with simple features:** I also calculated 9 hand-made features per window (amplitude, peak-to-peak range, roughness, standard deviation of first differences, skewness, kurtosis, and autocorrelation at lags 10, 25 and 50) and grouped those the same way.
- **Stability check:** I repeated the clustering with five random seeds.

---

## 📊 Results

### Dataset

| Star | Windows |
| --- | --- |
| HD 209458 | 843 |
| RR Lyrae | 127 |
| TOI-700 | 113 |
| Algol | 75 |
| **Total** | **1,158 windows × 256 samples** |

### Model v1, per star

"Improvement over zero baseline" compares the model with simply predicting a flat line at zero.

| Star | v1 MAE | Improvement over zero baseline | Did v1 train on it? |
| --- | --- | --- | --- |
| Algol | 0.140 | 81.2% | No (set aside as validation) |
| HD 209458 | 0.575 | 28.6% | Yes (all 843 windows) |
| RR Lyrae | 0.105 | 87.0% | No (held out on purpose; see the RR Lyrae caveat below) |
| TOI-700 | 0.668 | 15.8% | Partly (33 of 113 windows) |

Results differ a lot between stars, so I look at each star separately instead of trusting one overall score.

### Model v1 vs model v2 (damaged inputs)

Both models were asked to rebuild the **clean** window from an input with added noise and a hidden section.

| Evaluation | Model | MAE | Correlation |
| --- | --- | --- | --- |
| Temporal validation (207 windows) | v1 | 0.576 | 0.431 |
| Temporal validation (207 windows) | v2 | 0.149 | 0.916 |
| Held-out RR Lyrae (127 windows) | v1 | 0.201 | 0.912 |
| Held-out RR Lyrae (127 windows) | v2 | 0.109 | 0.960 |

v2 gave lower errors and higher correlations on both sets. Some things to keep in mind:

- v1 and v2 differ in **design as well as training** (v2 has no squeeze step, v1 does), so I cannot give the credit to the denoising training alone.
- On clean inputs, v2 almost copies the window (temporal validation MAE 0.022, correlation 0.999), so those scores say little. The damaged-input results are more informative.
- v1 had already trained on the HD 209458 windows in this validation set, which if anything helps v1.
- Good reconstruction does not prove the model learned meaningful astrophysical features or that it works on other stars.

### Clustering: learned features vs simple features

The learned-feature groups had 287, 704, 156 and 11 windows.

| Measure | Learned features | Hand-made features |
| --- | --- | --- |
| Silhouette score | 0.311 | 0.416 |

- Agreement between the two groupings was almost zero (ARI 0.001, NMI 0.021).
- The learned groups were identical across five random seeds (ARI 1.0).
- The simple features gave better-separated groups by this score, so I cannot claim the AI features are better.
- Stable groups are not the same as meaningful groups. They still need to be checked against verified labels and other stars.

**These are reconstruction and clustering results. They are not evidence of new astronomical discoveries.**

---

## ⚠️ Limitations (and things I found while checking my own work)

1. **Small sample.** Only four stars, so nothing here shows how the models behave on stars in general.
2. **Overlapping windows.** Neighbouring windows share half their data, so they are not independent samples.
3. **The v1 validation split.** I trained v1 with Keras's automatic 15% validation split, which takes the *last* windows in the array. My windows are stored one star after another, so it set aside all 75 Algol windows and the last 80 TOI-700 windows. I worked this out from the code and the log (28 training steps per epoch). So v1 actually trained on 876 windows (all of HD 209458 plus the first 33 of TOI-700), not on all three stars.
4. **RR Lyrae flux is not fixed.** About 30% of its raw flux values are negative, and I did not find the cause. I re-downloaded the official SPOC Sector 14 product, which has SAP and PDCSAP flux columns. The first flux values I checked matched the ones in my original file, and I have not yet checked or used the other columns. All RR Lyrae results, including the held-out ones, come from the current windows and should be treated with care.
5. **Different designs.** v1 and v2 are different architectures, so the comparison is not a clean test of one idea.
6. **No verified labels.** Reconstruction error is not the probability of an event. The 33 Algol runs are not confirmed eclipses. I have not checked the light curves at the predicted transit times.
7. **Period estimation is incomplete.** Autocorrelation at three fixed lags is not a validated period measurement.
8. **Cluster stability is not validity.** The same groups across five seeds do not show the groups are real star types.

---

## 🔭 What I want to do next

- Check target details and event labels against independent catalogues.
- Define clear event windows and good negative examples.
- Add more stars and split train, validation and test sets **by star**.
- Compare simple baselines with a 1D CNN if the bigger dataset supports it.
- Measure precision, recall, F1-score and PR-AUC, and test on stars the model has never seen.
- Fix the RR Lyrae flux issue and rebuild its windows.

---

## 📁 What is in this repository

| File or folder | What it is |
| --- | --- |
| `notebook_star_trace.ipynb` | The full notebook (data, cleaning, models, evaluation, clustering) |
| `app.py` | The Streamlit dashboard |
| `requirements.txt` | Packages needed to run the dashboard |
| `STARTRACE_deployment_assets.zip` | The two trained models plus result tables, figures and reports used by the dashboard |
| `VIsuals/` | The dashboard demo GIF and screenshots |

### Run the notebook

1. Open the notebook in Google Colab with the badge at the top.
2. Run the cells in order. Internet access is needed to download TESS data and query the NASA Exoplanet Archive.
3. Files are written under `/content/STARTRACE` (`data/raw`, `data/processed`, `results`, plus the two `.keras` model files).

I fixed the random seed at 42, but exact numbers may differ slightly between runs or library versions.

**Libraries:** `lightkurve`, `tensorflow` / `keras`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `astropy`, `ipywidgets`, `requests`.

---

## 📚 Data sources

- **TESS observations:** NASA's Mikulski Archive for Space Telescopes (MAST), accessed with `lightkurve`.
- **Planet catalogue and transit timing:** NASA Exoplanet Archive.
- **Extra transit timing for HD 209458:** NASA's transit service and the published timing solutions listed above.

---

## 💬 Why I built this

I am an undergraduate student who wants to work in space science, and I am interested in using AI and computation to study signals from space. STARTRACE let me practise with real mission data: building a time-series pipeline, training neural networks, and checking what their results really mean.

The biggest lesson was that a model scoring well is not the same as answering the science question. I had to check my own data, my splits and my comparisons, and I found mistakes along the way. I have kept those in this README instead of hiding them. I have not solved stellar signal classification, and I am not presenting STARTRACE as a discovery tool. It is an honest starting point that I want to improve with better data and stronger testing.

**My goal is to learn from real observations, test on stars the model has never seen, and only make claims that the evidence supports.**
