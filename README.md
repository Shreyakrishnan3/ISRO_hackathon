# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

This repository contains the end-to-end AI-driven data processing and vetting pipeline developed for **Problem Statement 7** of the **Bharatiya Antariksh Hackathon**.

## 👥 Team Information
* **Team Name:** StellarScan
* **Institution:** TKM College of Engineering, Kollam, Kerala
* **Team Members:**
  * Megha S
  * Shreya Krishnan (Team Lead)
  * Malavi G Kamal

---

## 🌌 Project Overview & Objective
The identification of sub-percent planetary transit signals in crowded stellar fields is heavily hindered by stellar blending, detector thermal responses, and intrinsic stellar activity (such as starspots). 

**StellarScan** addresses this challenge by deploying a robust, modular automated pipeline that:
1. Programmatically ingests high-cadence time-series photometry from space-based assets (TESS) via the MAST archive.
2. Conditions noisy light curves by removing extreme outliers and flattening low-frequency stellar variations.
3. Utilizes physics-grounded **Transit Least Squares (TLS)** optimized with analytical stellar limb-darkening laws to locate periodic dips.
4. Maps structural parameters into a **Random Forest Machine Learning Framework** to distinguish true exoplanetary transits from astrophysical false positives (e.g., eclipsing binaries and blends).

---

## 🛠️ Pipeline Architecture & Methodology

### 1. Data Cleaning & Conditioning
* **Outlier Removal:** Prunes measurement artifacts and cosmic ray strikes using a moving-window median filter with a two-sided $5\sigma$ clipping threshold.
* **Variability Flattening:** Eliminates rotational modulations from starspots using a high-pass **Savitzky-Golay (SG) filter** (window length $\approx 0.5$ days). This preserves the steep ingress/egress transit shapes while normalizing the stellar continuum to 1.0.

### 2. Signal Detection (TLS vs. BLS)
Instead of searching for basic geometric box shapes, our implementation relies on **Transit Least Squares (TLS)**. TLS searches for realistic transit paths using analytical limb-darkening equations:
$$\chi^2 = \sum_{i=1}^{N} \left(\frac{y_i - y_{\text{model}}(t_i)}{\sigma_i}\right)^2$$
This optimization vastly improves the Signal Detection Efficiency (SDE) for low-amplitude, shallow Earth and Neptune-sized planetary targets.

### 3. Machine Learning Vetting Engine
Once a signal peak is localized, the pipeline extracts a 7-dimensional feature vector for classification:
* **Signal-to-Noise Ratio (SNR):** Overall strength of the folded signal profile.
* **Orbital Period ($P$):** Temporal duration of the planet's orbit in days.
* **Transit Duration ($T_{\text{dur}}$):** The specific duration from ingress to egress.
* **Transit Depth ($\delta$):** Maximum drop in normalized flux.
* **Odd-Even Mismatch:** Quantifies depth disparities between odd and even events to flag eclipsing binaries.
* **Radius Ratio ($R_p/R_*$):** Geometric planet-to-star size ratio.
* **Distinct Dips:** Total count of captured transits across the sector time series.

The **Random Forest Ensemble** architecture categorizes these feature vectors into four clear quadrants: `Transit`, `Eclipse`, `Blend`, or `Noise`.

---

## 📊 Pipeline Validation & Results (Benchmark Target: TIC 25155310)
To test performance under realistic operational constraints, the pipeline evaluated the known exoplanetary system **WASP-126 b**:
* **Dataset Scale:** 18,254 independent photometric data points evaluated across 2,554 trial periods.
* **Recovered Orbital Period:** **3.28717 days** (Achieving **>99.9% accuracy** against official NASA Exoplanet Archive baselines).
* **Estimated Transit Duration:** **198.85 minutes**.
* **Calculated Transit Depth:** **0.2370%** flux reduction.
* **Signal Significance:** Verified at an exceptional operational **SNR of 43.63**.
* **AI Vetting Output:** Categorized cleanly as a true **TRANSIT** with an automated vetting confidence score of **58.50%** against complex false-positive models.

---

## 📈 Visual Diagnostics

Below is the multi-panel validation plot generated automatically by the StellarScan pipeline, tracking the full flattened light curve, periodogram power spectral density, and the phase-folded geometry layout:

![StellarScan Output Diagnostics](output_hackathon.png)

* **Top Panel:** Displays the clean, unambiguous periodogram spike right at 3.2872 days with zero competing mathematical aliases.
* **Bottom Panel:** Shows the phase-folded light curve overlaid with the analytical TLS limb-darkened transit model fit (red line), proving a distinctive planet-like U-shaped transit signature.

---

## 💻 Technical Setup & Dependencies

Ensure your environment features the required astronomical and machine learning packages before executing the core script:

```bash
pip install lightkurve transitleastsquares scikit-learn numpy pandas matplotlib astropy
```

### To Run the Pipeline:
```bash
python pipeline.py
```

---

## 📜 Key References
1. Lightkurve Collaboration et al., 2018. Astrophysics Source Code Library.
2. Hippke, M., & Heller, R. 2019. Astronomy & Astrophysics, 623, A39 (Transit Least Squares implementation).
3. Pedregosa et al., 2011. Journal of Machine Learning Research, 12, 2825-2830 (Scikit-Learn documentation).
