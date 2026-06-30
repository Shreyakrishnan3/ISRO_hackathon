!pip install lightkurve transitleastsquares scikit-learn astropy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from transitleastsquares import transitleastsquares
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import astropy.units as u

# 1. DATA INGESTION & PREPROCESSING PIPELINE

def download_and_clean_lc(tic_id, sector=None):
    """
    Downloads raw TESS light curves using Lightkurve, removes outliers,
    and flattens intrinsic stellar variability to isolate narrow dips.
    """
    print(f"[+] Fetching Light Curve for TIC {tic_id}...")
    try:
        # Search for SPOC (Science Processing Operations Center) high-cadence data
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector)
        if len(search_result) == 0:
            print(f"[-] No TESS data found for TIC {tic_id}")
            return None

        # Download the first available high-cadence product
        lc = search_result[0].download()

        # Preprocessing: Remove NaNs, quality flags, and extreme outliers
        lc = lc.remove_nans().remove_outliers(sigma=5)

        # Flattening: Flatten long-term trends (like starspots) using a Savitzky-Golay filter
        # window_length is set to ~0.5 days to preserve short transit features
        flattened_lc = lc.flatten(window_length=101)

        return flattened_lc
    except Exception as e:
        print(f"[-] Error downloading TIC {tic_id}: {e}")
        return None


# 2. SIGNAL DETECTION & FEATURE EXTRACTION (TLS)

def extract_signal_features(lc):
    """
    Uses Transit Least Squares (TLS) to detect periodic dips.
    Extracts high-utility features optimized for distinguishing
    transits from eclipsing binaries and blends.
    """
    if lc is None:
        return None

    print("[+] Running Transit Least Squares (TLS) detection...")
    # Convert TimeStub and Flux stub to standard numpy arrays for TLS
    time = lc.time.value
    flux = lc.flux.value

    model = transitleastsquares(time, flux)
    results = model.power()

    # Calculate basic feature vectors for classification
    features = {
        "s_to_n": float(results.snr),                     # Signal-to-Noise Ratio
        "period": float(results.period),                  # Detected period in days
        "duration": float(results.duration),              # Transit duration in days
        "depth": float(1.0 - results.depth),              # Depth of the dip
        "odd_even_mismatch": float(results.odd_even_mismatch), # EB indicator (secondary eclipse difference)
        "rp_rs": float(results.rp_rs if results.rp_rs is not None else 0), # Planet-to-star radius ratio
        "distinct_dips": float(len(results.transit_times))
    }
    return features, results


# 3. AI CLASSIFICATION ENGINE

class ExoplanetClassifierPipeline:
    def __init__(self):
        # Using Random Forest as an optimal balance of speed and tabular accuracy for hackathons
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        self.classes = ['transit', 'eclipse', 'blend', 'noise']

    def generate_synthetic_training_data(self):
        """
        Generates simulated training features mimicking TESS objects of interest (TOIs),
        eclipsing binaries (EBs), blended background sources, and variable noise.
        """
        print("[+] Generating baseline training catalog features...")
        np.random.seed(42)
        n_samples = 250

        data = []
        for _ in range(n_samples):
            # Transits: High SNR, tiny-to-moderate depths, stable odd-even transitions
            data.append([np.random.uniform(8, 30), np.random.uniform(1, 15), np.random.uniform(0.05, 0.2), np.random.uniform(0.001, 0.02), np.random.uniform(0, 1.2), np.random.uniform(0.01, 0.1), np.random.randint(3, 10), 0])
            # Eclipses: Massive SNR, deep dips, large secondary odd-even mismatches
            data.append([np.random.uniform(15, 100), np.random.uniform(0.5, 5), np.random.uniform(0.1, 0.4), np.random.uniform(0.05, 0.3), np.random.uniform(3.5, 12), np.random.uniform(0.2, 0.6), np.random.randint(5, 20), 1])
            # Blends: Diluted signals, shallow depths, highly localized field distortions, low SNR
            data.append([np.random.uniform(3, 7), np.random.uniform(2, 20), np.random.uniform(0.05, 0.3), np.random.uniform(0.0005, 0.003), np.random.uniform(0.5, 3.0), np.random.uniform(0.01, 0.05), np.random.randint(2, 8), 2])
            # Noise / Variable stars: Very low SNR, erratic periods, minimal geometric structures
            data.append([np.random.uniform(0, 2.8), np.random.uniform(0.1, 30), np.random.uniform(0.01, 0.5), np.random.uniform(0.0001, 0.005), np.random.uniform(0, 5), np.random.uniform(0, 0.05), np.random.randint(1, 15), 3])

        columns = ["s_to_n", "period", "duration", "depth", "odd_even_mismatch", "rp_rs", "distinct_dips", "label"]
        df = pd.DataFrame(data, columns=columns)

        X = df.drop(columns=["label"])
        y = df["label"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_pipeline(self):
        X_train, X_test, y_train, y_test = self.generate_synthetic_training_data()
        self.model.fit(X_train, y_train)

        # Evaluate internally
        preds = self.model.predict(X_test)
        print("\n--- Model Training Classification Report ---")
        print(classification_report(y_test, preds, target_names=self.classes))

    def classify_signal(self, feature_dict):
        feat_df = pd.DataFrame([feature_dict])
        pred_idx = self.model.predict(feat_df)[0]
        probabilities = self.model.predict_proba(feat_df)[0]
        return self.classes[pred_idx], probabilities[pred_idx]


# 4. VISUALIZATION AND DIAGNOSTICS GENERATOR

def generate_diagnostic_plots(lc, tls_results, target_name, assigned_class, confidence):
    """
    Creates validation charts containing the full flattened light curve,
    the periodogram power spectral density, and the phase-folded transit profile.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'hspace': 0.4})

    # Plot 1: Full Light Curve with marked transits
    axes[0].plot(lc.time.value, lc.flux.value, color='black', alpha=0.6, label='Flattened Flux')
    axes[0].set_title(f"Target: {target_name} | AI Classification: {assigned_class.upper()} ({confidence*100:.1f}% Confidence)")
    axes[0].set_xlabel("Time (BJD - 2457000)")
    axes[0].set_ylabel("Normalized Flux")
    axes[0].grid(True, linestyle='--')

    # Highlight identified transit center markers
    for t_center in tls_results.transit_times:
        axes[0].axvline(t_center, color='red', alpha=0.4, linestyle=':', label='Detected Dips' if t_center == tls_results.transit_times[0] else "")
    axes[0].legend(loc='best')

    # Plot 2: TLS Periodogram (Power Spectrum)
    axes[1].axvline(tls_results.period, alpha=0.4, color='red', label=f"Peak Period: {tls_results.period:.4f} days")
    axes[1].plot(tls_results.periods, tls_results.power, color='darkblue')
    axes[1].set_xlabel("Period (days)")
    axes[1].set_ylabel("SDE / Power")
    axes[1].set_title("Transit Least Squares Periodogram Spectrum")
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--')

    # Plot 3: Phase-Folded Signal with Fit Overlaid
    axes[2].plot(tls_results.folded_phase, tls_results.folded_y, 'k.', alpha=0.3, label='Folded Observations')
    axes[2].plot(tls_results.model_folded_phase, tls_results.model_folded_model, 'r-', lw=2, label='TLS Astrophysical Fit')
    axes[2].set_xlabel("Phase")
    axes[2].set_ylabel("Normalized Flux")
    axes[2].set_title("Phase-Folded View to Verify Geometry")
    axes[2].legend(loc='best')
    axes[2].grid(True, linestyle='--')

    # Save chart output
    output_filename = f"diagnostic_{target_name}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved analysis visualizations to: '{output_filename}'")


# UPDATED DATA INGESTION 

def download_and_clean_lc(tic_id, sector=None):
    """
    Downloads raw TESS light curves using Lightkurve across any available sector,
    removes outliers, and flattens intrinsic stellar variability.
    """
    print(f"[+] Fetching Light Curve for TIC {tic_id}...")
    try:
        # Search without forcing a specific sector if it fails initial query
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector)

        # Fallback: If sector specific query is empty, look for ANY sector
        if len(search_result) == 0 and sector is not None:
            print(f"[!] Target not in Sector {sector}. Searching all available sectors...")
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")

        if len(search_result) == 0:
            print(f"[-] No TESS data found anywhere for TIC {tic_id}")
            return None

        # Download the first available high-cadence data product
        print(f"[+] Found {len(search_result)} data products. Downloading index 0...")
        lc = search_result[0].download()

        # Preprocessing operations
        lc = lc.remove_nans().remove_outliers(sigma=5)
        flattened_lc = lc.flatten(window_length=101)

        return flattened_lc
    except Exception as e:
        print(f"[-] Error downloading TIC {tic_id}: {e}")
        return None


# UPDATED MAIN EXECUTION 

if __name__ == "__main__":
    
    print("BHARATIYA ANTARIKSH HACKATHON - EXOPLANET PIPELINE INITIALIZATION")
    

    # Initialize and train our vetting engine
    ai_pipeline = ExoplanetClassifierPipeline()
    ai_pipeline.train_pipeline()

    # Standard benchmark target: WASP-126 b (Guaranteed TESS Data catalog footprint)
    demo_tic_id = "25155310"

    # Step 1: Download & Clean using the new multi-sector fallback logic
    cleaned_lc = download_and_clean_lc(demo_tic_id)

    if cleaned_lc is not None:
        # Step 2: Extract structural parameters & TLS periodogram
        features, tls_results = extract_signal_features(cleaned_lc)

        # Step 3: Pass through AI vetting classifier
        predicted_category, confidence_score = ai_pipeline.classify_signal(features)

        # Step 4: Display parameters & generate dynamic charts
        
        print(f"Target Identification     : TIC {demo_tic_id}")
        print(f"AI Final Categorization   : {predicted_category.upper()}")

        scalar_confidence = float(confidence_score[0]) if hasattr(confidence_score, '__iter__') else float(confidence_score)
        print(f"Vetting Confidence Score  : {scalar_confidence * 100:.2f}%")
        print(f"Signal-to-Noise Ratio(SNR): {features['s_to_n']:.2f}")
        print("\n--- Estimated Astrophysical Parameters ---")
        print(f"Orbital Period            : {features['period']:.5f} days")
        print(f"Transit Duration          : {features['duration'] * 24 * 60:.2f} minutes")
        print(f"Transit Depth             : {features['depth'] * 100:.4f}% flux reduction")
     

        # Step 5: Draw, Save, and Display Plots directly in the Notebook
        generate_diagnostic_plots(cleaned_lc, tls_results, f"TIC_{demo_tic_id}", predicted_category, scalar_confidence)

        # Display the saved PNG inline inside your notebook cell output
        from IPython.display import Image, display
        display(Image(filename=f"diagnostic_TIC_{demo_tic_id}.png"))

    else:
        print("\n[-] Pipeline run failed. Verification target could not be acquired.")
