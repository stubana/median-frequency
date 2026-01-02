# median-frequency
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch

# ================= PARAMETERS =================
TR = 2.3   
fs = 1 / TR
FREQ_RANGE = (0.01, 0.25)   
MIN_LENGTH = 100
VAR_EPS = 1e-8

voxelwise_dir = Path('/home/stubanadean/voxelwise_timeseries_nilearn')
output_dir = Path('/home/stubanadean/mf_results')
output_dir.mkdir(exist_ok=True)
layers = ['Interoception', 'Exteroception', 'Cognition']

# ================= MEDIAN FREQUENCY FUNCTION =================
def median_frequency(x, fs, freqrange=FREQ_RANGE):
    f, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    mask = (f >= freqrange[0]) & (f <= freqrange[1])
    f = f[mask]
    psd = psd[mask]
    cumsum = np.cumsum(psd)
    mf_idx = np.where(cumsum >= (cumsum[-1] / 2))[0][0]
    return f[mf_idx]

# ================= MAIN ANALYSIS =================
results = []

for layer in layers:
    layer_dir = voxelwise_dir / layer
    roi_files = list(layer_dir.glob("*_voxelwise_timeseries.npy"))
    
    for ts_file in roi_files:
        roi_name = ts_file.name.replace("_voxelwise_timeseries.npy", "")
        ts_data = np.load(ts_file)  # shape: time x voxels
        T, V = ts_data.shape
        
        if T < MIN_LENGTH:
            print(f"Skipping {roi_name}: too short")
            continue
        
        voxel_std = np.std(ts_data, axis=0)
        valid_voxels = voxel_std > VAR_EPS
        ts_data = ts_data[:, valid_voxels]
        
        if ts_data.shape[1] == 0:
            print(f"Skipping {roi_name}: no valid voxels")
            continue
        
        mf_vals = np.array([median_frequency(ts_data[:, v], fs) for v in range(ts_data.shape[1])])
        
        results.append({
            "Layer": layer,
            "ROI": roi_name,
            "Timepoints": T,
            "Num_voxels_used": ts_data.shape[1],
            "Mean_MF_Hz": np.mean(mf_vals),
            "Median_MF_Hz": np.median(mf_vals),
            "Std_MF_Hz": np.std(mf_vals),
            "Min_MF_Hz": np.min(mf_vals),
            "Max_MF_Hz": np.max(mf_vals)
        })
        print(f"{layer} | {roi_name} -> Mean MF: {np.mean(mf_vals):.4f} Hz")

# ================= SAVE RESULTS =================
outpath = output_dir / "median_frequency_summary.csv"
pd.DataFrame(results).to_csv(outpath, index=False)
print(f"\nSaved results to: {outpath}")
