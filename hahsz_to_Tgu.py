"""
Build an interpolator from (z, z_rei, H_A, H_S) -> (T0, gamma, u0)

"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'legend.frameon': False
})

BASE_SIM_DIR = "data" 
OUT_MODEL_PATH = "emu/tgu_interpolator.pkl"
OUT_PLOT_PATH = "plots/tgu_test.pdf"

MAX_INDEX = 100
Z_ORDER = ["5.0", "4.6", "4.2"]
Z_OBS_VALUE = {"5.0": 5.0, "4.6": 4.6, "4.2": 4.2}
FILE_NUM_MAP = {"5.0": 0, "4.6": 1, "4.2": 2}

PARAM_LABELS = [
    r"$T_0 ~ [10^4\,\mathrm{K}]$", 
    r"$\gamma$", 
    r"$u_0 ~ [\mathrm{eV} \, m_\mathrm{p}^{-1}]$"
]

@dataclass
class GlobalTGUInterpolator:
    scaler: StandardScaler
    interp: RBFInterpolator

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_all_tgu_data(base_sim_dir: str, max_index: int) -> Tuple[np.ndarray, np.ndarray]:
    X_all, Y_all = [], []
    
    for z_str in Z_ORDER:
        num = FILE_NUM_MAP[z_str]
        z_obs = Z_OBS_VALUE[z_str]
        file_path = os.path.join(base_sim_dir, f"all_pk_num{num}_cdm.pkl")
        data = load_pkl(file_path)
        ids = np.unique(data["index"])
        ids = ids[(ids >= 0) & (ids < max_index)]
        
        x_list, y_list = [], []
        for idx in ids:
            row = data[data["index"] == idx][0]
            x_list.append([z_obs, row["z"], row["ha"], row["hs"]])
            y_list.append([row["T"], row["gamma"], row["u"]])
            
        x = np.asarray(x_list, dtype=np.float64)
        y = np.asarray(y_list, dtype=np.float64)

        xu, ui = np.unique(x, axis=0, return_index=True)
        X_all.append(xu)
        Y_all.append(y[ui])
        print(f"    z_obs={z_obs}:  {len(xu)} points loaded.")
        
    return np.vstack(X_all), np.vstack(Y_all)

def main():
    print("========== Global TGU Interpolator Training ==========")
    
    os.makedirs(os.path.dirname(OUT_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PLOT_PATH), exist_ok=True)

    X, Y = load_all_tgu_data(BASE_SIM_DIR, MAX_INDEX)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    scaler_test = StandardScaler()
    X_train_scaled = scaler_test.fit_transform(X_train)
    interp_test = RBFInterpolator(X_train_scaled, Y_train, kernel="cubic")

    X_test_scaled = scaler_test.transform(X_test)
    Y_pred = interp_test(X_test_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    z_obs_test = X_test[:, 0]
    unique_z = np.unique(z_obs_test)
    
    colors_map = {5.0: "C0", 4.6: "C1", 4.2: "C2"} 
    
    for j, param_name in enumerate(PARAM_LABELS):
        ax = axes[j]
        true_vals = Y_test[:, j].copy()
        pred_vals = Y_pred[:, j].copy()
        
        if j == 0:
            true_vals = true_vals / 10000.0
            pred_vals = pred_vals / 10000.0
        
        for z_val in unique_z:
            mask = (z_obs_test == z_val)
            ax.scatter(true_vals[mask], pred_vals[mask], 
                       color=colors_map.get(z_val, "k"), 
                       s=60, alpha=0.8, edgecolor='none', 
                       label=rf"$z={z_val}$", zorder=2)
            
        min_val = min(np.min(true_vals), np.min(pred_vals))
        max_val = max(np.max(true_vals), np.max(pred_vals))
        margin = (max_val - min_val) * 0.1
        
        ax.plot([min_val - margin, max_val + margin], 
                [min_val - margin, max_val + margin], 
                color='gray', linestyle='--', linewidth=1.5, alpha=0.8, 
                label=r"$y=x$", zorder=1)
        
        ax.set_title(param_name, pad=15)
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.grid(alpha=0.25, linestyle=':')
        
        if j == 0:
            ax.legend(loc='upper left')
            
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    interp_final = RBFInterpolator(X_scaled, Y, kernel="cubic")
    
    global_tgu_model = GlobalTGUInterpolator(scaler=scaler_final, interp=interp_final)
    save_pkl(OUT_MODEL_PATH, global_tgu_model)

if __name__ == "__main__":
    main()