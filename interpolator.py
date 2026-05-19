"""
Construct correction objects for the Lyman-alpha forest 1D flux power spectrum.

"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.direction': 'in',    
    'ytick.direction': 'in',
    'ytick.right': True,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'legend.frameon': False,     
})

BASE_DIR = ""
COR_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "emu")
PLOT_OUT_FILE = "plots/correction.pdf"

Z_ORDER = ["5.0", "4.6", "4.2"]
LOGK_BINS = np.round(np.arange(-2.2, -0.65, 0.1), 1)
TARGET_K = 10.0 ** LOGK_BINS  

PIXEL_COR = np.array([
    0.99999668, 0.99999474, 0.99999167, 0.99998679,
    0.99997907, 0.99996682, 0.99994742, 0.99991667,
    0.99986793, 0.99979069, 0.99966829, 0.99947431,
    0.99916694, 0.99867995, 0.99790851, 0.99668684,
], dtype=np.float64)

PATCHY_COR_DICT = {
    "5.0": np.array([1.054, 1.040, 1.036, 1.031, 1.024, 1.028, 1.024, 1.023, 1.013, 0.993, 0.972, 0.946, 0.928, 0.914, 0.905, 0.883], dtype=np.float64),
    "4.6": np.array([1.038, 1.029, 1.025, 1.020, 1.021, 1.017, 1.019, 1.012, 1.001, 0.980, 0.960, 0.935, 0.919, 0.917, 0.913, 0.879], dtype=np.float64),
    "4.2": np.array([1.029, 1.023, 1.020, 1.016, 1.014, 1.012, 1.011, 1.004, 0.990, 0.975, 0.949, 0.923, 0.909, 0.913, 0.913, 0.862], dtype=np.float64),
}

SIIII_PARAMS_DICT = {
    "5.0": (5.91e-2, -2.33e-2, 4.80e-3, 3.12e-2),
    "4.6": (5.85e-2, -2.32e-2, 5.58e-3, 3.31e-2),
    "4.2": (5.63e-2, -2.25e-2, 5.46e-3, 3.60e-2),
}

class LyaCorrector:
    def __init__(self, apply_patchy: bool):
        self.apply_patchy = apply_patchy
        self.pixel_multiplier = 1.0 / PIXEL_COR
        self.patchy_dict = PATCHY_COR_DICT
        self.siiii_dict = SIIII_PARAMS_DICT
        self.target_k = TARGET_K

    def get_siiii_correction(self, z_str: str) -> np.ndarray:
        k_auto, k_cross, a_auto, a_cross = self.siiii_dict[z_str]
        return (1.0 
                + a_auto * np.exp(self.target_k / k_auto) 
                + a_cross * np.exp(self.target_k / k_cross) * np.cos(2271.0 * self.target_k))

    def get_total_correction(self, z_str: str) -> np.ndarray:
        total_corr = self.pixel_multiplier.copy()
        total_corr *= self.get_siiii_correction(z_str)
        if self.apply_patchy:
            total_corr *= self.patchy_dict[z_str]
        return total_corr

def plot_correction_illustration(corrector: LyaCorrector, out_path: str):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    
    COLORS = ['#0072B2', '#D55E00', '#009E73']
    
    for i, z_str in enumerate(Z_ORDER):
        ax = axes[i]
        
        c_pixel = corrector.pixel_multiplier
        c_siiii = corrector.get_siiii_correction(z_str)
        c_patchy = corrector.patchy_dict[z_str]
        
        c_total = c_pixel * c_siiii * c_patchy
        
        ax.plot(LOGK_BINS, c_pixel, color=COLORS[0], label=r'Pixel Sampling ($C_{\mathrm{pixel}}$)')
        ax.plot(LOGK_BINS, c_siiii, color=COLORS[1], label=r'Si III ($C_{\mathrm{Si\,III}}$)')
        ax.plot(LOGK_BINS, c_patchy, color=COLORS[2], label=r'Patchy Reionization ($C_{\mathrm{patchy}}$)')
        ax.plot(LOGK_BINS, c_total, color='#333333', ls='--', lw=2.5, label=r'Total ($C_{\mathrm{total}}$)')
        
        ax.axhline(1.0, color='#555555', ls=':', lw=1.5, zorder=0)
        
        ax.set_title(rf'$z={z_str}$', fontsize=16)
        ax.set_xlabel(r'$\log_{10}(k_\mathrm{f}~[\mathrm{s} \, \mathrm{km}^{-1}])$')
        ax.grid(True, ls=':', color='#dddddd')
        
        if i == 0:
            ax.set_ylabel(r'Correction Multiplier $C(k_\mathrm{f})$')
            
        if i == 2:
            ax.legend(loc='best', fontsize=11)
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    corrector_no_patchy = LyaCorrector(apply_patchy=False)
    with open(os.path.join(OUT_DIR, "lya_corrector_no_patchy.pkl"), "wb") as f:
        pickle.dump(corrector_no_patchy, f)
    
    corrector_with_patchy = LyaCorrector(apply_patchy=True)
    with open(os.path.join(OUT_DIR, "lya_corrector_with_patchy.pkl"), "wb") as f:
        pickle.dump(corrector_with_patchy, f)
        
    plot_correction_illustration(corrector_with_patchy, PLOT_OUT_FILE)
    
if __name__ == "__main__":
    main()