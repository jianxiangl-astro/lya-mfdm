"""
Main Bayesian inference script for pure and mixed fuzzy dark matter constraints from the Lyman-alpha forest 1D flux power spectrum.

"""
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from multiprocessing import Pool
import emcee
from chainconsumer import ChainConsumer
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator, NearestNDInterpolator
import warnings

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False
})

BASE_SIM_DIR = "data"
OBS_FILE = os.path.join(BASE_SIM_DIR, "lya_data.pkl")

EMU_ROOT = "emu"
NN_EMU_DIR = os.path.join(EMU_ROOT, "emu_N100")
TGU_EMU_FILE = os.path.join(EMU_ROOT, "tgu_interpolator.pkl")
CORR_NO_PATCHY_FILE = os.path.join(EMU_ROOT, "lya_corrector_no_patchy.pkl")
CORR_WITH_PATCHY_FILE = os.path.join(EMU_ROOT, "lya_corrector_with_patchy.pkl")

OUT_RESULTS_DIR = "results"
OUT_PLOTS_DIR = "plots/mcmc"
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

MCMC_STEPS = 400000
N_WALKERS = 56
N_PROC = min(56, os.cpu_count() or 1)
N_FOLDS = 5
HIDDEN_DIM = 256
F_EPS = 1e-8

Z_ORDER = ["5.0", "4.6", "4.2"]
Z_FLOAT = {"5.0": 5.0, "4.6": 4.6, "4.2": 4.2}

BOUNDS_BASE = [
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],  # z=5.0 
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],  # z=4.6
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],  # z=4.2
]
BOUNDS_FULL = np.array([[-23.0, -19.0], [0.0, 1.0]] + BOUNDS_BASE, dtype=np.float64) 
BOUNDS_FIXED = np.array([[-23.0, -19.0]] + BOUNDS_BASE, dtype=np.float64)            

G_NN = None
G_TGU = None
G_OBS = None
G_CORRECTORS = None

@dataclass
class GlobalTGUInterpolator:
    scaler: any
    interp: RBFInterpolator

class LyaCorrector:
    def __init__(self, apply_patchy: bool):
        self.apply_patchy = apply_patchy
        self.pixel_multiplier = None
        self.patchy_dict = None
        self.siiii_dict = None
        self.target_k = None


    def get_siiii_correction(self, z_str: str) -> np.ndarray:
        k_auto, k_cross, a_auto, a_cross = self.siiii_dict[z_str]
        return (1.0 + a_auto * np.exp(self.target_k / k_auto) + 
                a_cross * np.exp(self.target_k / k_cross) * np.cos(2271.0 * self.target_k))


    def get_total_correction(self, z_str: str) -> np.ndarray:
        total_corr = self.pixel_multiplier.copy()
        total_corr *= self.get_siiii_correction(z_str)
        if self.apply_patchy:
            total_corr *= self.patchy_dict[z_str]
        return total_corr

sys.modules.setdefault("hahsz_to_Tgu", sys.modules[__name__])
sys.modules.setdefault("interpolator", sys.modules[__name__])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_mlp(in_dim, out_dim=16):
    return nn.Sequential(
        nn.Linear(in_dim, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, out_dim),
    )

def load_system():
    print("[*] Loading all emulators and data...")
    nn_pack = {
        "scaler_cdm": load_pkl(os.path.join(NN_EMU_DIR, "scaler_cdm.pkl")),
        "scaler_res": load_pkl(os.path.join(NN_EMU_DIR, "scaler_res.pkl")),
        "cdm_models": [], "res_models": []
    }
    for fold in range(N_FOLDS):
        c_mod = build_mlp(5, 16); c_mod.load_state_dict(torch.load(os.path.join(NN_EMU_DIR, f"model_cdm_fold{fold}.pth"), map_location="cpu")); c_mod.eval()
        r_mod = build_mlp(7, 16); r_mod.load_state_dict(torch.load(os.path.join(NN_EMU_DIR, f"model_res_fold{fold}.pth"), map_location="cpu")); r_mod.eval()
        nn_pack["cdm_models"].append(c_mod); nn_pack["res_models"].append(r_mod)
    
    tgu_pack = load_pkl(TGU_EMU_FILE)
    
    obs_raw = load_pkl(OBS_FILE)
    obs_pack = {}
    for z_str in Z_ORDER:
        cov = np.asarray(obs_raw[z_str]["Cov"], dtype=np.float64)
        cov_inv = np.asarray(obs_raw[z_str]["Cov-1"], dtype=np.float64) if "Cov-1" in obs_raw[z_str] else np.linalg.inv(cov)
        obs_pack[z_str] = {"Pk": np.asarray(obs_raw[z_str]["Pk"], dtype=np.float64), "Cov": cov, "Cov_inv": cov_inv}
    
    corr_pack = {
        "none": None,
        "no_patchy": load_pkl(CORR_NO_PATCHY_FILE),
        "with_patchy": load_pkl(CORR_WITH_PATCHY_FILE)
    }
    return nn_pack, tgu_pack, obs_pack, corr_pack

def init_worker(nn_pack, tgu_pack, obs_pack, corr_pack):
    global G_NN, G_TGU, G_OBS, G_CORRECTORS
    torch.set_num_threads(1)
    G_NN = nn_pack; G_TGU = tgu_pack; G_OBS = obs_pack; G_CORRECTORS = corr_pack

def predict_tgu(z_str, zrei, ha, hs):
    x = np.array([[Z_FLOAT[z_str], zrei, ha, hs]], dtype=np.float64)
    return np.asarray(G_TGU.interp(G_TGU.scaler.transform(x))[0], dtype=np.float64)

def predict_pk(m, f, zrei, ha, hs, taueff, z_str):
    z_obs = Z_FLOAT[z_str]
    x_cdm = G_NN["scaler_cdm"].transform(np.array([[z_obs, zrei, ha, hs, taueff]], dtype=np.float64))
    x_res = G_NN["scaler_res"].transform(np.array([[z_obs, m, f, zrei, ha, hs, taueff]], dtype=np.float64))
    
    with torch.no_grad():
        cdm_st = torch.stack([m(torch.tensor(x_cdm, dtype=torch.float32))[0] for m in G_NN["cdm_models"]]).mean(0).numpy()
        res_st = torch.stack([m(torch.tensor(x_res, dtype=torch.float32))[0] for m in G_NN["res_models"]]).mean(0).numpy()
    
    logp = cdm_st if f <= F_EPS else cdm_st + f * res_st
    return np.power(10.0, logp)

def log_probability(theta, case_name, fixed_f):
    theta = np.asarray(theta, dtype=np.float64)
    bounds = BOUNDS_FIXED if fixed_f else BOUNDS_FULL

    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]): return -np.inf, -np.inf

    m = theta[0]
    f = 1.0 if fixed_f else theta[1]
    z_params = theta[1:] if fixed_f else theta[2:]

    T_vals, u_vals, ln_prior_T = [], [], 0.0
    T_MEANS = {"5.0": 9286.5, "4.6": 8986.5, "4.2": 9155.5}
    
    for i, z_str in enumerate(Z_ORDER):
        zrei, ha, hs, tau = z_params[4*i : 4*i+4]
        T, gamma, u = predict_tgu(z_str, zrei, ha, hs)
        T_vals.append(T)
        u_vals.append(u)
        ln_prior_T += -0.5 * ((T - T_MEANS[z_str]) / 1000.0)**2

    if abs(T_vals[0] - T_vals[1]) > 5000: return -np.inf, -np.inf
    if abs(T_vals[1] - T_vals[2]) > 5000: return -np.inf, -np.inf
    if abs(u_vals[0] - u_vals[1]) > 5: return -np.inf, -np.inf
    if abs(u_vals[1] - u_vals[2]) > 5: return -np.inf, -np.inf

    ln_like = 0.0
    for i, z_str in enumerate(Z_ORDER):
        zrei, ha, hs, tau = z_params[4*i : 4*i+4]
        pred = predict_pk(m, f, zrei, ha, hs, tau, z_str)

        if case_name != "none":
            corr = G_CORRECTORS[case_name].get_total_correction(z_str)
            pred *= corr

        diff = pred - G_OBS[z_str]["Pk"]
        ln_like += -0.5 * float(diff @ G_OBS[z_str]["Cov_inv"] @ diff)

    return ln_prior_T + ln_like, ln_like

def transform_to_derived(chain, fixed_f):

    n_samples = len(chain)
    out_dim = 13 if fixed_f else 14
    derived = np.zeros((n_samples, out_dim), dtype=np.float64)

    derived[:, 0] = chain[:, 0] # m
    if not fixed_f: derived[:, 1] = chain[:, 1] # f
    
    z_offset_in = 1 if fixed_f else 2
    z_offset_out = 1 if fixed_f else 2

    for i, z_str in enumerate(Z_ORDER):
        off_in = z_offset_in + 4 * i
        off_out = z_offset_out + 4 * i
        
        zrei, ha, hs, tau = chain[:, off_in], chain[:, off_in+1], chain[:, off_in+2], chain[:, off_in+3]
        x_in = np.column_stack([np.full(n_samples, Z_FLOAT[z_str]), zrei, ha, hs])
        tgu = np.asarray(G_TGU.interp(G_TGU.scaler.transform(x_in)), dtype=np.float64)

        derived[:, off_out]   = tgu[:, 0] / 1e4   
        derived[:, off_out+1] = tgu[:, 1]         
        derived[:, off_out+2] = tgu[:, 2]         
        derived[:, off_out+3] = tau               

    return derived

def plot_corner(chain, labels, out_pdf):
    c = ChainConsumer()
    c.add_chain(chain=chain, parameters=labels, name="Real data", shade=True, shade_alpha=0.5)
    c.configure(summary=False, max_ticks=3, label_font_size=12, tick_font_size=10, usetex=False)
    fig = c.plotter.plot(parameters=labels)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_mcmc_case(case_name, fixed_f, init_args):
    cfg_name = f"{case_name}_" + ("fixed_f" if fixed_f else "free_f")
    print(f"\n{'='*60}\n[*] Running MCMC: {cfg_name}\n{'='*60}")

    dim = 13 if fixed_f else 14
    bounds = BOUNDS_FIXED if fixed_f else BOUNDS_FULL

    pos = []
    while len(pos) < N_WALKERS:
        p = np.random.uniform(bounds[:, 0], bounds[:, 1])
        lp, _ = log_probability(p, case_name, fixed_f) 
        if lp > -np.inf:
            pos.append(p)
    pos = np.asarray(pos, dtype=np.float64)

    with Pool(processes=N_PROC, initializer=init_worker, initargs=init_args) as pool:
        sampler = emcee.EnsembleSampler(N_WALKERS, dim, log_probability, args=(case_name, fixed_f), pool=pool)
        sampler.run_mcmc(pos, MCMC_STEPS, progress=True)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        burnin = int(min(20 * np.max(tau), MCMC_STEPS * 0.5)) if not np.isnan(tau).all() else int(MCMC_STEPS * 0.5)
    except emcee.autocorr.AutocorrError:
        burnin = int(MCMC_STEPS * 0.5)
        
    print(f"    Burn-in discarded: {burnin} steps.")
    flat_samples = sampler.get_chain(discard=burnin, thin=10, flat=True)
    lnprob = sampler.get_log_prob(discard=burnin, thin=10, flat=True)
    lnlikes = sampler.get_blobs(discard=burnin, thin=10, flat=True) 
    derived_samples = transform_to_derived(flat_samples, fixed_f)

    best_idx = np.argmax(lnlikes)
    max_like_params = flat_samples[best_idx]
    max_like_chi2 = -2.0 * lnlikes[best_idx]
    
    max_like_emu_preds = {}
    m = max_like_params[0]
    f = 1.0 if fixed_f else max_like_params[1]
    z_params = max_like_params[1:] if fixed_f else max_like_params[2:]
    
    for i, z_str in enumerate(Z_ORDER):
        zrei, ha, hs, tau = z_params[4*i : 4*i+4]
        pred = predict_pk(m, f, zrei, ha, hs, tau, z_str)
        if case_name != "none":
            corr = G_CORRECTORS[case_name].get_total_correction(z_str)
            pred *= corr
        max_like_emu_preds[z_str] = pred

    res_file = os.path.join(OUT_RESULTS_DIR, f"mcmc_{cfg_name}.pkl")
    with open(res_file, "wb") as f_out:
        pickle.dump({
            "raw": flat_samples, 
            "derived": derived_samples, 
            "burnin": burnin, 
            "lnprob": lnprob,
            "lnlike": lnlikes,
            "max_like_params": max_like_params,
            "max_like_chi2": max_like_chi2,
            "max_like_emu_preds": max_like_emu_preds
        }, f_out)
        
    labels_base = [
        r"$T_0^{5.0}~[10^4\,\mathrm{K}]$", r"$\gamma^{5.0}$", r"$u_0^{5.0}~[\mathrm{eV}\,m_\mathrm{p}^{-1}]$", r"$\tau_0^{5.0}$",
        r"$T_0^{4.6}~[10^4\,\mathrm{K}]$", r"$\gamma^{4.6}$", r"$u_0^{4.6}~[\mathrm{eV}\,m_\mathrm{p}^{-1}]$", r"$\tau_0^{4.6}$",
        r"$T_0^{4.2}~[10^4\,\mathrm{K}]$", r"$\gamma^{4.2}$", r"$u_0^{4.2}~[\mathrm{eV}\,m_\mathrm{p}^{-1}]$", r"$\tau_0^{4.2}$"
    ]
    labels = [r'$\log_{10} (m_{\mathrm{FDM}}~[\mathrm{eV}])$'] + ([] if fixed_f else [r"$f_{\rm FDM}$"]) + labels_base

    plot_file = os.path.join(OUT_PLOTS_DIR, f"corner_{cfg_name}.pdf")
    plot_corner(derived_samples, labels, plot_file)
    print(f"    Saved results -> {res_file}\n    Saved plot -> {plot_file}")

def main():
    global G_NN, G_TGU, G_OBS, G_CORRECTORS
    nn_pack, tgu_pack, obs_pack, corr_pack = load_system()
    G_NN = nn_pack
    G_TGU = tgu_pack
    G_OBS = obs_pack
    G_CORRECTORS = corr_pack
    init_args = (nn_pack, tgu_pack, obs_pack, corr_pack)

    cases = ["none", "no_patchy", "with_patchy"]
    
    for case in cases:
        for fixed_f in [False, True]:
            run_mcmc_case(case, fixed_f, init_args)

if __name__ == "__main__":
    main()
