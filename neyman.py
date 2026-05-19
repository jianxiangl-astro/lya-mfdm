import os
import sys
import time
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RBFInterpolator, CubicSpline
from scipy.optimize import minimize, brentq
from concurrent.futures import as_completed
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})


BASE_SIM_DIR = "data"
OBS_FILE = os.path.join(BASE_SIM_DIR, "lya_data.pkl")

EMU_ROOT = "emu"
NN_EMU_DIR = os.path.join(EMU_ROOT, "emu_N100")
TGU_EMU_FILE = os.path.join(EMU_ROOT, "tgu_interpolator.pkl")
CORR_WITH_PATCHY_FILE = os.path.join(EMU_ROOT, "lya_corrector_with_patchy.pkl")

OUT_RESULTS_DIR = "results"
OUT_PLOTS_DIR = "plots"
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

SEED_MCMC_FILE = "results/mcmc_with_patchy_fixed_f.pkl"
CASE_NAME = "with_patchy"

CONFIDENCE_LEVEL = 0.95449974
RANDOM_SEED = 20260429

M_MIN, M_MAX = -23.0, -19.0
N_M_GRID = 40
M_GRID = np.linspace(M_MIN, M_MAX, N_M_GRID)

N_TOYS = 448

N_SEED_STARTS = 80

N_RANDOM_STARTS_GLOBAL = 250
N_LOCAL_STARTS_GLOBAL = 14
MAXITER_GLOBAL = 1200

N_RANDOM_STARTS_COND = 180
N_LOCAL_STARTS_COND = 10
MAXITER_COND = 900

TOY_N_RANDOM_STARTS_GLOBAL = 80
TOY_N_LOCAL_STARTS_GLOBAL = 6
TOY_MAXITER_GLOBAL = 500

TOY_N_RANDOM_STARTS_COND = 60
TOY_N_LOCAL_STARTS_COND = 5
TOY_MAXITER_COND = 400

LOCAL_METHOD = "Powell"
BAD_CHI2 = 1.0e100

LOGK_LABELS = [f"{x:.1f}" for x in np.round(np.arange(-2.2, -0.65, 0.1), 1)]
KEEP_K_MASK = np.ones(len(LOGK_LABELS), dtype=bool)

USE_T0_GAUSSIAN_CONSTRAINT = True
T0_PRIOR_MEANS = {"5.0": 9286.5, "4.6": 8986.5, "4.2": 9155.5}
T0_PRIOR_SIGMA = 1000.0
DRAW_T0_TOYS = True

N_FOLDS = 5
HIDDEN_DIM = 256

Z_ORDER = ["5.0", "4.6", "4.2"]
Z_FLOAT = {"5.0": 5.0, "4.6": 4.6, "4.2": 4.2}

BOUNDS_BASE = [
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],
]
BOUNDS_FIXED = np.array([[M_MIN, M_MAX]] + BOUNDS_BASE, dtype=np.float64)
BOUNDS_NUIS = BOUNDS_FIXED[1:].copy()

G_NN = None
G_TGU = None
G_OBS = None
G_CORRECTORS = None
G_SEED_BANK = None


@dataclass
class GlobalTGUInterpolator:
    scaler: Any
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
        return (
            1.0
            + a_auto * np.exp(self.target_k / k_auto)
            + a_cross * np.exp(self.target_k / k_cross) * np.cos(2271.0 * self.target_k)
        )

    def get_total_correction(self, z_str: str) -> np.ndarray:
        total_corr = self.pixel_multiplier.copy() * self.get_siiii_correction(z_str)
        if self.apply_patchy:
            total_corr *= self.patchy_dict[z_str]
        return total_corr


sys.modules.setdefault("hahsz_to_Tgu", sys.modules[__name__])
sys.modules.setdefault("interpolator", sys.modules[__name__])


def load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_mlp(in_dim: int, out_dim: int = 16) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, out_dim),
    )


def load_system():
    nn_pack = {
        "scaler_cdm": load_pkl(os.path.join(NN_EMU_DIR, "scaler_cdm.pkl")),
        "scaler_res": load_pkl(os.path.join(NN_EMU_DIR, "scaler_res.pkl")),
        "cdm_models": [],
        "res_models": [],
    }

    for fold in range(N_FOLDS):
        c_mod = build_mlp(5, 16)
        r_mod = build_mlp(7, 16)

        c_mod.load_state_dict(torch.load(os.path.join(NN_EMU_DIR, f"model_cdm_fold{fold}.pth"), map_location="cpu"))
        r_mod.load_state_dict(torch.load(os.path.join(NN_EMU_DIR, f"model_res_fold{fold}.pth"), map_location="cpu"))

        c_mod.eval()
        r_mod.eval()
        nn_pack["cdm_models"].append(c_mod)
        nn_pack["res_models"].append(r_mod)

    obs_raw = load_pkl(OBS_FILE)
    obs_pack = {}
    for z in Z_ORDER:
        cov = np.asarray(obs_raw[z]["Cov"], dtype=np.float64)[np.ix_(KEEP_K_MASK, KEEP_K_MASK)]
        cov = 0.5 * (cov + cov.T)
        jitter = 1.0e-12 * max(1.0, float(np.max(np.diag(cov))))

        obs_pack[z] = {
            "Pk": np.asarray(obs_raw[z]["Pk"], dtype=np.float64)[KEEP_K_MASK],
            "Cov": cov,
            "Cov_inv": np.linalg.inv(cov),
            "Chol": np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0])),
        }

    return nn_pack, load_pkl(TGU_EMU_FILE), obs_pack, {CASE_NAME: load_pkl(CORR_WITH_PATCHY_FILE)}


def load_seed_bank() -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(SEED_MCMC_FILE):
        print(f"[Warning] Seed MCMC file not found at {SEED_MCMC_FILE}. Random starts only.")
        return None

    obj = load_pkl(SEED_MCMC_FILE)
    raw = np.asarray(obj["raw"], dtype=np.float64)
    lnprob = np.asarray(obj.get("lnprob", np.zeros(len(raw))), dtype=np.float64)

    good = np.all(np.isfinite(raw), axis=1) & np.isfinite(lnprob)
    return {"raw": raw[good], "lnprob": lnprob[good]}


def predict_pk(theta: np.ndarray, z_str: str) -> np.ndarray:
    z_obs = Z_FLOAT[z_str]
    i = Z_ORDER.index(z_str)
    zrei, ha, hs, tau = theta[1 + 4 * i: 5 + 4 * i]

    x_cdm = G_NN["scaler_cdm"].transform(np.array([[z_obs, zrei, ha, hs, tau]], dtype=np.float64))
    x_res = G_NN["scaler_res"].transform(np.array([[z_obs, theta[0], 1.0, zrei, ha, hs, tau]], dtype=np.float64))

    x_cdm_t = torch.tensor(x_cdm, dtype=torch.float32)
    x_res_t = torch.tensor(x_res, dtype=torch.float32)

    with torch.no_grad():
        cdm = torch.stack([mod(x_cdm_t)[0] for mod in G_NN["cdm_models"]]).mean(0).numpy()
        res = torch.stack([mod(x_res_t)[0] for mod in G_NN["res_models"]]).mean(0).numpy()

    pk = np.power(10.0, cdm + res)
    pk *= G_CORRECTORS[CASE_NAME].get_total_correction(z_str)

    return pk[KEEP_K_MASK]


def extract_tgu(theta: np.ndarray) -> Optional[np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)

    if theta.shape != (13,):
        return None
    if not (np.all(theta >= BOUNDS_FIXED[:, 0]) and np.all(theta <= BOUNDS_FIXED[:, 1])):
        return None

    try:
        out = []
        for i, z in enumerate(Z_ORDER):
            x = np.array([[Z_FLOAT[z], theta[1 + 4 * i], theta[2 + 4 * i], theta[3 + 4 * i]]], dtype=np.float64)
            xs = G_TGU.scaler.transform(x)
            out.append(G_TGU.interp(xs)[0])
        return np.asarray(out, dtype=np.float64)
    except Exception:
        return None


def get_domain_penalty(theta: np.ndarray) -> float:
    tgu = extract_tgu(theta)
    if tgu is None:
        return BAD_CHI2 / 10.0

    T = tgu[:, 0]
    u = tgu[:, 2]
    p = 0.0

    for i in [0, 1]:
        dT = abs(T[i] - T[i + 1])
        du = abs(u[i] - u[i + 1])

        if dT > 5000.0:
            p += 1.0e4 * ((dT - 5000.0) / 1000.0) ** 2
        if du > 5.0:
            p += 1.0e5 * (du - 5.0) ** 2

    return p


def chi2_total(theta: np.ndarray, data_pack: Dict[str, Dict[str, np.ndarray]], t0_means: Dict[str, float]) -> float:
    penalty = get_domain_penalty(theta)
    if penalty >= BAD_CHI2 / 100.0:
        return BAD_CHI2

    chi2 = penalty

    for z in Z_ORDER:
        diff = predict_pk(theta, z) - data_pack[z]["Pk"]
        chi2 += float(diff @ data_pack[z]["Cov_inv"] @ diff)

    if USE_T0_GAUSSIAN_CONSTRAINT:
        tgu = extract_tgu(theta)
        if tgu is None:
            return BAD_CHI2
        for i, z in enumerate(Z_ORDER):
            chi2 += ((tgu[i, 0] - t0_means[z]) / T0_PRIOR_SIGMA) ** 2

    return float(chi2) if np.isfinite(chi2) else BAD_CHI2


def pack_theta(x: np.ndarray, fixed_m: Optional[float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if fixed_m is None:
        return x
    return np.concatenate([[float(fixed_m)], x])


def prepare_starts(
    fixed_m: Optional[float],
    extra_thetas: Optional[List[np.ndarray]],
    rng: np.random.Generator,
    n_random: int,
) -> List[np.ndarray]:
    starts = []

    for th in extra_thetas or []:
        th = np.asarray(th, dtype=np.float64).copy()
        if fixed_m is not None:
            th[0] = float(fixed_m)
        if get_domain_penalty(th) == 0.0:
            starts.append(th)

    if G_SEED_BANK is not None:
        if fixed_m is None:
            order = np.argsort(G_SEED_BANK["lnprob"])[::-1]
        else:
            order = np.argsort(np.abs(G_SEED_BANK["raw"][:, 0] - float(fixed_m)))

        for ind in order[:N_SEED_STARTS]:
            th = G_SEED_BANK["raw"][ind].copy()
            if fixed_m is not None:
                th[0] = float(fixed_m)
            if get_domain_penalty(th) == 0.0:
                starts.append(th)

    bnds = BOUNDS_FIXED if fixed_m is None else BOUNDS_NUIS

    for _ in range(n_random):
        x = rng.uniform(bnds[:, 0], bnds[:, 1])
        th = pack_theta(x, fixed_m)
        if get_domain_penalty(th) == 0.0:
            starts.append(th)

    return starts


def minimize_total_chi2(
    data_pack: Dict,
    t0_means: Dict,
    fixed_m: Optional[float] = None,
    extra_thetas: Optional[List[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    toy_mode: bool = False,
) -> Tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng(RANDOM_SEED)

    if fixed_m is None:
        bnds = BOUNDS_FIXED
        n_random = TOY_N_RANDOM_STARTS_GLOBAL if toy_mode else N_RANDOM_STARTS_GLOBAL
        n_local = TOY_N_LOCAL_STARTS_GLOBAL if toy_mode else N_LOCAL_STARTS_GLOBAL
        maxiter = TOY_MAXITER_GLOBAL if toy_mode else MAXITER_GLOBAL
    else:
        bnds = BOUNDS_NUIS
        n_random = TOY_N_RANDOM_STARTS_COND if toy_mode else N_RANDOM_STARTS_COND
        n_local = TOY_N_LOCAL_STARTS_COND if toy_mode else N_LOCAL_STARTS_COND
        maxiter = TOY_MAXITER_COND if toy_mode else MAXITER_COND

    starts = prepare_starts(fixed_m, extra_thetas, rng, n_random)

    scored = []
    for th in starts:
        val = chi2_total(th, data_pack, t0_means)
        if val < BAD_CHI2 / 10.0:
            scored.append((val, th))

    if len(scored) == 0:
        raise RuntimeError("No valid optimization start was found. Domain may be too strict.")

    scored = sorted(scored, key=lambda item: item[0])[:n_local]
    best_val = float(scored[0][0])
    best_theta = scored[0][1].copy()

    def obj(x):
        th = pack_theta(x, fixed_m)
        return chi2_total(th, data_pack, t0_means)

    for _, th0 in scored:
        x0 = th0 if fixed_m is None else th0[1:]

        res = minimize(
            obj,
            x0,
            method=LOCAL_METHOD,
            bounds=bnds,
            options={"maxiter": maxiter, "xtol": 1.0e-4, "ftol": 1.0e-4, "disp": False},
        )

        th = pack_theta(res.x, fixed_m)
        if res.fun < best_val and get_domain_penalty(th) == 0.0:
            best_val = float(res.fun)
            best_theta = th.copy()

    return best_theta, best_val


def build_toy_data(mean_pk: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    toy_data = {}

    for z in Z_ORDER:
        noise = G_OBS[z]["Chol"] @ rng.normal(size=G_OBS[z]["Pk"].shape[0])
        toy_data[z] = {
            "Pk": mean_pk[z] + noise,
            "Cov": G_OBS[z]["Cov"],
            "Cov_inv": G_OBS[z]["Cov_inv"],
            "Chol": G_OBS[z]["Chol"],
        }

    return toy_data


def toy_worker(args):
    global G_NN, G_TGU, G_OBS, G_CORRECTORS, G_SEED_BANK
    
    if G_NN is None:
        G_NN, G_TGU, G_OBS, G_CORRECTORS = load_system()
        G_SEED_BANK = load_seed_bank()

    m0, toy_idx, mean_pk, t0_true, seed, th_g_obs, th_c_obs = args
    rng = np.random.default_rng(seed)

    try:
        toy_data = build_toy_data(mean_pk, rng)

        if DRAW_T0_TOYS:
            t0_toy = {z: t0_true[z] + rng.normal(scale=T0_PRIOR_SIGMA) for z in Z_ORDER}
        else:
            t0_toy = dict(T0_PRIOR_MEANS)

        th_g, chi2_g = minimize_total_chi2(
            toy_data,
            t0_toy,
            fixed_m=None,
            extra_thetas=[th_c_obs, th_g_obs],
            rng=rng,
            toy_mode=True,
        )

        th_c, chi2_c = minimize_total_chi2(
            toy_data,
            t0_toy,
            fixed_m=float(m0),
            extra_thetas=[th_c_obs, th_g, th_g_obs],
            rng=rng,
            toy_mode=True,
        )
        
        if chi2_c < chi2_g:
            chi2_g = chi2_c
            
        t_val = float(chi2_c - chi2_g)
        return {"ok": True, "toy_index": toy_idx, "t": t_val}

    except Exception as exc:
        return {"ok": False, "toy_index": toy_idx, "t": np.nan, "err": repr(exc)}


def find_accepted_intervals(m_grid: np.ndarray, t_obs: np.ndarray, q95: np.ndarray) -> List[Tuple[float, float]]:
    diff = np.asarray(q95, dtype=float) - np.asarray(t_obs, dtype=float)
    good = np.isfinite(diff)

    if good.sum() < 4:
        return []

    mg = m_grid[good]
    dg = diff[good]

    cs = CubicSpline(mg, dg)
    fine = np.linspace(mg[0], mg[-1], 1200)
    val = cs(fine)

    roots = []
    for i in np.where(np.diff(np.signbit(val)))[0]:
        try:
            roots.append(brentq(cs, fine[i], fine[i + 1]))
        except ValueError:
            pass

    raw_intervals = []
    start = mg[0] if dg[0] >= 0.0 else None

    for r in roots:
        if start is None:
            start = r
        else:
            raw_intervals.append((float(start), float(r)))
            start = None

    if start is not None and dg[-1] >= 0.0:
        raw_intervals.append((float(start), float(mg[-1])))

    if not raw_intervals:
        return []

    grid_step = np.median(np.diff(mg))
    
    gap_tolerance = 2.0 * grid_step 
    
    merged_intervals = [raw_intervals[0]]
    for current in raw_intervals[1:]:
        prev = merged_intervals[-1]
        gap_width = current[0] - prev[1]
        
        if gap_width <= gap_tolerance:
            merged_intervals[-1] = (prev[0], current[1])
        else:
            merged_intervals.append(current)

    return merged_intervals


def plot_and_save(results: Dict, prof_pdf: str, info_pdf: str):
    m = results["m_grid"]
    t_obs = results["t_obs"]
    q95 = results["q95"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 1.5,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.set_yscale("symlog", linthresh=5.0)

    color_obs = "#2c3e50"     
    color_critical = "#c0392b" 
    color_wilks = "0.5"        

    ax.plot(m, t_obs, marker="o", markersize=6, ls="-", color=color_obs, lw=2.0, label=r"Observed $t_{m_{\mathrm{FDM}}}$")
    ax.plot(m, q95, marker="s", markersize=5, ls="--", color=color_critical, lw=1.8, label=r"Neyman 95% threshold")
    

    for a, b in results["accepted_intervals"]:
        ax.axvspan(a, b, color="#bdc3c7", alpha=0.3, lw=0, zorder=-1)

    ax.set_xlabel(r'$\log_{10} (m_{\mathrm{FDM}}~[\mathrm{eV}])$')
    ax.set_ylabel(r"$t_{m_{\mathrm{FDM}}}$")
    
    ax.grid(False) 
    ax.minorticks_on()
    
    ax.legend(loc="upper right")

    fig.savefig(prof_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    with PdfPages(info_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.axis("off")

        lines = [
            "Neyman Construction Results (with_patchy)",
            "-" * 44,
            "Accepted 95% confidence intervals:",
        ]

        if results["accepted_intervals"]:
            for a, b in results["accepted_intervals"]:
                lines.append(f"  {a:.4f} < log10(m_FDM eV) < {b:.4f}")
        else:
            lines.append("  No accepted intervals.")

        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            family="monospace",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    global G_NN, G_TGU, G_OBS, G_CORRECTORS, G_SEED_BANK

    print(f"[*] Loading system: {CASE_NAME}")
    print(f"[*] Seed MCMC file: {SEED_MCMC_FILE}")

    G_NN, G_TGU, G_OBS, G_CORRECTORS = load_system()
    G_SEED_BANK = load_seed_bank()

    rng = np.random.default_rng(RANDOM_SEED)

    print("\n[*] Initial Global Optimization...")
    t0 = time.perf_counter()
    th_g_obs_initial, chi2_g_obs_initial = minimize_total_chi2(G_OBS, T0_PRIOR_MEANS, rng=rng, toy_mode=False)
    print(f"    Initial Global chi2 = {chi2_g_obs_initial:.6f} | Initial Global m = {th_g_obs_initial[0]:.6f}")
    print(f"    Time = {time.perf_counter() - t0:.1f} s")

    m_grid = np.asarray(M_GRID, dtype=float)
    
    print("\n[*] Pass 1: Scanning conditional likelihoods to lock in true global minimum...")
    true_global_chi2 = chi2_g_obs_initial
    true_global_th = th_g_obs_initial.copy()
    
    cached_cond_results = {}
    
    for im, m0 in enumerate(m_grid):
        print(f"    Scanning mass point {im + 1}/{len(m_grid)}: m0 = {m0:.6f}...", end="\r", flush=True)
        th_c_obs, chi2_c_obs = minimize_total_chi2(
            G_OBS,
            T0_PRIOR_MEANS,
            fixed_m=float(m0),
            extra_thetas=[true_global_th],
            rng=np.random.default_rng(RANDOM_SEED + im),
            toy_mode=False,
        )
        
        cached_cond_results[m0] = {"th": th_c_obs, "chi2": chi2_c_obs}
        
        if chi2_c_obs < true_global_chi2:
            true_global_chi2 = chi2_c_obs
            true_global_th = th_c_obs.copy()
            
    print(f"\n    [+] Pass 1 Complete. True Global chi2 = {true_global_chi2:.6f} at m = {true_global_th[0]:.6f}")

    t_obs_list = []
    q95_list = []
    n_ok_list = []

    res_path = os.path.join(OUT_RESULTS_DIR, "neyman_pure_fdm_with_patchy_fast_results.pkl")
    partial_path = res_path.replace(".pkl", "_partial.pkl")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    print(f"\n[*] Pass 2: Starting MPI pool with {size} workers for Toys...")
    print(
        f"[*] Toy settings: N_TOYS={N_TOYS}, "
        f"global starts={TOY_N_RANDOM_STARTS_GLOBAL}/{TOY_N_LOCAL_STARTS_GLOBAL}, "
        f"conditional starts={TOY_N_RANDOM_STARTS_COND}/{TOY_N_LOCAL_STARTS_COND}"
    )

    with MPIPoolExecutor() as pool:
        for im, m0 in enumerate(m_grid):
            mt0 = time.perf_counter()
            print(f"\n[*] Computing Point {im + 1}/{len(m_grid)}: m0 = {m0:.6f}", flush=True)

            th_c_obs = cached_cond_results[m0]["th"]
            chi2_c_obs = cached_cond_results[m0]["chi2"]

            t_obs = float(chi2_c_obs - true_global_chi2)
            t_obs_list.append(t_obs)
            print(f"    Cond chi2 = {chi2_c_obs:.6f} | t_m = {t_obs:.6f}", flush=True)

            mean_pk = {z: predict_pk(th_c_obs, z) for z in Z_ORDER}

            tgu_true = extract_tgu(th_c_obs)
            if tgu_true is None:
                t0_true = dict(T0_PRIOR_MEANS)
            else:
                t0_true = {z: float(tgu_true[i, 0]) for i, z in enumerate(Z_ORDER)}

            seeds = rng.integers(1, 2**31 - 1, size=N_TOYS)
            
            tasks = [
                (float(m0), j, mean_pk, t0_true, int(seeds[j]), true_global_th.copy(), th_c_obs.copy())
                for j in range(N_TOYS)
            ]

            toy_recs = []
            
            futures = [pool.submit(toy_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=N_TOYS, desc=f"toys m={m0:.3f}", dynamic_ncols=True):
                toy_recs.append(future.result())

            toy_t = np.array([r["t"] for r in toy_recs if r.get("ok", False) and np.isfinite(r.get("t", np.nan))], dtype=float)

            q95 = float(np.percentile(toy_t, 100.0 * CONFIDENCE_LEVEL)) if len(toy_t) > 0 else np.nan
            q95_list.append(q95)
            n_ok_list.append(len(toy_t))

            print(f"    Toys = {len(toy_t)}/{N_TOYS} | q95 = {q95:.6f} | accepted = {t_obs <= q95}", flush=True)
            print(f"    Mass-point time = {time.perf_counter() - mt0:.1f} s", flush=True)


    results = {
        "m_grid": m_grid,
        "t_obs": np.asarray(t_obs_list),
        "q95": np.asarray(q95_list),
        "n_ok": np.asarray(n_ok_list),
    }
    results["accepted_intervals"] = find_accepted_intervals(results["m_grid"], results["t_obs"], results["q95"])

    prof_pdf = os.path.join(OUT_PLOTS_DIR, "neyman_pure_fdm_with_patchy_profile_fast.pdf")
    info_pdf = os.path.join(OUT_PLOTS_DIR, "neyman_pure_fdm_with_patchy_95CL_info_fast.pdf")
    plot_and_save(results, prof_pdf, info_pdf)

    print("\n" + "=" * 60)
    print("[*] Done.")
    print(f"    Profile PDF: {prof_pdf}")
    print(f"    Info PDF   : {info_pdf}")
    print("=" * 60)


if __name__ == "__main__":
    main()