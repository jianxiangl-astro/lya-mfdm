"""
Mock data recovery test for the Lyman-alpha forest 1D flux power spectrum emulator and inference pipeline.

"""
import os
import sys
import pickle
import random
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
import emcee
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from chainconsumer import ChainConsumer
from scipy.interpolate import RBFInterpolator  
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False
})

BASE_SIM_DIR = "data"
OBS_FILE = "data/lya_data.pkl"

EMU_ROOT = "emu"
NN_EMU_DIR = os.path.join(EMU_ROOT, "emu_N100")
TGU_EMU_FILE = os.path.join(EMU_ROOT, "tgu_interpolator.pkl")

OUT_DIR = "plots/mock"
os.makedirs(OUT_DIR, exist_ok=True)

MOCK_INDICES = list(range(200, 210))
MOCK_SEED_BASE = 123412324
ADD_NOISE = False         

MCMC_STEPS = 400000
N_WALKERS = 56
N_PROC = min(56, os.cpu_count() or 1)
BURNIN_FRAC = 0.5
THIN = 10
N_FOLDS = 5
HIDDEN_DIM = 256
F_EPS = 1e-8


Z_ORDER = ["5.0", "4.6", "4.2"]
Z_FLOAT = {"5.0": 5.0, "4.6": 4.6, "4.2": 4.2}
FILE_NUM_MAP = {"5.0": 0, "4.6": 1, "4.2": 2}
LOGK_LABELS = [f"{x:.1f}" for x in np.round(np.arange(-2.2, -0.65, 0.1), 1)]

BOUNDS = np.array([
    [-23.0, -19.0],  
    [0.0, 1.0],      
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],  # z=5.0
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],  # z=4.6
    [6.0, 15.0], [0.05, 4.0], [-1.0, 1.0], [0.3, 1.8],  # z=4.2
], dtype=np.float64)

DERIVED_LABELS = [
    r'$\log_{10} (m_{\mathrm{FDM}}~[\mathrm{eV}])$',
    r"$f_{\rm FDM}$",
        r"$T_0^{5.0}~[10^4 \, \mathrm{K}]$",
        r"$\gamma^{5.0}$",
        r"$u_0^{5.0}~[\mathrm{eV}\,m_\mathrm{p}^{-1}]$",
        r"$\tau_0^{5.0}$",
        r"$T_0^{4.6}~[10^4 \, \mathrm{K}]$",
        r"$\gamma^{4.6}$",
        r"$u_0^{4.6}~[\mathrm{eV}\,m_\mathrm{p}^{-1}]$",
        r"$\tau_0^{4.6}$",
        r"$T_0^{4.2}~[10^4 \, \mathrm{K}]$",
        r"$\gamma^{4.2}$",
        r"$u_0^{4.2}~[\mathrm{{eV}}\,m_\mathrm{p}^{-1}]$",
        r"$\tau_0^{4.2}$",
]

G_NN = None
G_TGU = None
G_OBS = None

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@dataclass
class GlobalTGUInterpolator:
    scaler: StandardScaler
    interp: RBFInterpolator

sys.modules.setdefault("hahsz_to_Tgu", sys.modules[__name__])


def build_mlp(in_dim, out_dim=16):
    return nn.Sequential(
        nn.Linear(in_dim, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, out_dim),
    )

def load_nn_emulator():
    print(f"[*] Loading NN emulator from: {NN_EMU_DIR}")

    nn_pack = {
        "scaler_cdm": load_pkl(os.path.join(NN_EMU_DIR, "scaler_cdm.pkl")),
        "scaler_res": load_pkl(os.path.join(NN_EMU_DIR, "scaler_res.pkl")),
        "cdm_models": [],
        "res_models": [],
    }

    for fold in range(N_FOLDS):
        cdm_model = build_mlp(in_dim=5, out_dim=16)
        cdm_path = os.path.join(NN_EMU_DIR, f"model_cdm_fold{fold}.pth")
        cdm_model.load_state_dict(torch.load(cdm_path, map_location="cpu"))
        cdm_model.eval()
        nn_pack["cdm_models"].append(cdm_model)

        res_model = build_mlp(in_dim=7, out_dim=16)
        res_path = os.path.join(NN_EMU_DIR, f"model_res_fold{fold}.pth")
        res_model.load_state_dict(torch.load(res_path, map_location="cpu"))
        res_model.eval()
        nn_pack["res_models"].append(res_model)

    print(f"    Loaded {len(nn_pack['cdm_models'])} CDM folds and {len(nn_pack['res_models'])} residual folds.")
    return nn_pack


def load_tgu_emulator():
    print(f"[*] Loading TGU emulator from: {TGU_EMU_FILE}")
    return load_pkl(TGU_EMU_FILE)


def predict_tgu(z_str, zrei, ha, hs):
    x = np.array([[Z_FLOAT[z_str], zrei, ha, hs]], dtype=np.float64)
    x_scaled = G_TGU.scaler.transform(x)
    return np.asarray(G_TGU.interp(x_scaled)[0], dtype=np.float64)


def predict_pk(m, f, zrei, ha, hs, taueff, z_str):

    z_obs = Z_FLOAT[z_str]

    x_cdm = np.array([[z_obs, zrei, ha, hs, taueff]], dtype=np.float64)
    x_res = np.array([[z_obs, m, f, zrei, ha, hs, taueff]], dtype=np.float64)

    x_cdm = G_NN["scaler_cdm"].transform(x_cdm)
    x_res = G_NN["scaler_res"].transform(x_res)

    x_cdm_t = torch.tensor(x_cdm, dtype=torch.float32)
    x_res_t = torch.tensor(x_res, dtype=torch.float32)

    with torch.no_grad():
        cdm_stack = torch.stack([model(x_cdm_t)[0] for model in G_NN["cdm_models"]])
        res_stack = torch.stack([model(x_res_t)[0] for model in G_NN["res_models"]])

    logp_cdm = cdm_stack.mean(dim=0).cpu().numpy()
    g_res = res_stack.mean(dim=0).cpu().numpy()

    if f <= F_EPS:
        logp = logp_cdm
    else:
        logp = logp_cdm + f * g_res

    return np.power(10.0, logp)

def log_prior(theta):
    theta = np.asarray(theta, dtype=np.float64)

    if theta.shape != (14,):
        return -np.inf
    if np.any(theta < BOUNDS[:, 0]) or np.any(theta > BOUNDS[:, 1]):
        return -np.inf

    return 0.0


def log_likelihood(theta):
    m, f = float(theta[0]), float(theta[1])
    lnlike = 0.0

    for i, z_str in enumerate(Z_ORDER):
        off = 2 + 4 * i
        zrei, ha, hs, taueff = theta[off:off + 4]

        pred = predict_pk(m, f, zrei, ha, hs, taueff, z_str)
        obs = G_OBS[z_str]["Pk"]
        cov_inv = G_OBS[z_str]["Cov_inv"]

        diff = pred - obs
        lnlike += -0.5 * float(diff @ cov_inv @ diff)

    return lnlike


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def init_worker(nn_pack, tgu_pack, obs_pack):
    global G_NN, G_TGU, G_OBS
    torch.set_num_threads(1)
    G_NN = nn_pack
    G_TGU = tgu_pack
    G_OBS = obs_pack


def load_observation_covariance():
    obs_raw = load_pkl(OBS_FILE)
    obs_cov = {}
    for z_str in Z_ORDER:
        cov = np.asarray(obs_raw[z_str]["Cov"], dtype=np.float64)
        if "Cov-1" in obs_raw[z_str]:
            cov_inv = np.asarray(obs_raw[z_str]["Cov-1"], dtype=np.float64)
        else:
            cov_inv = np.linalg.inv(cov)
        obs_cov[z_str] = {"Cov": cov, "Cov_inv": cov_inv}
    return obs_cov


def build_mock_data(mock_idx, seed):

    rng = np.random.default_rng(seed)
    obs_cov = load_observation_covariance()
    param_all = load_pkl(os.path.join(BASE_SIM_DIR, "param.pkl"))

    truth_row = param_all[param_all["index"] == mock_idx][0]
    truth = np.zeros(14, dtype=np.float64)
    truth[0] = float(truth_row["m"])
    truth[1] = float(truth_row["f"])

    mock_obs = {}
    row_id = None

    for i, z_str in enumerate(Z_ORDER):
        num = FILE_NUM_MAP[z_str]
        data = load_pkl(os.path.join(BASE_SIM_DIR, f"all_pk_num{num}.pkl"))
        rows = np.sort(data[data["index"] == mock_idx], order="taueff")

        if len(rows) == 0:
            raise RuntimeError(f"No rows found for mock_idx={mock_idx}, z={z_str}")

        if row_id is None:
            row_id = int(rng.integers(0, len(rows)))
        picked = rows[row_id]

        pk_true = np.array([picked[k] for k in LOGK_LABELS], dtype=np.float64)
        cov = obs_cov[z_str]["Cov"]
        cov_inv = obs_cov[z_str]["Cov_inv"]

        if ADD_NOISE:
            pk_mock = rng.multivariate_normal(pk_true, cov)
        else:
            pk_mock = pk_true.copy()

        mock_obs[z_str] = {"Pk": pk_mock, "Cov": cov, "Cov_inv": cov_inv}

        off = 2 + 4 * i
        truth[off + 0] = float(picked["T"])
        truth[off + 1] = float(picked["gamma"])
        truth[off + 2] = float(picked["u"])
        truth[off + 3] = float(picked["taueff"])

    return mock_obs, truth


def raw_to_derived(raw_samples):
    raw_samples = np.asarray(raw_samples, dtype=np.float64)
    derived = np.zeros_like(raw_samples)

    derived[:, 0] = raw_samples[:, 0]
    derived[:, 1] = raw_samples[:, 1]

    for i, z_str in enumerate(Z_ORDER):
        off = 2 + 4 * i
        zrei = raw_samples[:, off + 0]
        ha = raw_samples[:, off + 1]
        hs = raw_samples[:, off + 2]
        tau = raw_samples[:, off + 3]

        x = np.column_stack([np.full(len(raw_samples), Z_FLOAT[z_str]), zrei, ha, hs])
        x_scaled = G_TGU.scaler.transform(x)
        tgu = np.asarray(G_TGU.interp(x_scaled), dtype=np.float64)

        derived[:, off + 0:off + 3] = tgu
        derived[:, off + 3] = tau

    return derived


def make_initial_walkers():
    pos = []
    n_try = 0
    while len(pos) < N_WALKERS:
        n_try += 1
        p = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        if np.isfinite(log_prior(p)):
            pos.append(p)
        if n_try > 200000:
            raise RuntimeError("Failed to initialize walkers. Please check bounds or TGU prior.")
    return np.asarray(pos, dtype=np.float64)


def plot_corner(samples, truth, out_pdf, cols=None):
    if cols is None:
        chain = samples
        labels = DERIVED_LABELS
        truth_use = truth
    else:
        chain = samples[:, cols]
        labels = [DERIVED_LABELS[i] for i in cols]
        truth_use = truth[cols]

    truth_dict = {lab: float(val) for lab, val in zip(labels, truth_use)}

    c = ChainConsumer()
    c.add_chain(
        chain=chain,
        parameters=labels,
        name="Mock data",
        shade=True,
        shade_alpha=0.55,
        linewidth=1.3,
    )
    c.configure(
        summary=False,
        max_ticks=2,
        diagonal_tick_labels=False,
        label_font_size=7,
        tick_font_size=7,
        usetex=False,
        legend_kwargs={"loc": "upper right", "fontsize": 18}
    )
    c.configure_truth(color="black", linestyle="--", linewidth=1.2, alpha=0.9)

    fig = c.plotter.plot(parameters=labels, truth=truth_dict,legend=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved corner: {out_pdf}")


def run_one_mock(mock_idx):
    global G_OBS

    mock_name = f"mock{mock_idx - 200}"
    seed = MOCK_SEED_BASE + mock_idx
    np.random.seed(seed)
    random.seed(seed)

    print("\n" + "=" * 72)
    print(f"[*] Running {mock_name}: simulation index = {mock_idx}, seed = {seed}")
    print("=" * 72)

    G_OBS, truth = build_mock_data(mock_idx, seed)
    pos = make_initial_walkers()

    print(f"    MCMC: steps={MCMC_STEPS}, walkers={N_WALKERS}, processes={N_PROC}")
    with Pool(processes=N_PROC, initializer=init_worker, initargs=(G_NN, G_TGU, G_OBS)) as pool:
        sampler = emcee.EnsembleSampler(N_WALKERS, 14, log_probability, pool=pool)
        sampler.run_mcmc(pos, MCMC_STEPS, progress=True)

    burnin = int(MCMC_STEPS * BURNIN_FRAC)
    raw_samples = sampler.get_chain(discard=burnin, thin=THIN, flat=True)
    lnprob = sampler.get_log_prob(discard=burnin, thin=THIN, flat=True)
    derived_samples = raw_to_derived(raw_samples)
    derived_samples[:,2] = derived_samples[:,2]/1e4
    derived_samples[:,6] = derived_samples[:,6]/1e4
    derived_samples[:,10] = derived_samples[:,10]/1e4
    truth[2] = truth[2]/1e4
    truth[6] = truth[6]/1e4
    truth[10] = truth[10]/1e4
    plot_corner(
        derived_samples,
        truth,
        os.path.join(OUT_DIR, f"{mock_name}_corner_full.pdf"),
        cols=None,
    )

    simple_cols = [0, 1, 2, 4, 6, 8, 10, 12]
    plot_corner(
        derived_samples,
        truth,
        os.path.join(OUT_DIR, f"{mock_name}_corner_mf_Tu.pdf"),
        cols=simple_cols,
    )


def main():
    global G_NN, G_TGU

    print("========== Mock MCMC with new N=100 NN emulator ==========")
    G_NN = load_nn_emulator()
    G_TGU = load_tgu_emulator()

    for mock_idx in MOCK_INDICES:
        run_one_mock(mock_idx)

    print("\n[*] All mock MCMC runs finished.")
    print(f"[*] Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
