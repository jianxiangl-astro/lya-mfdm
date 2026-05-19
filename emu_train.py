"""
Train the two-stage neural network emulator for the Lyman-alpha forest 1D flux power spectrum.

The emulator consists of two parts:
1. A CDM 1D flux power spectrum emulator 
2. A transfer function emulator for the 1D flux power spectrum
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import warnings
import random

warnings.filterwarnings('ignore')

NUM_THREADS = 56
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)

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
OBS_FILE = "data/lya_data.pkl"
OUT_DIR = "emu"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_START, TEST_END = 200, 209
N_FOLDS = 5
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 1200
LR = 1e-3

WEIGHT_DECAY = 1e-6
PATIENCE_LR = 60
LR_FACTOR = 0.5
MIN_LR = 1e-6
PATIENCE_EARLY_STOP = 200

F_EPS = 1e-8

TARGET_COLS = [
    "-2.2", "-2.1", "-2.0", "-1.9", "-1.8", "-1.7", "-1.6", "-1.5",
    "-1.4", "-1.3", "-1.2", "-1.1", "-1.0", "-0.9", "-0.8", "-0.7"
]

Z_VALUES = ["5.0", "4.6", "4.2"]
Z_FLOAT = {"5.0": 5.0, "4.6": 4.6, "4.2": 4.2}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
        nn.Linear(HIDDEN_DIM, out_dim)
    )

class WeightedLogLoss(nn.Module):
    def forward(self, pred, true, sigma):
        weight = 1.0 / torch.clamp(sigma, min=1e-12) ** 2
        err2 = (pred - true) ** 2
        return (err2 * weight).mean() / torch.clamp(weight.mean(), min=1e-30)

class ResidualFinalLogLoss(nn.Module):
    def forward(self, pred_g, true_g, f_raw, sigma_log):
        weight = 1.0 / torch.clamp(sigma_log, min=1e-12) ** 2

        f_safe = torch.where(
            f_raw > F_EPS,
            f_raw,
            torch.zeros_like(f_raw)
        )

        delta_log = f_safe * (pred_g - true_g)
        err2 = delta_log ** 2

        return (err2 * weight).mean() / torch.clamp(weight.mean(), min=1e-30)

def get_joint_data(max_index):
    obs_data = load_pkl(OBS_FILE)

    x_cdm_list, y_cdm_list, sig_cdm_list, idx_cdm_list = [], [], [], []
    x_res_list, y_res_list, sig_res_list, f_res_list, idx_res_list = [], [], [], [], []
    test_data = {}

    for z_str in Z_VALUES:
        idx_suffix = {"5.0": "0", "4.6": "1", "4.2": "2"}[z_str]

        ncdm_all = load_pkl(os.path.join(BASE_SIM_DIR, f"all_pk_num{idx_suffix}.pkl"))
        cdm_all = load_pkl(os.path.join(BASE_SIM_DIR, f"all_pk_num{idx_suffix}_cdm.pkl"))

        obs_sigma = np.sqrt(np.diag(obs_data[z_str]["Cov"])).astype(np.float64)

        sim_ids_ncdm = ncdm_all['index']
        sim_ids_cdm = cdm_all['index']
        common_ids = np.intersect1d(sim_ids_ncdm, sim_ids_cdm)
        train_ids = common_ids[common_ids < max_index]

        ncdm_train_raw = ncdm_all[np.isin(sim_ids_ncdm, train_ids)]
        cdm_train_raw = cdm_all[np.isin(sim_ids_cdm, train_ids)]

        ncdm_train = np.sort(ncdm_train_raw, order=['index', 'taueff'])
        cdm_train = np.sort(cdm_train_raw, order=['index', 'taueff'])

        test_mask = (ncdm_all['index'] >= TEST_START) & (ncdm_all['index'] <= TEST_END)
        ncdm_test = np.sort(ncdm_all[test_mask], order=['index', 'taueff'])

        z_obs = Z_FLOAT[z_str]

        x_c = np.column_stack((
            np.full(len(cdm_train), z_obs),
            cdm_train['z'],
            cdm_train['ha'],
            cdm_train['hs'],
            cdm_train['taueff']
        ))

        y_c = np.log10(np.column_stack([cdm_train[c] for c in TARGET_COLS]))
        sig_c = obs_sigma / (10 ** y_c * np.log(10.0))

        x_cdm_list.append(x_c)
        y_cdm_list.append(y_c)
        sig_cdm_list.append(sig_c)
        idx_cdm_list.append(cdm_train['index'])

        x_r = np.column_stack((
            np.full(len(ncdm_train), z_obs),
            ncdm_train['m'],
            ncdm_train['f'],
            ncdm_train['z'],
            ncdm_train['ha'],
            ncdm_train['hs'],
            ncdm_train['taueff']
        ))

        y_n = np.log10(np.column_stack([ncdm_train[c] for c in TARGET_COLS]))
        f_arr = ncdm_train['f'].reshape(-1, 1).astype(np.float64)

        safe_f = np.where(f_arr > F_EPS, f_arr, 1.0)
        y_r = (y_n - y_c) / safe_f
        y_r[f_arr[:, 0] <= F_EPS] = 0.0

        sig_res = obs_sigma / (10 ** y_n * np.log(10.0))

        x_res_list.append(x_r)
        y_res_list.append(y_r)
        sig_res_list.append(sig_res)
        f_res_list.append(f_arr)
        idx_res_list.append(ncdm_train['index'])

        test_data[z_str] = {
            'x_cdm': np.column_stack((
                np.full(len(ncdm_test), z_obs),
                ncdm_test['z'],
                ncdm_test['ha'],
                ncdm_test['hs'],
                ncdm_test['taueff']
            )),
            'x_res': np.column_stack((
                np.full(len(ncdm_test), z_obs),
                ncdm_test['m'],
                ncdm_test['f'],
                ncdm_test['z'],
                ncdm_test['ha'],
                ncdm_test['hs'],
                ncdm_test['taueff']
            )),
            'f_val': ncdm_test['f'].reshape(-1, 1),
            'true_linear': np.column_stack([ncdm_test[c] for c in TARGET_COLS]),
            'obs_sigma': obs_sigma
        }

    train_data = {
        'x_cdm': np.vstack(x_cdm_list),
        'y_cdm': np.vstack(y_cdm_list),
        'sig_cdm': np.vstack(sig_cdm_list),
        'id_cdm': np.concatenate(idx_cdm_list),

        'x_res': np.vstack(x_res_list),
        'y_res': np.vstack(y_res_list),
        'sig_res': np.vstack(sig_res_list),
        'f_res': np.vstack(f_res_list),
        'id_res': np.concatenate(idx_res_list)
    }

    return train_data, test_data


def train_and_evaluate(max_index):

    save_dir = os.path.join(OUT_DIR, f"emu_N{max_index}")
    os.makedirs(save_dir, exist_ok=True)

    train_data, test_data = get_joint_data(max_index)

    scaler_cdm = MinMaxScaler().fit(train_data['x_cdm'])
    scaler_res = MinMaxScaler().fit(train_data['x_res'])

    with open(os.path.join(save_dir, 'scaler_cdm.pkl'), 'wb') as f:
        pickle.dump(scaler_cdm, f)

    with open(os.path.join(save_dir, 'scaler_res.pkl'), 'wb') as f:
        pickle.dump(scaler_res, f)

    x_c_scaled = scaler_cdm.transform(train_data['x_cdm'])
    x_r_scaled = scaler_res.transform(train_data['x_res'])

    unique_sim_ids = np.unique(train_data['id_cdm'])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    cdm_models, res_models = [], []

    criterion_cdm = WeightedLogLoss()
    criterion_res = ResidualFinalLogLoss()

    for fold, (train_id_idx, val_id_idx) in enumerate(kf.split(unique_sim_ids)):
        print(f"  --> Training Fold {fold + 1}/{N_FOLDS} ...", end=" ")
        set_seed(42 + fold)
        
        group_train_ids = unique_sim_ids[train_id_idx]
        group_val_ids = unique_sim_ids[val_id_idx]
        mask_t_c = np.isin(train_data['id_cdm'], group_train_ids)
        mask_v_c = np.isin(train_data['id_cdm'], group_val_ids)

        ds_t_c = TensorDataset(
            torch.tensor(x_c_scaled[mask_t_c], dtype=torch.float32),
            torch.tensor(train_data['y_cdm'][mask_t_c], dtype=torch.float32),
            torch.tensor(train_data['sig_cdm'][mask_t_c], dtype=torch.float32)
        )

        ds_v_c = TensorDataset(
            torch.tensor(x_c_scaled[mask_v_c], dtype=torch.float32),
            torch.tensor(train_data['y_cdm'][mask_v_c], dtype=torch.float32),
            torch.tensor(train_data['sig_cdm'][mask_v_c], dtype=torch.float32)
        )

        dl_t_c = DataLoader(ds_t_c, batch_size=BATCH_SIZE, shuffle=True)
        dl_v_c = DataLoader(ds_v_c, batch_size=BATCH_SIZE, shuffle=False)

        model_c = build_mlp(x_c_scaled.shape[1], 16)

        opt_c = optim.AdamW(
            model_c.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler_c = optim.lr_scheduler.ReduceLROnPlateau(
            opt_c,
            mode='min',
            factor=LR_FACTOR,
            patience=PATIENCE_LR,
            min_lr=MIN_LR
        )

        best_loss_c = np.inf
        patience_c = 0
        best_state_c = None

        for ep in range(EPOCHS):
            model_c.train()

            for xb, yb, sb in dl_t_c:
                opt_c.zero_grad(set_to_none=True)
                loss = criterion_cdm(model_c(xb), yb, sb)
                loss.backward()
                opt_c.step()

            model_c.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion_cdm(model_c(xv), yv, sv).item() * xv.shape[0]
                    for xv, yv, sv in dl_v_c
                ) / len(ds_v_c)

            scheduler_c.step(val_loss)

            if val_loss < best_loss_c:
                best_loss_c = val_loss
                best_state_c = {
                    k: v.detach().cpu().clone()
                    for k, v in model_c.state_dict().items()
                }
                patience_c = 0
            else:
                patience_c += 1
                if patience_c >= PATIENCE_EARLY_STOP:
                    break

        if best_state_c is not None:
            model_c.load_state_dict(best_state_c)

        model_c.eval()
        cdm_models.append(model_c)
        torch.save(model_c.state_dict(), os.path.join(save_dir, f"model_cdm_fold{fold}.pth"))

        mask_t_r = np.isin(train_data['id_res'], group_train_ids)
        mask_v_r = np.isin(train_data['id_res'], group_val_ids)

        ds_t_r = TensorDataset(
            torch.tensor(x_r_scaled[mask_t_r], dtype=torch.float32),
            torch.tensor(train_data['y_res'][mask_t_r], dtype=torch.float32),
            torch.tensor(train_data['f_res'][mask_t_r], dtype=torch.float32),
            torch.tensor(train_data['sig_res'][mask_t_r], dtype=torch.float32)
        )

        ds_v_r = TensorDataset(
            torch.tensor(x_r_scaled[mask_v_r], dtype=torch.float32),
            torch.tensor(train_data['y_res'][mask_v_r], dtype=torch.float32),
            torch.tensor(train_data['f_res'][mask_v_r], dtype=torch.float32),
            torch.tensor(train_data['sig_res'][mask_v_r], dtype=torch.float32)
        )

        dl_t_r = DataLoader(ds_t_r, batch_size=BATCH_SIZE, shuffle=True)
        dl_v_r = DataLoader(ds_v_r, batch_size=BATCH_SIZE, shuffle=False)

        model_r = build_mlp(x_r_scaled.shape[1], 16)

        opt_r = optim.AdamW(
            model_r.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler_r = optim.lr_scheduler.ReduceLROnPlateau(
            opt_r,
            mode='min',
            factor=LR_FACTOR,
            patience=PATIENCE_LR,
            min_lr=MIN_LR
        )

        best_loss_r = np.inf
        patience_r = 0
        best_state_r = None

        for ep in range(EPOCHS):
            model_r.train()

            for xb, yb, fb, sb in dl_t_r:
                opt_r.zero_grad(set_to_none=True)
                loss = criterion_res(model_r(xb), yb, fb, sb)
                loss.backward()
                opt_r.step()

            model_r.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion_res(model_r(xv), yv, fv, sv).item() * xv.shape[0]
                    for xv, yv, fv, sv in dl_v_r
                ) / len(ds_v_r)

            scheduler_r.step(val_loss)

            if val_loss < best_loss_r:
                best_loss_r = val_loss
                best_state_r = {
                    k: v.detach().cpu().clone()
                    for k, v in model_r.state_dict().items()
                }
                patience_r = 0
            else:
                patience_r += 1
                if patience_r >= PATIENCE_EARLY_STOP:
                    break

        if best_state_r is not None:
            model_r.load_state_dict(best_state_r)

        model_r.eval()
        res_models.append(model_r)
        torch.save(model_r.state_dict(), os.path.join(save_dir, f"model_res_fold{fold}.pth"))

        print(f"Done. Best Val Loss (CDM: {best_loss_c:.2e}, RES: {best_loss_r:.2e})")

    results = {}

    for z_str, d in test_data.items():
        xc_t = torch.tensor(scaler_cdm.transform(d['x_cdm']), dtype=torch.float32)
        xr_t = torch.tensor(scaler_res.transform(d['x_res']), dtype=torch.float32)

        with torch.no_grad():
            pred_c_log = torch.mean(
                torch.stack([m(xc_t) for m in cdm_models]),
                dim=0
            ).numpy()

            pred_g = torch.mean(
                torch.stack([m(xr_t) for m in res_models]),
                dim=0
            ).numpy()

        f_arr = d['f_val']
        pred_r_log = pred_g * f_arr
        pred_r_log[f_arr[:, 0] <= F_EPS] = 0.0

        pred_linear = 10 ** (pred_c_log + pred_r_log)
        true_linear = d['true_linear']

        metric = (pred_linear - true_linear) / d['obs_sigma'].reshape(1, -1)
        results[z_str] = metric

        rms = np.sqrt(np.mean(metric ** 2))
        print(f" z={z_str}: RMS Error = {rms:.4f} sigma")

    with open(os.path.join(OUT_DIR, f"results_max_{max_index}.pkl"), 'wb') as f:
        pickle.dump(results, f)


def plot_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    k_bins = np.array([
        -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5,
        -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7
    ])
    z_plot_order = ["5.0", "4.6", "4.2"]

    cmap = plt.get_cmap("Spectral_r")
    styles = {
        25: {
            "color": cmap(0.18),
            "label": r"$N_{\mathrm{pair}} = 25$",
            "alpha": 0.20,
            "zorder": 2,
            "lw": 1.1,
        },
        50: {
            "color": cmap(0.50),
            "label": r"$N_{\mathrm{pair}} = 50$",
            "alpha": 0.26,
            "zorder": 3,
            "lw": 1.3,
        },
        100: {
            "color": cmap(0.82),
            "label": r"$N_{\mathrm{pair}} = 100$",
            "alpha": 0.34,
            "zorder": 4,
            "lw": 1.5,
        },
    }

    all_metrics = {}
    for max_idx in [25, 50, 100]:
        with open(os.path.join(OUT_DIR, f"results_max_{max_idx}.pkl"), "rb") as f:
            all_metrics[max_idx] = pickle.load(f)

    for i, z_str in enumerate(z_plot_order):
        ax = axes[i]

        ax.fill_between(
            k_bins, -1.0, 1.0,
            color="0.5", alpha=0.12,
            label=r"$\pm 1\sigma_{\mathrm{obs}}$" if i == 0 else None,
            zorder=0,
            linewidth=0,
        )

        for max_idx in [25, 50, 100]:
            metric = all_metrics[max_idx][z_str]

            p16 = np.percentile(metric, 15.865, axis=0)
            p84 = np.percentile(metric, 84.135, axis=0)

            c = styles[max_idx]["color"]

            ax.fill_between(
                k_bins, p16, p84,
                color=c,
                alpha=styles[max_idx]["alpha"],
                label=styles[max_idx]["label"] if i == 0 else None,
                zorder=styles[max_idx]["zorder"],
                linewidth=0,
            )

            ax.plot(
                k_bins, p16,
                color=c,
                lw=styles[max_idx]["lw"],
                alpha=0.90,
                zorder=styles[max_idx]["zorder"] + 0.1,
            )
            ax.plot(
                k_bins, p84,
                color=c,
                lw=styles[max_idx]["lw"],
                alpha=0.90,
                zorder=styles[max_idx]["zorder"] + 0.1,
            )

        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            lw=1.2,
            alpha=0.75,
            zorder=10,
        )

        ax.set_title(fr"$z = {z_str}$", fontsize=16)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(k_bins[0], k_bins[-1])
        ax.set_xlabel(
            r"$\log_{10}(k_\mathrm{f}~[\mathrm{s}\,\mathrm{km}^{-1}])$",
            fontsize=16,
        )
        ax.set_xticks(k_bins[::2])

        ax.tick_params(axis="both", which="both", direction="in", right=True)

        if i == 0:
            ax.set_ylabel(
                r"$(P_{\mathrm{f,pred}} - P_{\mathrm{f,true}}) / \sigma_{\mathrm{obs}}$",
                fontsize=16,
            )
            ax.legend(loc="upper left", fontsize=13, frameon=False)

    plt.subplots_adjust(wspace=0.05)
    
    plot_path = "plots/paper/Error_Bounds_Comparison.pdf"
    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == '__main__':
    for max_idx in [25, 50, 100]:
        print(f"\n{'=' * 50}\n MAX_INDEX = {max_idx}\n{'=' * 50}")
        train_and_evaluate(max_idx)

    plot_comparison()