# =============================================================================
# Quantum Reservoir Computing — MerLin v4 — Swaption Vol Surface Forecasting
# =============================================================================
#
# Dataset : Quandela/Challenge_Swaptions (Qvolution hackathon)
# Task    : Next-day implied volatility surface prediction (regression)
#
# Pipeline
# --------
#   1. Load level-1 (train) and level-2 (test) vol surface data
#   2. PCA compress: 224-dim surface -> N_MODES latent scores
#      (vol surfaces are smooth; first 4 PCs typically explain >98% variance)
#   3. Build lag pairs: X[t] = today's scores, y[t+1] = tomorrow's scores
#   4. Frozen QRC reservoir: N_MODES-dim -> Fock probability features
#   5. Linear readout: Fock features -> N_MODES predicted scores (next day)
#   6. PCA reconstruct: predicted scores -> full 224-dim surface
#   7. Full ablation suite vs classical baselines
#   8. Rich dashboard: RMSE/R2, loss curves, surface heatmaps,
#      time-series comparison, PCA of reservoir features, ablation bar chart
#
# Ablation suite
# --------------
#   1. Persistence (naive baseline: predict today = tomorrow)
#   2. Ridge regression on raw PCA scores (classical linear)
#   3. Random projection + linear readout  (is QRC > noise?)
#   4. Classical MLP on PCA scores         (classical nonlinear)
#   5. QRC-Linear (reservoir + linear readout)  <-- true QRC
#   6. QRC-MLP    (reservoir + MLP readout)     <-- comparison
#
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import pandas as pd
from datasets import load_dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from merlin import MeasurementStrategy, QuantumLayer
from merlin.builder import CircuitBuilder

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Colours ───────────────────────────────────────────────────────────────────
ACCENT  = "#00e5ff"   # QRC-Linear
ACCENT2 = "#a78bfa"   # QRC-MLP
OK      = "#22c55e"
WARN    = "#f59e0b"   # Classical MLP
RED     = "#ef4444"   # Random projection
MUTED   = "#475569"   # Classical baselines

# =============================================================================
# 1) Load data
# =============================================================================
print("Loading datasets...")

ds_train = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-1_Future_prediction/train.csv",
    split="train",
)
ds_test = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-2_Missing_data_prediction/train_level2.csv",
    split="train",
)

df_train = ds_train.to_pandas()
df_test  = ds_test.to_pandas()

VOL_COLS = [c for c in df_train.columns if c != "Date"]

X_train_raw = df_train[VOL_COLS].values.astype("float32")  # (494, 224)
X_test_raw  = df_test[VOL_COLS].values.astype("float32")   # (489, 224)

# Impute NaNs in test set with train column means
col_means = np.nanmean(X_train_raw, axis=0)
for j in range(X_test_raw.shape[1]):
    mask = np.isnan(X_test_raw[:, j])
    if mask.any():
        X_test_raw[mask, j] = col_means[j]

print(f"  Train surface shape : {X_train_raw.shape}")
print(f"  Test  surface shape : {X_test_raw.shape}")

# =============================================================================
# 2) PCA compression: 224-dim surface -> N_MODES latent scores
# =============================================================================
N_MODES   = 4    # PCA components = quantum optical modes
DEPTH     = 5    # reservoir depth
N_PHOTONS = 3    # Fock space = C(N_MODES+N_PHOTONS, N_PHOTONS) states

pca_full = PCA(n_components=min(20, X_train_raw.shape[1])).fit(X_train_raw)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)

pca = PCA(n_components=N_MODES, random_state=SEED)
pca.fit(X_train_raw)
var_explained = pca.explained_variance_ratio_.sum()
print(f"\n  PCA variance explained by {N_MODES} components: {var_explained:.2%}")

scores_train = pca.transform(X_train_raw).astype("float32")  # (494, 4)
scores_test  = pca.transform(X_test_raw).astype("float32")   # (489, 4)

scaler = StandardScaler()
scores_train = scaler.fit_transform(scores_train).astype("float32")
scores_test  = scaler.transform(scores_test).astype("float32")

# =============================================================================
# 3) Lag pairs for next-day forecasting
#    X[t] = today's PCA scores  ->  y[t+1] = tomorrow's PCA scores
# =============================================================================
X_tr = torch.tensor(scores_train[:-1], dtype=torch.float32)   # (493, 4)
y_tr = torch.tensor(scores_train[1:],  dtype=torch.float32)   # (493, 4)
X_te = torch.tensor(scores_test[:-1],  dtype=torch.float32)   # (488, 4)
y_te = torch.tensor(scores_test[1:],   dtype=torch.float32)   # (488, 4)

# True surfaces for evaluation (in original vol space)
true_surface_train = pca.inverse_transform(scaler.inverse_transform(scores_train[1:]))
true_surface_test  = pca.inverse_transform(scaler.inverse_transform(scores_test[1:]))

print(f"\n  Training lag pairs : {X_tr.shape[0]}")
print(f"  Test lag pairs     : {X_te.shape[0]}")

# =============================================================================
# 4) QRC Reservoir
#    Single angle-encoding block (name="enc") -> nb_input_tensor = 1.
#    Reservoir layers: superpositions -> rotations -> entangling (Clements order).
#    trainable=True so weights are nn.Parameters inside QuantumLayer, then frozen.
# =============================================================================
def build_reservoir(
    n_photons:      int                 = N_PHOTONS,
    encoding_scale: float               = np.pi / 3,
    measurement:    MeasurementStrategy = None,
    seed:           int                 = SEED,
):
    if measurement is None:
        measurement = MeasurementStrategy.probs()

    torch.manual_seed(seed)
    np.random.seed(seed)

    builder = CircuitBuilder(n_modes=N_MODES)
    builder.add_angle_encoding(
        modes=list(range(N_MODES)),
        name="enc",
        scale=encoding_scale,
    )
    for d in range(DEPTH):
        builder.add_superpositions(depth=1, trainable=True, name=f"bs_{d}")
        builder.add_rotations(trainable=True,               name=f"phi_{d}")
        builder.add_entangling_layer(trainable=True,        name=f"ent_{d}")

    layer = QuantumLayer(
        input_size=N_MODES,
        builder=builder,
        n_photons=n_photons,
        measurement_strategy=measurement,
    )
    for p in layer.parameters():
        p.requires_grad_(False)
    return layer


def extract_features(reservoir, X):
    with torch.no_grad():
        q = reservoir(X)
    if torch.is_complex(q):
        return torch.cat([q.real, q.imag], dim=1).float()
    return q.float()

# Build the main QRC reservoir (Config D: +probs, best physics settings)
print("\nBuilding QRC reservoir...")
reservoir = build_reservoir(
    n_photons=N_PHOTONS,
    encoding_scale=np.pi / 3,
    measurement=MeasurementStrategy.probs(),
)
print(f"  output_size = {reservoir.output_size}")

print("Extracting Fock features (reservoir runs once)...")
F_tr = extract_features(reservoir, X_tr)
F_te = extract_features(reservoir, X_te)
FEAT_DIM = F_tr.shape[1]
print(f"  Fock feature dim: {FEAT_DIM}")

# =============================================================================
# 5) Training helpers
# =============================================================================
mse_loss = nn.MSELoss()

def train_model(model, X_feat, y, lr=0.01, epochs=3000, patience=100, label=""):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss, wait, best_state = float("inf"), 0, None
    history = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = mse_loss(model(X_feat), y)
        loss.backward()
        opt.step()
        sched.step()

        lv = loss.item()
        history.append(lv)

        if lv < best_loss - 1e-8:
            best_loss, wait = lv, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if label:
                    print(f"  [{label}] early stop @ epoch {epoch+1}  loss={lv:.6f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


def predict_surface(model, X_feat):
    """Predict PCA scores then reconstruct full vol surface."""
    model.eval()
    with torch.no_grad():
        pred_scores = model(X_feat).numpy()
    return pca.inverse_transform(scaler.inverse_transform(pred_scores))


def surface_metrics(pred, true):
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = float(r2_score(true.flatten(), pred.flatten()))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def make_readout(in_dim, out_dim=N_MODES):
    return nn.Linear(in_dim, out_dim)


def make_mlp(in_dim, out_dim=N_MODES):
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32),     nn.ReLU(),
        nn.Linear(32, out_dim),
    )

# =============================================================================
# 6) Ablation Suite
#
#   1. Persistence   — predict tomorrow = today (naive baseline)
#   2. Ridge         — linear regression on PCA scores
#   3. Random proj   — random nonlinear projection + linear readout
#   4. Classical MLP — nonlinear model on raw PCA scores
#   5. QRC-Linear    — frozen reservoir + linear readout (true QRC)
#   6. QRC-MLP       — frozen reservoir + MLP readout (extended QRC)
# =============================================================================
ablation = {}   # name -> {metrics_tr, metrics_te, history}

# ── 1. Persistence baseline ───────────────────────────────────────────────────
print("\n[1/6] Persistence baseline")
persist_pred_train = pca.inverse_transform(scaler.inverse_transform(scores_train[:-1]))
persist_pred_test  = pca.inverse_transform(scaler.inverse_transform(scores_test[:-1]))
ablation["Persistence"] = dict(
    metrics_tr=surface_metrics(persist_pred_train, true_surface_train),
    metrics_te=surface_metrics(persist_pred_test,  true_surface_test),
    history=None,
)
print(f"  Test RMSE={ablation['Persistence']['metrics_te']['rmse']:.6f}  "
      f"R2={ablation['Persistence']['metrics_te']['r2']:.4f}")

# ── 2. Ridge regression on PCA scores ─────────────────────────────────────────
print("\n[2/6] Ridge regression (classical linear)")
ridge = Ridge(alpha=1.0)
ridge.fit(scores_train[:-1], scores_train[1:])

ridge_pred_tr = pca.inverse_transform(scaler.inverse_transform(ridge.predict(scores_train[:-1])))
ridge_pred_te = pca.inverse_transform(scaler.inverse_transform(ridge.predict(scores_test[:-1])))
ablation["Ridge"] = dict(
    metrics_tr=surface_metrics(ridge_pred_tr, true_surface_train),
    metrics_te=surface_metrics(ridge_pred_te, true_surface_test),
    history=None,
)
print(f"  Test RMSE={ablation['Ridge']['metrics_te']['rmse']:.6f}  "
      f"R2={ablation['Ridge']['metrics_te']['r2']:.4f}")

# ── 3. Random projection + linear readout ─────────────────────────────────────
print("\n[3/6] Random projection + linear readout")
gen       = torch.Generator().manual_seed(SEED)
rand_proj = torch.randn(N_MODES, FEAT_DIM, generator=gen)
rand_proj /= rand_proj.norm(dim=0, keepdim=True).clamp(min=1e-8)

X_tr_rand = X_tr @ rand_proj
X_te_rand  = X_te @ rand_proj

rand_lin    = make_readout(FEAT_DIM)
hist_rand   = train_model(rand_lin, X_tr_rand, y_tr, lr=0.01, label="RandProj")
rand_pred_tr = predict_surface(rand_lin, X_tr_rand)
rand_pred_te = predict_surface(rand_lin, X_te_rand)
ablation["Random Projection"] = dict(
    metrics_tr=surface_metrics(rand_pred_tr, true_surface_train),
    metrics_te=surface_metrics(rand_pred_te, true_surface_test),
    history=hist_rand,
)
print(f"  Test RMSE={ablation['Random Projection']['metrics_te']['rmse']:.6f}  "
      f"R2={ablation['Random Projection']['metrics_te']['r2']:.4f}")

# ── 4. Classical MLP on PCA scores ────────────────────────────────────────────
print("\n[4/6] Classical MLP")
classic_mlp  = make_mlp(N_MODES)
hist_cmlp    = train_model(classic_mlp, X_tr, y_tr, lr=0.005, label="ClassicMLP")
cmlp_pred_tr = predict_surface(classic_mlp, X_tr)
cmlp_pred_te = predict_surface(classic_mlp, X_te)
ablation["Classical MLP"] = dict(
    metrics_tr=surface_metrics(cmlp_pred_tr, true_surface_train),
    metrics_te=surface_metrics(cmlp_pred_te, true_surface_test),
    history=hist_cmlp,
)
print(f"  Test RMSE={ablation['Classical MLP']['metrics_te']['rmse']:.6f}  "
      f"R2={ablation['Classical MLP']['metrics_te']['r2']:.4f}")

# ── 5. QRC-Linear ─────────────────────────────────────────────────────────────
print("\n[5/6] QRC-Linear (true reservoir computing)")
qrc_lin      = make_readout(FEAT_DIM)
hist_qlin    = train_model(qrc_lin, F_tr, y_tr, lr=0.01, label="QRC-Linear")
qlin_pred_tr = predict_surface(qrc_lin, F_tr)
qlin_pred_te = predict_surface(qrc_lin, F_te)
ablation["QRC-Linear"] = dict(
    metrics_tr=surface_metrics(qlin_pred_tr, true_surface_train),
    metrics_te=surface_metrics(qlin_pred_te, true_surface_test),
    history=hist_qlin,
)
print(f"  Test RMSE={ablation['QRC-Linear']['metrics_te']['rmse']:.6f}  "
      f"R2={ablation['QRC-Linear']['metrics_te']['r2']:.4f}")

# ── 6. QRC-MLP ────────────────────────────────────────────────────────────────
print("\n[6/6] QRC-MLP")
qrc_mlp      = make_mlp(FEAT_DIM)
hist_qmlp    = train_model(qrc_mlp, F_tr, y_tr, lr=0.005, label="QRC-MLP")
qmlp_pred_tr = predict_surface(qrc_mlp, F_tr)
qmlp_pred_te = predict_surface(qrc_mlp, F_te)
ablation["QRC-MLP"] = dict(
    metrics_tr=surface_metrics(qmlp_pred_tr, true_surface_train),
    metrics_te=surface_metrics(qmlp_pred_te, true_surface_test),
    history=hist_qmlp,
)
print(f"  Test RMSE={ablation['QRC-MLP']['metrics_te']['rmse']:.6f}  "
      f"R2={ablation['QRC-MLP']['metrics_te']['r2']:.4f}")

# =============================================================================
# 7) Visualization Dashboard
# =============================================================================
plt.rcParams.update({
    "font.family":      "monospace",
    "axes.facecolor":   "#0d1117",
    "figure.facecolor": "#0a0c12",
    "axes.edgecolor":   "#2a2d3e",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1a1d2e",
    "grid.linewidth":   0.6,
})

abl_keys   = list(ablation.keys())
abl_rmse   = [ablation[k]["metrics_te"]["rmse"] for k in abl_keys]
abl_r2     = [ablation[k]["metrics_te"]["r2"]   for k in abl_keys]

bar_colors = []
for k in abl_keys:
    if k == "QRC-MLP":              bar_colors.append(ACCENT2)
    elif k == "QRC-Linear":         bar_colors.append(ACCENT)
    elif "Random" in k:             bar_colors.append(RED)
    elif "Classical" in k:          bar_colors.append(WARN)
    elif k == "Ridge":              bar_colors.append(MUTED)
    else:                           bar_colors.append("#334155")  # Persistence

fig = plt.figure(figsize=(24, 22))
fig.patch.set_facecolor("#0a0c12")
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.40)

fig.text(0.5, 0.978,
         "QRC v4 -- Swaption Vol Surface Forecasting -- MerLin",
         ha="center", va="top", fontsize=16, fontweight="bold", color="#e2e8f0")
fig.text(0.5, 0.962,
         f"PCA {N_MODES} components ({var_explained:.1%} variance explained)  "
         f"| {N_MODES} modes  {N_PHOTONS} photons  depth {DEPTH}  "
         f"| Fock feature dim {FEAT_DIM}",
         ha="center", va="top", fontsize=9, color="#64748b")

# ── Panel 1: Ablation RMSE bar chart ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
x   = np.arange(len(abl_keys))
bars = ax1.bar(x, abl_rmse, color=bar_colors, width=0.55,
               edgecolor="#0a0c12", linewidth=0.5, zorder=2)
for bar, val in zip(bars, abl_rmse):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 2e-5,
             f"{val:.5f}", ha="center", va="bottom", fontsize=7.5,
             fontweight="bold", color="#e2e8f0")
ax1.set_xticks(x)
ax1.set_xticklabels([k.replace("\n", "\n") for k in abl_keys], fontsize=8)
ax1.set_ylabel("Test RMSE (implied vol units)")
ax1.set_title("Ablation Suite -- Test RMSE  (lower is better)",
              fontsize=11, color="#e2e8f0", pad=8)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.set_facecolor("#0d1117")
legend_handles = [
    Patch(color=ACCENT,   label="QRC-Linear (true reservoir readout)"),
    Patch(color=ACCENT2,  label="QRC-MLP (nonlinear readout)"),
    Patch(color=RED,      label="Random projection (sanity check)"),
    Patch(color=WARN,     label="Classical MLP"),
    Patch(color=MUTED,    label="Ridge regression"),
    Patch(color="#334155",label="Persistence (naive baseline)"),
]
ax1.legend(handles=legend_handles, framealpha=0.15, facecolor="#1a1d2e",
           fontsize=7, loc="upper right", ncol=2)

# ── Panel 2: Ablation R² bar chart ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
bars2 = ax2.bar(x, abl_r2, color=bar_colors, width=0.55,
                edgecolor="#0a0c12", linewidth=0.5, zorder=2)
for bar, val in zip(bars2, abl_r2):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7.5,
             fontweight="bold", color="#e2e8f0")
ax2.axhline(1.0, color=OK, lw=0.8, ls=":", alpha=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels([k.replace("\n", "\n") for k in abl_keys], fontsize=8)
ax2.set_ylabel("Test R2 score  (higher is better)")
ax2.set_title("Ablation Suite -- Test R2",
              fontsize=11, color="#e2e8f0", pad=8)
ax2.set_ylim(max(0, min(abl_r2) - 0.05), 1.05)
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.set_facecolor("#0d1117")

# ── Panel 3: Training loss curves (models that were trained) ──────────────────
ax3 = fig.add_subplot(gs[1, :2])
trained = {
    "Random Proj":   (hist_rand,  RED),
    "Classical MLP": (hist_cmlp,  WARN),
    "QRC-Linear":    (hist_qlin,  ACCENT),
    "QRC-MLP":       (hist_qmlp,  ACCENT2),
}
for label, (hist, col) in trained.items():
    if hist:
        ax3.plot(hist, color=col, lw=1.4, label=label, alpha=0.9)
ax3.set_title("Training Loss (MSE on PCA scores)", fontsize=11, color="#e2e8f0", pad=8)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("MSE")
ax3.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8)
ax3.grid(True, alpha=0.3); ax3.set_facecolor("#0d1117")

# ── Panel 4: PCA variance explained ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2:])
ax4.plot(range(1, len(cumvar) + 1), cumvar,
         color=ACCENT, lw=2, marker="o", markersize=4)
ax4.axvline(N_MODES, color=WARN, lw=1.5, ls="--",
            label=f"{N_MODES} components used ({var_explained:.1%})")
ax4.axhline(0.99, color=OK, lw=1, ls=":", alpha=0.6, label="99% threshold")
ax4.set_xlabel("Number of PCA components")
ax4.set_ylabel("Cumulative variance explained")
ax4.set_title("Vol Surface PCA -- Variance Explained", fontsize=11, color="#e2e8f0", pad=8)
ax4.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8)
ax4.grid(True, alpha=0.3); ax4.set_facecolor("#0d1117")
ax4.set_ylim(0, 1.05)

# ── Panel 5 & 6: Predicted vs True vol surface heatmaps (QRC-Linear, test day 0)
tenors     = sorted(set(int(re.search(r"Tenor : (\d+)", c).group(1))         for c in VOL_COLS))
maturities = sorted(set(float(re.search(r"Maturity : ([\d.]+)", c).group(1)) for c in VOL_COLS))
T, M       = len(tenors), len(maturities)

def to_grid(surface_row):
    grid = np.zeros((T, M))
    for idx, col in enumerate(VOL_COLS):
        ti = tenors.index(int(re.search(r"Tenor : (\d+)", col).group(1)))
        mi = maturities.index(float(re.search(r"Maturity : ([\d.]+)", col).group(1)))
        grid[ti, mi] = surface_row[idx]
    return grid

DAY        = 0
pred_grid  = to_grid(qlin_pred_te[DAY])
true_grid  = to_grid(true_surface_test[DAY])
vmin, vmax = min(pred_grid.min(), true_grid.min()), max(pred_grid.max(), true_grid.max())

xt_ticks  = range(M)[::2]
xt_labels = [f"{maturities[i]:.1f}" for i in xt_ticks]

ax5 = fig.add_subplot(gs[2, :2])
im5 = ax5.imshow(pred_grid, aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)
ax5.set_title(f"QRC-Linear Predicted Surface -- Test Day {DAY+1}",
              fontsize=10, color="#e2e8f0", pad=6)
ax5.set_xlabel("Maturity"); ax5.set_ylabel("Tenor")
ax5.set_xticks(list(xt_ticks)); ax5.set_xticklabels(xt_labels, fontsize=6, rotation=45)
ax5.set_yticks(range(T)); ax5.set_yticklabels([str(t) for t in tenors], fontsize=7)
plt.colorbar(im5, ax=ax5, fraction=0.03).set_label("Implied Vol", color="#94a3b8")

ax6 = fig.add_subplot(gs[2, 2:])
im6 = ax6.imshow(true_grid, aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)
ax6.set_title(f"True Vol Surface -- Test Day {DAY+1}",
              fontsize=10, color="#e2e8f0", pad=6)
ax6.set_xlabel("Maturity"); ax6.set_ylabel("Tenor")
ax6.set_xticks(list(xt_ticks)); ax6.set_xticklabels(xt_labels, fontsize=6, rotation=45)
ax6.set_yticks(range(T)); ax6.set_yticklabels([str(t) for t in tenors], fontsize=7)
plt.colorbar(im6, ax=ax6, fraction=0.03).set_label("Implied Vol", color="#94a3b8")

# ── Panel 7: Error heatmap (QRC-Linear vs True) ───────────────────────────────
ax7 = fig.add_subplot(gs[3, :2])
err_grid = pred_grid - true_grid
emax     = np.abs(err_grid).max()
im7 = ax7.imshow(err_grid, aspect="auto", cmap="RdBu_r", vmin=-emax, vmax=emax)
ax7.set_title(f"QRC-Linear Prediction Error (Pred - True) -- Test Day {DAY+1}",
              fontsize=10, color="#e2e8f0", pad=6)
ax7.set_xlabel("Maturity"); ax7.set_ylabel("Tenor")
ax7.set_xticks(list(xt_ticks)); ax7.set_xticklabels(xt_labels, fontsize=6, rotation=45)
ax7.set_yticks(range(T)); ax7.set_yticklabels([str(t) for t in tenors], fontsize=7)
plt.colorbar(im7, ax=ax7, fraction=0.03).set_label("Error", color="#94a3b8")

# ── Panel 8: Time-series comparison for one vol cell ──────────────────────────
ax8      = fig.add_subplot(gs[3, 2:])
CELL_COL = "Tenor : 5; Maturity : 5"
cidx     = VOL_COLS.index(CELL_COL)
N_DAYS   = min(120, len(true_surface_test))

ax8.plot(true_surface_test[:N_DAYS, cidx],  color=ACCENT,  lw=1.5,
         label="True",         alpha=0.95)
ax8.plot(qlin_pred_te[:N_DAYS, cidx],       color=WARN,    lw=1.5,
         label="QRC-Linear",   alpha=0.9,  ls="--")
ax8.plot(qmlp_pred_te[:N_DAYS, cidx],       color=ACCENT2, lw=1.3,
         label="QRC-MLP",      alpha=0.85, ls="--")
ax8.plot(ridge_pred_te[:N_DAYS, cidx],      color=MUTED,   lw=1.2,
         label="Ridge",        alpha=0.75, ls=":")
ax8.plot(persist_pred_test[:N_DAYS, cidx],  color=RED,     lw=1.0,
         label="Persistence",  alpha=0.6,  ls=":")
ax8.set_title(f"Time Series: {CELL_COL} -- first {N_DAYS} test days",
              fontsize=10, color="#e2e8f0", pad=6)
ax8.set_xlabel("Test day index"); ax8.set_ylabel("Implied Vol")
ax8.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8, ncol=2)
ax8.grid(True, alpha=0.3); ax8.set_facecolor("#0d1117")

plt.savefig("qrc_swaptions_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("\nDashboard saved as qrc_swaptions_dashboard.png")

# =============================================================================
# 8) Final Summary
# =============================================================================
print("\n" + "="*68)
print("  SWAPTION VOL SURFACE FORECASTING -- ABLATION SUMMARY")
print("="*68)
print(f"  {'Model':<22} {'Train RMSE':>11}  {'Test RMSE':>10}  {'Test R2':>8}  {'vs Ridge':>10}")
print("-"*68)

ridge_rmse = ablation["Ridge"]["metrics_te"]["rmse"]

for k in abl_keys:
    m_tr  = ablation[k]["metrics_tr"]
    m_te  = ablation[k]["metrics_te"]
    delta = m_te["rmse"] - ridge_rmse
    flag  = f"{delta:+.6f}"
    print(f"  {k.replace(chr(10),' '):<22} {m_tr['rmse']:>11.6f}  "
          f"{m_te['rmse']:>10.6f}  {m_te['r2']:>8.4f}  {flag:>10}")

print("="*68)

qrc_lin_rmse = ablation["QRC-Linear"]["metrics_te"]["rmse"]
qrc_mlp_rmse = ablation["QRC-MLP"]["metrics_te"]["rmse"]
rand_rmse    = ablation["Random Projection"]["metrics_te"]["rmse"]

print("\n  Diagnostics:")
if qrc_lin_rmse < rand_rmse:
    print(f"  [OK] QRC-Linear beats random projection "
          f"({qrc_lin_rmse:.6f} vs {rand_rmse:.6f})")
    print(f"       -> Quantum circuit generating useful structure")
else:
    print(f"  [!!] QRC-Linear does not beat random projection")
    print(f"       -> Try N_PHOTONS=4 or N_MODES=6")

if qrc_lin_rmse < ridge_rmse:
    improvement = (ridge_rmse - qrc_lin_rmse) / ridge_rmse * 100
    print(f"  [OK] QRC-Linear beats Ridge by {improvement:.2f}% RMSE reduction")
else:
    print(f"  [--] QRC-Linear within {qrc_lin_rmse - ridge_rmse:.6f} RMSE of Ridge")

gap = qrc_mlp_rmse - qrc_lin_rmse
if abs(gap) < 0.0005:
    print(f"  [OK] QRC-Linear approx QRC-MLP (gap={gap:+.6f})")
    print(f"       -> Reservoir doing the nonlinear work, linear readout sufficient")
else:
    print(f"  [--] QRC-MLP vs QRC-Linear gap={gap:+.6f}")
    print(f"       -> Linear readout not fully exploiting Fock features")
print("="*68)
