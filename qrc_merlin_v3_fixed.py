# =============================================================================
# Quantum Reservoir Computing — MerLin v3 (Physically Corrected)
# =============================================================================
#
# Key physics changes vs v2:
#
#   1. ENCODING SCALE: scale=pi/3 maps ±3σ standardized data → ±π radians.
#      Covers the full Hilbert space. scale=1.0 was under-rotating.
#
#   2. MORE PHOTONS: n_photons=3 (Fock space C(6,3)=20) or 4 (C(7,4)=35).
#      With n_photons=2 you only get degree-2 polynomial features — too weak.
#      n_photons=3 gives degree-3 polynomials, enough for Iris linear separation.
#
#   3. MEASUREMENT: MeasurementStrategy.probs() instead of PHOTON_COUNTING.
#      Fock state probability distributions are real, non-negative, and directly
#      interpretable as occupation probabilities. The linear readout works
#      much better on these than on complex amplitudes.
#      (AMPLITUDES is kept as an option for comparison.)
#
#   4. RESERVOIR BLOCK ORDER: superpositions → rotations → entangling.
#      This mirrors the Clements decomposition of a Haar-random unitary —
#      the theoretically optimal reservoir for linear optics.
#      Previous order (entangling → rotations → superpositions) was reversed.
#
#   5. INPUT-IN-UNITARY encoding via parametric phase layers.
#      Instead of encoding only into the input state (fixed U · |ψ(x)⟩),
#      we inject x into phase shifters between reservoir layers.
#      This makes each layer's transformation x-dependent, creating
#      far richer polynomial entanglement between data and circuit.
#
#   6. ARCHITECTURE COMPARISON: runs 4 reservoir configs and compares them
#      so you can see exactly which physical change helps most.
#
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from merlin import MeasurementStrategy, QuantumLayer
from merlin.builder import CircuitBuilder

try:
    import perceval as pcvl
    HAS_PERCEVAL = True
except ImportError:
    HAS_PERCEVAL = False
    print("perceval not installed — circuit diagram skipped.")

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# 1) Data
# =============================================================================
iris       = load_iris()
X_raw      = iris.data.astype("float32")
y_raw      = iris.target.astype("int64")
CLASS_NAMES = iris.target_names

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.24, stratify=y_raw, random_state=SEED
)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

X_train = torch.tensor(X_tr, dtype=torch.float32)
X_test  = torch.tensor(X_te, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.long)
y_test  = torch.tensor(y_te, dtype=torch.long)

# =============================================================================
# 2) Reservoir builder — physically motivated design
#
#   Block order: superpositions (BS) → phase shifts → entangling
#   This is the Clements / Reck decomposition order for a Haar-random unitary.
#   Theoretical result (Jing et al., 2017): random unitary matrices sampled
#   this way cover the full unitary group uniformly (Haar measure), giving
#   the richest possible linear-optical reservoir.
# =============================================================================
N_MODES   = 4
DEPTH     = 5

def build_reservoir(
    n_photons:    int   = 3,           # FIX: use 3 or 4, not 2
    encoding_scale: float = np.pi/3,   # FIX: cover full Hilbert space
    measurement:  MeasurementStrategy = None,  # defaults to probs() below
    use_reuploading: bool = True,
    input_in_unitary: bool = True,     # FIX: encode into unitary, not just state
    seed: int = SEED,
):
    """
    Build a photonic reservoir with physically motivated choices.

    Parameters
    ----------
    n_photons        : int   — more photons = higher-degree polynomial features
    encoding_scale   : float — maps input values to rotation angles; pi/3
                               covers ±3σ standardized data over full 2π
    measurement      : MeasurementStrategy — MeasurementStrategy.probs() (recommended)
                       or MeasurementStrategy.AMPLITUDES
    use_reuploading  : bool  — re-inject input between reservoir layers
    input_in_unitary : bool  — if True, data modulates phase shifters inside
                               the circuit (input-in-unitary); if False, data
                               only enters as the initial state (input-in-state)
    seed             : int   — controls random circuit weights
    """
    if measurement is None:
        measurement = MeasurementStrategy.probs()   # FIX: was PHOTON_COUNTING

    torch.manual_seed(seed)
    np.random.seed(seed)

    builder = CircuitBuilder(n_modes=N_MODES)
    n_enc_blocks = 0

    def enc():
        nonlocal n_enc_blocks
        # ALL encoding blocks share the same name "enc".
        # MerLin counts nb_input_tensor by unique names, giving nb_input_tensor=1
        # regardless of how many times we re-upload. The layer broadcasts the
        # single (batch, N_MODES) input to every encoding block automatically.
        builder.add_angle_encoding(
            modes=list(range(N_MODES)),
            name="enc",
            scale=encoding_scale,
        )
        n_enc_blocks += 1

    def reservoir_block(idx):
        # Clements order: BS -> phase -> entangling
        # trainable=True registers weights as nn.Parameters inside QuantumLayer.
        # trainable=False exposes them as external inputs (causes extra tensor error).
        builder.add_superpositions(depth=1, trainable=True, name=f"bs_{idx}")
        builder.add_rotations(trainable=True,               name=f"phi_{idx}")
        builder.add_entangling_layer(trainable=True,        name=f"ent_{idx}")

    # Initial encoding
    enc()

    for d in range(DEPTH):
        if input_in_unitary and d > 0:
            enc()   # re-upload between layers (input-in-unitary)

        reservoir_block(d)

        if use_reuploading and not input_in_unitary and d < DEPTH - 1:
            enc()   # re-upload after each reservoir block (input-in-state)

    # input_size = N_MODES (raw feature count, not multiplied by n_enc_blocks).
    # nb_input_tensor=1 because all enc blocks share the same name.
    input_size = N_MODES

    layer = QuantumLayer(
        input_size=input_size,
        builder=builder,
        n_photons=n_photons,
        measurement_strategy=measurement,
    )
    for p in layer.parameters():
        p.requires_grad_(False)

    return layer, n_enc_blocks


# =============================================================================
# 3) Perceval Circuit Visualization
#    Tries common MerLin API patterns to find the underlying Perceval circuit.
# =============================================================================
def visualize_circuit(builder, title="Photonic Reservoir Circuit"):
    if not HAS_PERCEVAL:
        print("Perceval not available for circuit visualization.")
        return

    circuit = None
    # Try common MerLin attribute names for the underlying Perceval circuit
    for attr in ["circuit", "_circuit", "perceval_circuit", "_perceval_circuit"]:
        if hasattr(builder, attr):
            circuit = getattr(builder, attr)
            break
    if circuit is None:
        for method in ["build_perceval_circuit", "get_circuit", "to_perceval"]:
            if hasattr(builder, method):
                try:
                    circuit = getattr(builder, method)()
                    break
                except Exception:
                    pass

    if circuit is None:
        print("Could not extract Perceval circuit from builder.")
        print("Available builder attributes:", [a for a in dir(builder) if not a.startswith("__")])
        return

    try:
        fig = pcvl.pdisplay(circuit, output_format=pcvl.Format.MPLOT)
        if fig:
            fig.suptitle(title, fontsize=10, fontweight="bold")
            fig.tight_layout()
            fig.savefig("circuit_diagram.png", dpi=150, bbox_inches="tight")
            plt.show()
            print("Circuit saved as circuit_diagram.png")
    except Exception as e:
        print(f"pdisplay failed: {e}")
        print("Try: pcvl.pdisplay(builder.circuit) manually in your notebook.")


# =============================================================================
# 4) Feature extraction helpers
# =============================================================================
def extract_features(reservoir, X):
    # Pass X directly (batch, N_MODES=4). No tiling needed — all encoding
    # blocks share the name "enc" so MerLin broadcasts this single tensor
    # to every re-upload block automatically.
    with torch.no_grad():
        q = reservoir(X)
    if torch.is_complex(q):
        feat = torch.cat([q.real, q.imag], dim=1).float()
    else:
        feat = q.float()
    return feat


# =============================================================================
# 5) Training + evaluation helpers
# =============================================================================
criterion = nn.CrossEntropyLoss()

def train_model(model, X_feat, y, lr=0.005, epochs=2000, patience=80, label=""):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss, wait, best_state = float("inf"), 0, None
    history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_feat)
        loss   = criterion(logits, y)
        loss.backward(); opt.step(); sched.step()

        lv = loss.item()
        av = (logits.argmax(1) == y).float().mean().item()
        history["loss"].append(lv)
        history["acc"].append(av)

        if lv < best_loss - 1e-5:
            best_loss, wait = lv, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return history

def evaluate(model, X_feat, y):
    model.eval()
    with torch.no_grad():
        pred = model(X_feat).argmax(1).numpy()
    y_np = y.numpy()
    return {
        "acc": accuracy_score(y_np, pred),
        "cm":  confusion_matrix(y_np, pred),
        "pred": pred,
    }

def make_linear(feat_dim):
    return nn.Linear(feat_dim, 3)

def make_mlp(feat_dim):
    return nn.Sequential(
        nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3)
    )


# =============================================================================
# 6) Architecture comparison
#    Run 4 reservoir configs so you can isolate which physical change helps.
#
#    Config A — v2 baseline:  n_photons=2, scale=1.0, AMPLITUDES, input-in-state
#    Config B — more photons: n_photons=3, scale=1.0, AMPLITUDES, input-in-state
#    Config C — better scale: n_photons=3, scale=pi/3, probs(),    input-in-state
#    Config D — full fix:     n_photons=3, scale=pi/3, probs(),    input-in-unitary
# =============================================================================
configs = {
    "A: baseline\n(v2)": dict(
        n_photons=2, encoding_scale=1.0,
        measurement=MeasurementStrategy.AMPLITUDES,
        use_reuploading=True, input_in_unitary=False,
    ),
    "B: +photons": dict(
        n_photons=3, encoding_scale=1.0,
        measurement=MeasurementStrategy.AMPLITUDES,
        use_reuploading=True, input_in_unitary=False,
    ),
    "C: +scale\n+PROBS": dict(                          # FIX: was PHOTON_COUNT
        n_photons=3, encoding_scale=np.pi/3,
        measurement=MeasurementStrategy.probs(),         # FIX: was PHOTON_COUNTING
        use_reuploading=True, input_in_unitary=False,
    ),
    "D: input-in\nunitary (full)": dict(
        n_photons=3, encoding_scale=np.pi/3,
        measurement=MeasurementStrategy.probs(),         # FIX: was PHOTON_COUNTING
        use_reuploading=False, input_in_unitary=True,
    ),
}

results = {}   # config → {lin_tr, lin_te, mlp_tr, mlp_te, feat_dim, hist_lin, hist_mlp}

for cfg_name, cfg_kwargs in configs.items():
    print(f"\n{'='*60}")
    print(f"  Config {cfg_name.split(chr(10))[0]}")
    print(f"{'='*60}")

    reservoir, n_enc = build_reservoir(**cfg_kwargs)
    flat_name = cfg_name.replace("\n", " ")
    print(f"  Photons={cfg_kwargs['n_photons']}  scale={cfg_kwargs['encoding_scale']:.3f}"
          f"  enc_blocks={n_enc}  output_size={reservoir.output_size}")

    # Visualize first config's circuit
    if cfg_name == "D: input-in\nunitary (full)":
        # Rebuild builder for visualization (build_reservoir doesn't expose it)
        b_vis = CircuitBuilder(n_modes=N_MODES)
        b_vis.add_angle_encoding(modes=list(range(N_MODES)), name="enc_0", scale=np.pi/3)
        for d in range(DEPTH):
            if d > 0:
                b_vis.add_angle_encoding(modes=list(range(N_MODES)), name=f"enc_inU_{d}", scale=np.pi/3)
            b_vis.add_superpositions(depth=1, trainable=False, name=f"bs_{d}")
            b_vis.add_rotations(trainable=False, name=f"phi_{d}")
            b_vis.add_entangling_layer(trainable=False, name=f"ent_{d}")
        visualize_circuit(b_vis, title=f"Config D — Input-in-Unitary Reservoir  ({N_MODES} modes, 3 photons, depth {DEPTH})")

    # Extract features once
    X_tr_feat = extract_features(reservoir, X_train)
    X_te_feat = extract_features(reservoir, X_test)
    feat_dim  = X_tr_feat.shape[1]
    print(f"  Feature dim: {feat_dim}")

    # Train linear readout
    lin = make_linear(feat_dim)
    h_lin = train_model(lin, X_tr_feat, y_train, lr=0.005, label=f"{flat_name} Lin")
    r_lin_tr = evaluate(lin, X_tr_feat, y_train)
    r_lin_te = evaluate(lin, X_te_feat,  y_test)

    # Train MLP readout
    mlp = make_mlp(feat_dim)
    h_mlp = train_model(mlp, X_tr_feat, y_train, lr=0.001, label=f"{flat_name} MLP")
    r_mlp_tr = evaluate(mlp, X_tr_feat, y_train)
    r_mlp_te = evaluate(mlp, X_te_feat,  y_test)

    print(f"  Linear  → train {r_lin_tr['acc']:.4f} | test {r_lin_te['acc']:.4f}")
    print(f"  MLP     → train {r_mlp_tr['acc']:.4f} | test {r_mlp_te['acc']:.4f}")

    gap = r_mlp_te["acc"] - r_lin_te["acc"]
    if gap < 0.03:
        print(f"  ✓ Linear ≈ MLP (gap={gap:+.3f}) — reservoir is doing nonlinear work")
    else:
        print(f"  ⚠ MLP >> Linear (gap={gap:+.3f}) — reservoir not rich enough for linear separation")

    results[cfg_name] = dict(
        lin_tr=r_lin_tr["acc"], lin_te=r_lin_te["acc"],
        mlp_tr=r_mlp_tr["acc"], mlp_te=r_mlp_te["acc"],
        feat_dim=feat_dim,
        hist_lin=h_lin, hist_mlp=h_mlp,
        cm_lin=r_lin_te["cm"], cm_mlp=r_mlp_te["cm"],
        X_tr_feat=X_tr_feat,
    )


# =============================================================================
# 7) Classical baselines
# =============================================================================
print("\n── Classical baselines ──")

lr_clf = LogisticRegression(max_iter=1000, random_state=SEED)
lr_clf.fit(X_tr, y_tr)
acc_logreg = accuracy_score(y_te, lr_clf.predict(X_te))

classic_mlp = make_mlp(4)
# swap first layer
classic_mlp[0] = nn.Linear(4, 64)
train_model(classic_mlp, X_train, y_train, lr=0.001, label="ClassicMLP")
acc_cmlp = evaluate(classic_mlp, X_test, y_test)["acc"]

gen = torch.Generator().manual_seed(SEED)
best_feat_dim = results[list(results.keys())[-1]]["feat_dim"]
rand_proj = torch.randn(4, best_feat_dim, generator=gen)
rand_proj /= rand_proj.norm(dim=0, keepdim=True).clamp(min=1e-8)
rand_lin = make_linear(best_feat_dim)
train_model(rand_lin, X_train @ rand_proj, y_train, lr=0.005, label="RandProj")
acc_rand = evaluate(rand_lin, X_test @ rand_proj, y_test)["acc"]

print(f"  LogisticRegression : {acc_logreg:.4f}")
print(f"  Classical MLP      : {acc_cmlp:.4f}")
print(f"  Random Projection  : {acc_rand:.4f}")


# =============================================================================
# 8) Visualization Dashboard
# =============================================================================
plt.rcParams.update({
    "font.family": "monospace",
    "axes.facecolor": "#0d1117", "figure.facecolor": "#0a0c12",
    "axes.edgecolor": "#2a2d3e", "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b",    "ytick.color": "#64748b",
    "text.color": "#e2e8f0",     "grid.color": "#1a1d2e",
    "grid.linewidth": 0.6,
})

C = dict(A="#64748b", B="#f59e0b", C="#a78bfa", D="#00e5ff")
cfg_keys  = list(results.keys())
cfg_short = ["A: baseline", "B: +photons", "C: +scale/probs", "D: input-in-U"]

fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor("#0a0c12")
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

fig.text(0.5, 0.977, "QRC Physics Improvement Study — MerLin · Iris",
         ha="center", va="top", fontsize=16, fontweight="bold", color="#e2e8f0")
fig.text(0.5, 0.960,
         "A: baseline (n_ph=2, scale=1, AMPLITUDES)  →  "
         "B: +photons  →  C: +scale+PROBS  →  D: input-in-unitary",
         ha="center", va="top", fontsize=9, color="#64748b")

# ── Row 0: Linear readout test accuracy across all configs ────────────────────
ax0 = fig.add_subplot(gs[0, :2])
lin_accs = [results[k]["lin_te"] for k in cfg_keys]
mlp_accs = [results[k]["mlp_te"] for k in cfg_keys]
x_pos    = np.arange(len(cfg_keys))
w = 0.35
bars_lin = ax0.bar(x_pos - w/2, lin_accs, width=w,
                   color=[C[k[0]] for k in ["A","B","C","D"]], label="Linear readout", alpha=0.9)
bars_mlp = ax0.bar(x_pos + w/2, mlp_accs, width=w,
                   color=[C[k[0]] for k in ["A","B","C","D"]], label="MLP readout", alpha=0.45)
for bar, val in zip(list(bars_lin)+list(bars_mlp), lin_accs+mlp_accs):
    ax0.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#e2e8f0")
ax0.axhline(acc_logreg, color="#ef4444", lw=1, ls="--", label=f"LogReg {acc_logreg:.3f}")
ax0.axhline(acc_rand,   color="#475569", lw=1, ls=":",  label=f"RandProj {acc_rand:.3f}")
ax0.set_xticks(x_pos); ax0.set_xticklabels(cfg_short, fontsize=8)
ax0.set_ylim(0, 1.15); ax0.set_ylabel("Test Accuracy")
ax0.set_title("Config Comparison — Linear vs MLP Readout", fontsize=11, color="#e2e8f0", pad=8)
ax0.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8, ncol=2)
ax0.grid(axis="y", alpha=0.3); ax0.set_facecolor("#0d1117")

# ── Row 0 right: Linear vs MLP gap (key diagnostic) ──────────────────────────
ax0b = fig.add_subplot(gs[0, 2:])
gaps = [results[k]["mlp_te"] - results[k]["lin_te"] for k in cfg_keys]
bar_colors_gap = ["#22c55e" if g < 0.03 else "#ef4444" for g in gaps]
gb = ax0b.bar(x_pos, gaps, color=bar_colors_gap, width=0.5, alpha=0.85)
for bar, val in zip(gb, gaps):
    ax0b.text(bar.get_x()+bar.get_width()/2, val+0.002, f"{val:+.3f}",
              ha="center", va="bottom", fontsize=9, fontweight="bold", color="#e2e8f0")
ax0b.axhline(0.03, color="#f59e0b", lw=1, ls="--", label="Gap=0.03 threshold")
ax0b.axhline(0,    color="#2a2d3e", lw=0.8)
ax0b.set_xticks(x_pos); ax0b.set_xticklabels(cfg_short, fontsize=8)
ax0b.set_ylabel("MLP − Linear accuracy"); ax0b.set_ylim(-0.05, 0.35)
ax0b.set_title("Reservoir Quality Diagnostic\n(green = reservoir doing nonlinear work)", fontsize=10, color="#e2e8f0", pad=8)
ax0b.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8)
ax0b.grid(axis="y", alpha=0.3); ax0b.set_facecolor("#0d1117")

# ── Row 1: Training loss curves for all 4 configs (linear readout) ────────────
ax1 = fig.add_subplot(gs[1, :2])
for i, k in enumerate(cfg_keys):
    h = results[k]["hist_lin"]["loss"]
    ax1.plot(h, color=list(C.values())[i], lw=1.4, label=cfg_short[i], alpha=0.9)
ax1.set_title("Training Loss — Linear Readout", fontsize=11, color="#e2e8f0", pad=8)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy")
ax1.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8)
ax1.grid(True, alpha=0.3); ax1.set_facecolor("#0d1117")

ax1b = fig.add_subplot(gs[1, 2:])
for i, k in enumerate(cfg_keys):
    h = results[k]["hist_lin"]["acc"]
    ax1b.plot(h, color=list(C.values())[i], lw=1.4, label=cfg_short[i], alpha=0.9)
ax1b.set_title("Training Accuracy — Linear Readout", fontsize=11, color="#e2e8f0", pad=8)
ax1b.set_xlabel("Epoch"); ax1b.set_ylabel("Accuracy"); ax1b.set_ylim(0, 1.05)
ax1b.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8)
ax1b.grid(True, alpha=0.3); ax1b.set_facecolor("#0d1117")

# ── Row 2: PCA feature visualization — best config (D) vs baseline (A) ────────
for col, cfg_k in enumerate(["A: baseline\n(v2)", "D: input-in\nunitary (full)"]):
    ax = fig.add_subplot(gs[2, col*2:(col+1)*2])
    feat_np = results[cfg_k]["X_tr_feat"].numpy()
    pca2    = PCA(n_components=2)
    f2d     = pca2.fit_transform(feat_np)
    colors  = ["#00e5ff", "#a78bfa", "#f59e0b"]
    for c in range(3):
        mask = y_train.numpy() == c
        ax.scatter(f2d[mask,0], f2d[mask,1], c=colors[c],
                   label=CLASS_NAMES[c], alpha=0.75, s=45, edgecolors="none")
    var = pca2.explained_variance_ratio_.sum()
    short_name = cfg_k.split("\n")[0]
    ax.set_title(f"PCA Reservoir Features — {short_name} (var={var:.1%})",
                 fontsize=10, color="#e2e8f0", pad=8)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
    ax.legend(framealpha=0.15, facecolor="#1a1d2e", fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_facecolor("#0d1117")

# ── Row 3: Confusion matrices for best config ────────────────────────────────
best_cfg = max(results, key=lambda k: results[k]["lin_te"])
for col, (label, cm_key) in enumerate([("Linear Readout", "cm_lin"), ("MLP Readout", "cm_mlp")]):
    ax = fig.add_subplot(gs[3, col*2:(col+1)*2])
    cm = results[best_cfg][cm_key]
    cmap = "Blues" if col == 0 else "Purples"
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()*0.5 else "#64748b")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(CLASS_NAMES, rotation=20, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    best_short = best_cfg.split("\n")[0]
    acc_val = results[best_cfg]["lin_te"] if col==0 else results[best_cfg]["mlp_te"]
    ax.set_title(f"Best Config ({best_short}) — {label}  (acc={acc_val:.3f})",
                 fontsize=10, color="#e2e8f0", pad=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_facecolor("#0d1117")

plt.savefig("qrc_physics_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("\nDashboard saved as qrc_physics_dashboard.png")

# =============================================================================
# 9) Final Summary
# =============================================================================
best = max(results, key=lambda k: results[k]["lin_te"])
best_lin = results[best]["lin_te"]
best_mlp = results[best]["mlp_te"]
gap_best = best_mlp - best_lin

print("\n" + "═"*58)
print("  PHYSICS IMPROVEMENT SUMMARY")
print("═"*58)
print(f"  {'Config':<32} {'Lin':>6}  {'MLP':>6}  {'Gap':>6}")
print("─"*58)
for k in cfg_keys:
    r = results[k]
    flag = " ✓" if r["mlp_te"]-r["lin_te"] < 0.03 else "  "
    kf = k.replace("\n"," ")
    print(f"  {kf:<32} {r['lin_te']:>6.4f}  {r['mlp_te']:>6.4f}  {r['mlp_te']-r['lin_te']:>+6.3f}{flag}")
print("─"*58)
print(f"  Classical LogReg               {acc_logreg:>6.4f}")
print(f"  Classical MLP                  {acc_cmlp:>6.4f}")
print(f"  Random Projection              {acc_rand:>6.4f}")
print("═"*58)

best_short = best.replace("\n"," ")
print(f"\n  Best linear readout: {best_short}  ({best_lin:.4f})")

if best_lin > acc_rand + 0.03:
    print(f"  ✓ QRC outperforms random projection — quantum structure is contributing")
else:
    print(f"  ⚠ QRC ≈ random projection — try n_photons=4 or wider circuit")

if gap_best < 0.03:
    print(f"  ✓ Linear ≈ MLP (gap={gap_best:+.3f}) — reservoir is linearly separating classes")
else:
    print(f"  ⚠ MLP still >> Linear (gap={gap_best:+.3f})")
    print(f"    Next steps: n_photons=4, wider modes (N_MODES=6), or trainable encoding scale")
print("═"*58)
