# =============================================================================
# Quantum Reservoir Computing — MerLin / Iris
# Corrected & Optimized Script
#
# Changes vs original:
#   1. Pre-compute frozen reservoir + projection features ONCE before training
#   2. Reproducible projection (explicit torch.Generator seed)
#   3. Complex vs real output detection — avoids wasted zero imag features
#   4. Full feature vector passed to readout (no lossy random compression)
#   5. CosineAnnealingLR scheduler on the linear readout
#   6. Patience-based early stopping
#   7. Ablation helper — sanity-checks the reservoir is doing real work
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)

# MerLin imports — adjust to your actual package structure
from merlin import CircuitBuilder, QuantumLayer, MeasurementStrategy

# -----------------------------------------------------------------------------
# 0) Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# 1) Data — same preprocessing as original
# -----------------------------------------------------------------------------
iris = load_iris()
X = iris.data.astype("float32")
y = iris.target.astype("int64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.24, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

# -----------------------------------------------------------------------------
# 2) Photonic Reservoir — same architecture as original
# -----------------------------------------------------------------------------
n_modes          = 4
n_photons        = 3
reservoir_layers = 2

builder = CircuitBuilder(n_modes=n_modes)

builder.add_angle_encoding(
    modes=list(range(n_modes)),
    name="input",
    scale=1.0
)

for l in range(reservoir_layers):
    builder.add_superpositions(depth=1, trainable=False, name=f"mix_{l}")
    builder.add_entangling_layer(trainable=False, name=f"ent_{l}")
    builder.add_rotations(trainable=False, name=f"rot_{l}")

reservoir = QuantumLayer(
    input_size=4,
    builder=builder,
    n_photons=n_photons,
    measurement_strategy=MeasurementStrategy.AMPLITUDES
)

# Freeze all reservoir parameters (QRC requirement)
for p in reservoir.parameters():
    p.requires_grad_(False)

print(f"Reservoir output_size: {reservoir.output_size}")

# -----------------------------------------------------------------------------
# 3) Feature extraction — run ONCE, not every epoch
#
#    FIX 1 (Critical): reservoir + projection are frozen, so there is no reason
#    to call them inside the training loop. Pre-compute and cache.
#
#    FIX 2 (Conceptual): removed the random projection that compressed features
#    back down to n_modes=4. That discarded most of what the quantum circuit
#    computed. The readout now trains on the full Fock-space feature vector.
#
#    FIX 3 (Correctness): detect whether MerLin returns complex or real tensors.
#    AMPLITUDES can be real-valued depending on the MerLin version/backend.
#    Using torch.imag on a real tensor produces all zeros — wasted dimensions.
# -----------------------------------------------------------------------------

def extract_features(reservoir: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Run the frozen reservoir once and return a real-valued feature matrix.
    Handles both complex and real reservoir outputs automatically.
    """
    with torch.no_grad():
        q = reservoir(X)

    if torch.is_complex(q):
        # Complex amplitudes — concatenate real and imaginary parts
        features = torch.cat([q.real, q.imag], dim=1).float()
        print(f"Complex output detected. Feature dim: {features.shape[1]} "
              f"(2 × {q.shape[1]})")
    else:
        # Real output — use as-is; no point concatenating zeros
        features = q.float()
        print(f"Real output detected. Feature dim: {features.shape[1]}")

    return features


print("\nExtracting features (runs reservoir once per split)...")
X_train_feat = extract_features(reservoir, X_train)  # (n_train, feat_dim)
X_test_feat  = extract_features(reservoir, X_test)   # (n_test,  feat_dim)

feat_dim = X_train_feat.shape[1]
print(f"Training features: {X_train_feat.shape}")
print(f"Test features    : {X_test_feat.shape}")

# -----------------------------------------------------------------------------
# 4) Ablation check — does the quantum circuit add value?
#
#    Replace the reservoir with a random linear projection of the same
#    input dimensionality and compare accuracy. If performance is similar,
#    the circuit is not contributing meaningful nonlinearity and the encoding
#    strategy should be revisited before scaling up.
#
#    Set RUN_ABLATION = True to enable.
# -----------------------------------------------------------------------------
RUN_ABLATION = True

if RUN_ABLATION:
    print("\n--- Ablation: random linear baseline ---")
    gen_abl = torch.Generator().manual_seed(SEED)
    rand_proj = torch.randn(X_train.shape[1], feat_dim, generator=gen_abl)
    rand_proj /= rand_proj.norm(dim=0, keepdim=True).clamp(min=1e-8)

    X_train_rand = X_train @ rand_proj
    X_test_rand  = X_test  @ rand_proj

    lin_abl = nn.Linear(feat_dim, 3)
    opt_abl = torch.optim.Adam(lin_abl.parameters(), lr=0.01)
    crit    = nn.CrossEntropyLoss()

    for _ in range(500):
        opt_abl.zero_grad()
        loss_abl = crit(lin_abl(X_train_rand), y_train)
        loss_abl.backward()
        opt_abl.step()

    with torch.no_grad():
        abl_pred = lin_abl(X_test_rand).argmax(1).numpy()
    abl_acc = accuracy_score(y_test.numpy(), abl_pred)
    print(f"Ablation (random projection) test accuracy: {abl_acc:.3f}")
    print("--- End ablation ---\n")

# -----------------------------------------------------------------------------
# 5) Linear readout model — trains on pre-computed features
# -----------------------------------------------------------------------------
linear_readout = nn.Linear(feat_dim, 3)

# FIX 4: CosineAnnealingLR — decays LR smoothly to avoid overshooting
optimizer = torch.optim.Adam(linear_readout.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs    = 500
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# FIX 5: Early stopping — stop when loss plateaus
patience    = 40
best_loss   = float("inf")
wait        = 0
best_state  = None

# -----------------------------------------------------------------------------
# 6) Training loop — pure linear algebra, no quantum circuit calls
# -----------------------------------------------------------------------------
print("Training linear readout on pre-computed features...")

for epoch in range(epochs):
    linear_readout.train()
    optimizer.zero_grad()

    logits = linear_readout(X_train_feat)
    loss   = criterion(logits, y_train)

    loss.backward()
    optimizer.step()
    scheduler.step()

    # Early stopping
    loss_val = loss.item()
    if loss_val < best_loss - 1e-4:
        best_loss  = loss_val
        wait       = 0
        best_state = {k: v.clone() for k, v in linear_readout.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    if epoch % 50 == 0:
        train_acc = (logits.argmax(1) == y_train).float().mean().item()
        print(f"Epoch {epoch:3d} | Loss: {loss_val:.4f} | "
              f"TrainAcc: {train_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.5f}")

# Restore best weights
if best_state is not None:
    linear_readout.load_state_dict(best_state)
    print("Restored best model weights.")

# -----------------------------------------------------------------------------
# 7) Evaluation
# -----------------------------------------------------------------------------
linear_readout.eval()
with torch.no_grad():
    train_pred = linear_readout(X_train_feat).argmax(1).numpy()
    test_pred  = linear_readout(X_test_feat).argmax(1).numpy()

y_train_np = y_train.numpy()
y_test_np  = y_test.numpy()

train_acc = accuracy_score(y_train_np, train_pred)
test_acc  = accuracy_score(y_test_np,  test_pred)
precision = precision_score(y_test_np, test_pred, average="macro", zero_division=0)
recall    = recall_score(y_test_np,    test_pred, average="macro", zero_division=0)
cm        = confusion_matrix(y_test_np, test_pred)

print("\n" + "=" * 40)
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test  Accuracy : {test_acc:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print("=" * 40)

if RUN_ABLATION:
    print(f"\nAblation baseline  : {abl_acc:.4f}")
    print(f"QRC test accuracy  : {test_acc:.4f}")
    delta = test_acc - abl_acc
    if delta > 0.02:
        print(f"✓ Quantum reservoir adds value (+{delta:.3f} over random projection)")
    elif delta > -0.02:
        print(f"⚠ Marginal difference ({delta:+.3f}) — consider revisiting encoding strategy")
    else:
        print(f"✗ Random projection outperforms reservoir ({delta:+.3f}) — encoding needs rethinking")
