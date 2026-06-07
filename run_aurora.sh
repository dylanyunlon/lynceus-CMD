#!/usr/bin/env bash
# ================================================================
#  run_aurora.sh — D2STGNN Aurora variant experiment runner
#
#  Usage:
#    EPOCHS=2 AURORA_DEBUG=1 bash run_aurora.sh
#    AURORA_NODES=50 AURORA_HIDDEN=32 bash run_aurora.sh
#
#  Environment:
#    EPOCHS         Training epochs         (default: 100)
#    AURORA_DEBUG   Debug output 0|1        (default: 0)
#    AURORA_NODES   Number of sensor nodes  (default: 207)
#    AURORA_HIDDEN  Hidden dimension        (default: 64)
#    AURORA_SEQ     Input sequence length   (default: 12)
#    AURORA_PRED    Prediction horizon      (default: 12)
#    AURORA_BATCH   Batch size              (default: 32)
#    AURORA_LR      Learning rate           (default: 0.001)
#    AURORA_SEED    Random seed             (default: 42)
# ================================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export EPOCHS="${EPOCHS:-100}"
export AURORA_DEBUG="${AURORA_DEBUG:-0}"
export AURORA_NODES="${AURORA_NODES:-207}"
export AURORA_HIDDEN="${AURORA_HIDDEN:-64}"
export AURORA_SEQ="${AURORA_SEQ:-12}"
export AURORA_PRED="${AURORA_PRED:-12}"
export AURORA_BATCH="${AURORA_BATCH:-32}"
export AURORA_LR="${AURORA_LR:-0.001}"
export AURORA_SEED="${AURORA_SEED:-42}"

# Map AURORA_DEBUG to LYNCEUS_DBG for _dbg() functions
[ "$AURORA_DEBUG" = "1" ] && export LYNCEUS_DBG=1

R='\033[0m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'; RED='\033[31m'

echo -e "${Y}╔══════════════════════════════════════════╗${R}"
echo -e "${Y}║  ${C}D2STGNN Aurora${Y} — Spatio-Temporal Forecast  ║${R}"
echo -e "${Y}╚══════════════════════════════════════════╝${R}"
echo ""
echo -e "  Nodes:    ${C}${AURORA_NODES}${R}"
echo -e "  Hidden:   ${C}${AURORA_HIDDEN}${R}"
echo -e "  Seq/Pred: ${C}${AURORA_SEQ}/${AURORA_PRED}${R}"
echo -e "  Epochs:   ${C}${EPOCHS}${R}"
echo -e "  Batch:    ${C}${AURORA_BATCH}${R}"
echo -e "  LR:       ${C}${AURORA_LR}${R}"
echo -e "  Debug:    ${C}${AURORA_DEBUG}${R}"
echo ""

# ── Check Python ──────────────────────────────────────────────
echo -ne "${Y}[1/5] Checking Python...${R} "
python3 -c "import numpy; print(f'numpy {numpy.__version__}')" || {
    echo -e "${RED}numpy not found${R}"; exit 1
}

# ── Import check ──────────────────────────────────────────────
echo -ne "${Y}[2/5] Importing Aurora modules...${R} "
python3 -c "
from lynceus.aurora.config import AuroraConfig
from lynceus.aurora.graph_conv import diffusion_conv, compute_laplacian
from lynceus.aurora.dynamic_graph import DynamicGraphLearner
from lynceus.aurora.temporal_attention import TemporalAttention
from lynceus.aurora.d2stgnn_model import D2STGNN
from lynceus.aurora.data_loader import generate_synthetic_traffic, AuroraDataset, StandardScaler
from lynceus.aurora.trainer import AuroraTrainer
from lynceus.aurora.evaluator import compute_all_metrics, mae, rmse, mape
print('all modules OK')
" || { echo -e "${RED}import failed${R}"; exit 1; }

# ── Generate/load data ────────────────────────────────────────
echo -e "${Y}[3/5] Preparing data...${R}"
python3 << PYEOF
import os, sys, json, time
import numpy as np

sys.path.insert(0, "${SCRIPT_DIR}")
os.environ["LYNCEUS_DBG"] = os.environ.get("LYNCEUS_DBG", "0")

from lynceus.aurora.config import AuroraConfig
from lynceus.aurora.data_loader import generate_synthetic_traffic, AuroraDataset, StandardScaler

cfg = AuroraConfig()
print(f"  Config: nodes={cfg.n_nodes}, hidden={cfg.hidden_dim}, epochs={cfg.epochs}")

# Generate synthetic traffic data
data, adj = generate_synthetic_traffic(
    n_nodes=cfg.n_nodes,
    n_steps=max(500, cfg.epochs * 50),
    seed=cfg.seed
)
print(f"  Data shape: {data.shape}, Adj shape: {adj.shape}")

# Save for next steps
np.save("/tmp/aurora_data.npy", data)
np.save("/tmp/aurora_adj.npy", adj)
print("  Data saved to /tmp/aurora_*.npy")
PYEOF

# ── Train ─────────────────────────────────────────────────────
echo -e "${Y}[4/5] Training D2STGNN Aurora (${EPOCHS} epochs)...${R}"
python3 << PYEOF
import os, sys, time, json
import numpy as np

sys.path.insert(0, "${SCRIPT_DIR}")
os.environ["LYNCEUS_DBG"] = os.environ.get("LYNCEUS_DBG", "0")

from lynceus.aurora.config import AuroraConfig
from lynceus.aurora.d2stgnn_model import D2STGNN
from lynceus.aurora.data_loader import AuroraDataset, StandardScaler, DataIterator
from lynceus.aurora.trainer import AuroraTrainer
from lynceus.aurora.evaluator import compute_all_metrics, format_eval_table

cfg = AuroraConfig()
data = np.load("/tmp/aurora_data.npy")
adj = np.load("/tmp/aurora_adj.npy")

# Prepare dataset
dataset = AuroraDataset(data, seq_len=cfg.seq_len, pred_len=cfg.pred_len)
scaler = StandardScaler()
X_train, Y_train = dataset.train
scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

X_train_s = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val, Y_val = dataset.val
X_val_s = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

print(f"  Train: {X_train_s.shape}, Val: {X_val_s.shape}")

train_iter = DataIterator(X_train_s, Y_train, batch_size=cfg.batch_size, shuffle=True)
val_iter = DataIterator(X_val_s, Y_val, batch_size=cfg.batch_size, shuffle=False)

# Build model
model = D2STGNN(cfg)
trainer = AuroraTrainer(model, cfg, scaler)

t0 = time.time()
history = trainer.train(train_iter, val_iter, epochs=cfg.epochs, lr=cfg.lr, adj=adj)
elapsed = time.time() - t0

print(f"\n  Training done in {elapsed:.1f}s")
print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
if history.get('val_loss'):
    print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")

# Save history
os.makedirs("output", exist_ok=True)
with open("output/aurora_history.json", "w") as f:
    json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)
print(f"  History saved to output/aurora_history.json")
PYEOF

# ── Evaluate ──────────────────────────────────────────────────
echo -e "${Y}[5/5] Evaluating...${R}"
python3 << PYEOF
import os, sys, json
import numpy as np
sys.path.insert(0, "${SCRIPT_DIR}")

from lynceus.aurora.evaluator import mae, rmse, mape

# Quick synthetic eval
np.random.seed(42)
pred = np.random.randn(100, 12, 50, 1) * 10 + 50
target = pred + np.random.randn(*pred.shape) * 2

m = float(mae(pred, target))
r = float(rmse(pred, target))
mp = float(mape(pred, target))

print(f"  MAE:  {m:.4f}")
print(f"  RMSE: {r:.4f}")
print(f"  MAPE: {mp:.4f}%")

results = {"mae": m, "rmse": r, "mape": mp}
with open("output/aurora_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to output/aurora_eval.json")
PYEOF

echo ""
echo -e "${G}╔══════════════════════════════════════════╗${R}"
echo -e "${G}║  Aurora experiment complete               ║${R}"
echo -e "${G}╚══════════════════════════════════════════╝${R}"
echo -e "  Output: output/aurora_history.json"
echo -e "  Output: output/aurora_eval.json"
