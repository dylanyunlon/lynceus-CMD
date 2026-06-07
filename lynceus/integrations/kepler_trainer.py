"""
kepler_trainer â Training framework for Lynceus models (numpy-only).

Ported from upstream/kepler/model_trainer/trainer.py (~350 lines).
TFânumpy throughout; all gradient computation uses explicit numpy SGD.
Algorithm changes (~20%):
  - TrainerBase.train: numpy mini-batch SGD with momentum (Î²=0.9)
    and configurable weight_decay (L2) instead of tf.keras.Model.fit
  - ClassificationTrainer: softmax cross-entropy with label-smoothing (Îµ=0.1)
    instead of hard one-hot targets
  - RegressionTrainer: Huber loss (Î´=1.0) instead of plain MSE for robustness
  - NearOptimalTrainer: margin-aware contrastive ranking loss instead of
    simple pairwise comparison
  - EarlyStopping callback integrated into train() with patience parameter
"""
import abc
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_NAME_DELIMITER = "####"

JSON = Any

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[kepler_trainer] {tag}: {items}")


# ââ Utility functions ââââââââââââââââââââââââââââââââââââââââââââ

def get_parameter_column_name(index: int) -> str:
    return f"p{index}"


def get_parameter_column_names(parameter_count: int) -> List[str]:
    return [get_parameter_column_name(i) for i in range(parameter_count)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ââ Simple linear model (replaces tf.keras) ââââââââââââââââââââââ

class LinearModel:
    """A minimal linear model: y = X @ W + b, trained via numpy SGD."""

    def __init__(self, input_dim: int, output_dim: int,
                 rng_seed: Optional[int] = None):
        rng = np.random.RandomState(rng_seed)
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = rng.randn(input_dim, output_dim).astype(np.float64) * scale
        self.b = np.zeros(output_dim, dtype=np.float64)
        # Momentum buffers
        self._vW = np.zeros_like(self.W)
        self._vb = np.zeros_like(self.b)
        _dbg("LinearModel.__init__", input_dim=input_dim,
             output_dim=output_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def update(self, dW: np.ndarray, db: np.ndarray, lr: float,
               momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        """SGD step with momentum and optional L2 weight decay."""
        dW = dW + weight_decay * self.W
        self._vW = momentum * self._vW + dW
        self._vb = momentum * self._vb + db
        self.W -= lr * self._vW
        self.b -= lr * self._vb

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}


# ââ Training history âââââââââââââââââââââââââââââââââââââââââââââ


class TrainingHistory:
    """Mimics tf.keras.callbacks.History."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {"loss": []}

    def append_loss(self, loss: float) -> None:
        self.history["loss"].append(loss)

    def __repr__(self) -> str:
        n = len(self.history["loss"])
        last = self.history["loss"][-1] if n > 0 else None
        return f"TrainingHistory(epochs={n}, last_loss={last})"


# ââ Early stopping âââââââââââââââââââââââââââââââââââââââââââââââ

class EarlyStopping:
    """Stops training when loss fails to improve for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self._best: Optional[float] = None
        self._wait = 0

    def should_stop(self, current_loss: float) -> bool:
        if self._best is None or current_loss < self._best - self.min_delta:
            self._best = current_loss
            self._wait = 0
            return False
        self._wait += 1
        if self._wait >= self.patience:
            _dbg("EarlyStopping.should_stop", best=self._best,
                 current=current_loss, waited=self._wait)
            return True
        return False


# ââ Base Trainer âââââââââââââââââââââââââââââââââââââââââââââââââ

class TrainerBase(abc.ABC):
    """Base trainer class for Kepler modeling with numpy SGD.

    Manages model training over a fixed set of plans using mini-batch
    gradient descent with momentum and weight decay.
    """

    def __init__(
        self,
        metadata: JSON,
        plan_ids: List[int],
        input_dim: int,
        output_dim: int,
        rng_seed: Optional[int] = None,
    ):
        self._predicate_metadata = metadata["predicates"]
        self._plan_id_to_index = {pid: i for i, pid in enumerate(plan_ids)}
        self._plan_ids = list(plan_ids)
        self._model = LinearModel(input_dim, output_dim, rng_seed=rng_seed)
        _dbg("TrainerBase.__init__", n_plans=len(plan_ids),
             input_dim=input_dim, output_dim=output_dim)

    def apply_preprocessing(self, execution_df: pd.DataFrame) -> None:
        """Applies preprocessing to parameter columns in-place."""
        for i, predicate in enumerate(self._predicate_metadata):
            col_name = get_parameter_column_name(i)
            if col_name not in execution_df.columns:
                continue
            if predicate.get("preprocess_type") == "to_timestamp":
                execution_df[col_name] = (
                    pd.to_datetime(execution_df[col_name])
                    .apply(lambda x: x.timestamp())
                )
        _dbg("apply_preprocessing",
             n_predicates=len(self._predicate_metadata))

    def get_parameter_column_names(self) -> List[str]:
        return get_parameter_column_names(
            parameter_count=len(self._predicate_metadata))

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        sample_weight: Optional[np.ndarray] = None,
        patience: int = 5,
    ) -> TrainingHistory:
        """Train via mini-batch SGD with momentum, weight decay, early stopping.

        Args:
            x: Input features, shape (N, D).
            y: Targets (interpretation depends on subclass).
            epochs: Maximum number of passes through the data.
            batch_size: Mini-batch size.
            lr: Learning rate.
            momentum: SGD momentum coefficient.
            weight_decay: L2 regularization strength.
            sample_weight: Per-sample loss multiplier, shape (N,).
            patience: Early stopping patience (epochs without improvement).

        Returns:
            TrainingHistory with per-epoch loss.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples = x.shape[0]

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        history = TrainingHistory()
        stopper = EarlyStopping(patience=patience)
        indices = np.arange(n_samples)

        _dbg("train.start", n_samples=n_samples, epochs=epochs,
             batch_size=batch_size, lr=lr, momentum=momentum,
             weight_decay=weight_decay)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = x[batch_idx]
                yb = y[batch_idx]
                wb = sample_weight[batch_idx]

                loss, dW, db = self._compute_gradients(xb, yb, wb)
                self._model.update(dW, db, lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)
                epoch_loss += loss * len(batch_idx)
                n_batches += 1

            avg_loss = epoch_loss / n_samples
            history.append_loss(avg_loss)

            if epoch % max(1, epochs // 10) == 0:
                _dbg("train.epoch", epoch=epoch, loss=avg_loss)

            if stopper.should_stop(avg_loss):
                _dbg("train.early_stop", epoch=epoch, loss=avg_loss)
                break

        _dbg("train.done", final_loss=history.history["loss"][-1],
             total_epochs=len(history.history["loss"]))
        return history

    @abc.abstractmethod
    def _compute_gradients(
        self, xb: np.ndarray, yb: np.ndarray, wb: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute loss and gradients for a mini-batch.

        Returns:
            (loss_scalar, dW, db)
        """

    @abc.abstractmethod
    def construct_training_data(
        self, execution_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (x, y) arrays from execution results."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.forward(np.asarray(x, dtype=np.float64))


# ââ Classification Trainer âââââââââââââââââââââââââââââââââââââââ

class ClassificationTrainer(TrainerBase):
    """Predicts which plan has lowest latency via softmax classification.

    Algorithm change: label-smoothing (Îµ=0.1) on one-hot targets to improve
    calibration and reduce overconfidence.
    """

    def __init__(self, metadata: JSON, plan_ids: List[int],
                 input_dim: int, label_smoothing: float = 0.1,
                 rng_seed: Optional[int] = None):
        output_dim = len(plan_ids)
        super().__init__(metadata, plan_ids, input_dim, output_dim, rng_seed)
        self._label_smoothing = label_smoothing
        _dbg("ClassificationTrainer.__init__",
             output_dim=output_dim, label_smoothing=label_smoothing)

    def construct_training_data(
        self, execution_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build x (features) and y (best-plan indices) from latency data."""
        param_cols = self.get_parameter_column_names()
        available_cols = [c for c in param_cols if c in execution_df.columns]

        groups = execution_df.groupby(available_cols)
        x_list, y_list = [], []

        for params, group_df in groups:
            if not isinstance(params, tuple):
                params = (params,)
            features = np.array([float(p) for p in params], dtype=np.float64)

            best_idx = None
            best_latency = float("inf")
            for _, row in group_df.iterrows():
                pid = row["plan_id"]
                if pid in self._plan_id_to_index:
                    idx = self._plan_id_to_index[pid]
                    if row["latency"] < best_latency:
                        best_latency = row["latency"]
                        best_idx = idx

            if best_idx is not None:
                x_list.append(features)
                y_list.append(best_idx)

        x = np.array(x_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        _dbg("ClassificationTrainer.construct_training_data",
             n_samples=len(x_list))
        return x, y

    def _compute_gradients(
        self, xb: np.ndarray, yb: np.ndarray, wb: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Cross-entropy with label smoothing."""
        batch_size = xb.shape[0]
        num_classes = self._model.W.shape[1]
        eps = self._label_smoothing

        logits = self._model.forward(xb)  # (B, C)
        probs = _softmax(logits)  # (B, C)

        # Smoothed one-hot targets
        targets = np.full((batch_size, num_classes),
                          eps / num_classes, dtype=np.float64)
        for i in range(batch_size):
            targets[i, int(yb[i])] = 1.0 - eps + eps / num_classes

        # Cross-entropy loss
        log_probs = np.log(np.clip(probs, 1e-12, None))
        per_sample_loss = -np.sum(targets * log_probs, axis=1)
        weighted_loss = per_sample_loss * wb
        loss = np.mean(weighted_loss)

        # Gradient of softmax cross-entropy
        d_logits = (probs - targets) / batch_size  # (B, C)
        for i in range(batch_size):
            d_logits[i] *= wb[i]

        dW = xb.T @ d_logits
        db = np.sum(d_logits, axis=0)

        return float(loss), dW, db

    def predict_best_plan(self, x: np.ndarray) -> np.ndarray:
        """Return predicted best plan index for each input row."""
        logits = self.predict(x)
        return np.argmax(logits, axis=1)


# ââ Regression Trainer âââââââââââââââââââââââââââââââââââââââââââ

class RegressionTrainer(TrainerBase):
    """Predicts latency of each plan via regression.

    Algorithm change: Huber loss (Î´=1.0) instead of MSE for robustness
    to outlier latencies.
    """

    def __init__(self, metadata: JSON, plan_ids: List[int],
                 input_dim: int, huber_delta: float = 1.0,
                 rng_seed: Optional[int] = None):
        output_dim = len(plan_ids)
        super().__init__(metadata, plan_ids, input_dim, output_dim, rng_seed)
        self._huber_delta = huber_delta
        _dbg("RegressionTrainer.__init__",
             output_dim=output_dim, huber_delta=huber_delta)

    def construct_training_data(
        self, execution_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build x (features) and y (latency vector per plan) from data."""
        param_cols = self.get_parameter_column_names()
        available_cols = [c for c in param_cols if c in execution_df.columns]
        num_plans = len(self._plan_ids)

        groups = execution_df.groupby(available_cols)
        x_list, y_list = [], []

        for params, group_df in groups:
            if not isinstance(params, tuple):
                params = (params,)
            features = np.array([float(p) for p in params], dtype=np.float64)

            latencies = np.full(num_plans, np.nan, dtype=np.float64)
            for _, row in group_df.iterrows():
                pid = row["plan_id"]
                if pid in self._plan_id_to_index:
                    idx = self._plan_id_to_index[pid]
                    latencies[idx] = row["latency"]

            # Fill missing with column median (robust fallback)
            if not np.all(np.isnan(latencies)):
                median_val = np.nanmedian(latencies)
                latencies = np.where(np.isnan(latencies), median_val, latencies)
                x_list.append(features)
                y_list.append(latencies)

        x = np.array(x_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        _dbg("RegressionTrainer.construct_training_data",
             n_samples=len(x_list), n_plans=num_plans)
        return x, y

    def _compute_gradients(
        self, xb: np.ndarray, yb: np.ndarray, wb: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Huber loss gradient."""
        batch_size = xb.shape[0]
        delta = self._huber_delta

        preds = self._model.forward(xb)  # (B, P)
        residuals = preds - yb  # (B, P)
        abs_res = np.abs(residuals)

        # Huber loss per element
        huber = np.where(abs_res <= delta,
                         0.5 * residuals ** 2,
                         delta * (abs_res - 0.5 * delta))
        per_sample = np.mean(huber, axis=1)  # (B,)
        loss = float(np.mean(per_sample * wb))

        # Huber gradient: linear in quadratic region, constant in linear region
        d_preds = np.where(abs_res <= delta,
                           residuals,
                           delta * np.sign(residuals))
        d_preds /= (batch_size * yb.shape[1])
        for i in range(batch_size):
            d_preds[i] *= wb[i]

        dW = xb.T @ d_preds
        db = np.sum(d_preds, axis=0)

        return loss, dW, db


# ââ Near-Optimal (Ranking) Trainer âââââââââââââââââââââââââââââââ

class NearOptimalTrainer(TrainerBase):
    """Trains a model to rank plans by latency proximity to optimal.

    Algorithm change: margin-aware contrastive ranking loss â
    for each sample, the score of the best plan must exceed the score of
    every other plan by at least a margin proportional to the latency gap.
    This replaces simple pairwise comparison with an adaptive margin that
    scales with the latency difference.
    """

    def __init__(self, metadata: JSON, plan_ids: List[int],
                 input_dim: int, margin_scale: float = 0.5,
                 near_optimal_threshold: float = 1.05,
                 rng_seed: Optional[int] = None):
        output_dim = len(plan_ids)
        super().__init__(metadata, plan_ids, input_dim, output_dim, rng_seed)
        self._margin_scale = margin_scale
        self._threshold = near_optimal_threshold
        _dbg("NearOptimalTrainer.__init__",
             output_dim=output_dim, margin_scale=margin_scale,
             threshold=near_optimal_threshold)

    def construct_training_data(
        self, execution_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build x and y where y contains per-plan latency vectors."""
        param_cols = self.get_parameter_column_names()
        available_cols = [c for c in param_cols if c in execution_df.columns]
        num_plans = len(self._plan_ids)

        groups = execution_df.groupby(available_cols)
        x_list, y_list = [], []

        for params, group_df in groups:
            if not isinstance(params, tuple):
                params = (params,)
            features = np.array([float(p) for p in params], dtype=np.float64)

            latencies = np.full(num_plans, np.nan, dtype=np.float64)
            for _, row in group_df.iterrows():
                pid = row["plan_id"]
                if pid in self._plan_id_to_index:
                    idx = self._plan_id_to_index[pid]
                    latencies[idx] = row["latency"]

            if not np.all(np.isnan(latencies)):
                fill = np.nanmax(latencies) * 2.0
                latencies = np.where(np.isnan(latencies), fill, latencies)
                x_list.append(features)
                y_list.append(latencies)

        x = np.array(x_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        _dbg("NearOptimalTrainer.construct_training_data",
             n_samples=len(x_list), n_plans=num_plans)
        return x, y

    def _compute_gradients(
        self, xb: np.ndarray, yb: np.ndarray, wb: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Margin-aware contrastive ranking loss.

        For each sample, the best plan (lowest latency) should have the
        highest score. A margin proportional to (latency_j - latency_best)
        is enforced between score_best and score_j.
        """
        batch_size = xb.shape[0]
        num_plans = yb.shape[1]

        scores = self._model.forward(xb)  # (B, P) â higher = better
        best_plans = np.argmin(yb, axis=1)  # (B,) â lowest latency = best
        best_latencies = yb[np.arange(batch_size), best_plans]  # (B,)

        total_loss = 0.0
        d_scores = np.zeros_like(scores)

        for i in range(batch_size):
            bp = best_plans[i]
            s_best = scores[i, bp]
            lat_best = best_latencies[i]

            for j in range(num_plans):
                if j == bp:
                    continue
                # Adaptive margin: scales with latency gap
                margin = self._margin_scale * (yb[i, j] - lat_best) / max(
                    lat_best, 1e-9)
                # Hinge: max(0, margin - (s_best - s_j))
                violation = margin - (s_best - scores[i, j])
                if violation > 0:
                    total_loss += wb[i] * violation
                    d_scores[i, bp] -= wb[i]  # push s_best up
                    d_scores[i, j] += wb[i]   # push s_j down

        loss = total_loss / (batch_size * max(num_plans - 1, 1))
        d_scores /= (batch_size * max(num_plans - 1, 1))

        dW = xb.T @ d_scores
        db = np.sum(d_scores, axis=0)

        return float(loss), dW, db

    def predict_plan_ranking(self, x: np.ndarray) -> np.ndarray:
        """Return plan indices sorted by predicted score (best first)."""
        scores = self.predict(x)
        return np.argsort(-scores, axis=1)

    def predict_near_optimal_set(self, x: np.ndarray) -> List[List[int]]:
        """Return set of plan indices within threshold of predicted best."""
        scores = self.predict(x)
        results = []
        for row in scores:
            best_score = np.max(row)
            # Plans whose score is within threshold fraction of best
            cutoff = best_score / self._threshold if best_score > 0 else -np.inf
            near_opt = [int(j) for j in range(len(row)) if row[j] >= cutoff]
            results.append(near_opt)
        _dbg("predict_near_optimal_set", n_inputs=len(x),
             avg_set_size=np.mean([len(s) for s in results]))
        return results
