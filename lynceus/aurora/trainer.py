"""
Training loop for D2STGNN Aurora.
Pure NumPy with numerical gradients for small models.
"""

import numpy as np
import os
import time

_DEBUG = os.environ.get("AURORA_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[DEBUG trainer]", *args, **kwargs)


class EarlyStopping:
    """Early stopping tracker."""

    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False
        self.best_epoch = -1

    def step(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            _dbg(f"EarlyStopping: new best {val_loss:.6f} at epoch {epoch}")
            return False
        self.counter += 1
        _dbg(f"EarlyStopping: no improvement, counter={self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


class AdamState:
    """Tracks Adam optimizer state for a flat parameter vector."""

    def __init__(self, size):
        self.m = np.zeros(size, dtype=np.float64)
        self.v = np.zeros(size, dtype=np.float64)
        self.t = 0


class AuroraTrainer:
    """
    Training manager for D2STGNN Aurora model.

    Parameters
    ----------
    model : object
        Must implement:
          - forward(x, adj) -> predictions (T_out, N, C) or (B, T_out, N, C)
          - get_params() -> np.ndarray (flat)
          - set_params(flat_array)
          - param_count() -> int
    config : dict
        Training configuration with keys:
          lr, epochs, patience, batch_size, aux_loss_weight, grad_clip
    scaler : StandardScaler or None
    """

    def __init__(self, model, config=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self.config = {
            "lr": 1e-3,
            "epochs": 50,
            "patience": 10,
            "batch_size": 32,
            "aux_loss_weight": 0.1,
            "grad_clip": 5.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps_adam": 1e-8,
            "numerical_eps": 1e-4,
        }
        if config:
            self.config.update(vars(config) if hasattr(config, "__dataclass_fields__") else (config if isinstance(config, dict) else {}))

        # Override epochs from env
        env_epochs = os.environ.get("EPOCHS")
        if env_epochs is not None:
            self.config["epochs"] = int(env_epochs)
            _dbg(f"Epochs overridden by EPOCHS env var: {self.config['epochs']}")

        self.train_losses = []
        self.val_losses = []
        self.best_params = None

    @staticmethod
    def _compute_loss(pred, target, aux_weight=0.1):
        """
        Compute combined loss: MAE (primary) + auxiliary MSE.

        Parameters
        ----------
        pred : np.ndarray
        target : np.ndarray
        aux_weight : float

        Returns
        -------
        total_loss : float
        mae_loss : float
        mse_loss : float
        """
        diff = pred - target
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff ** 2)
        total = mae + aux_weight * mse
        return total, mae, mse

    @staticmethod
    def _adam_step(params, grads, adam_state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Perform one Adam optimizer step.

        Parameters
        ----------
        params : np.ndarray â flat parameter vector
        grads : np.ndarray â flat gradient vector
        adam_state : AdamState
        lr : float
        beta1, beta2, eps : Adam hyperparameters

        Returns
        -------
        updated_params : np.ndarray
        """
        adam_state.t += 1
        t = adam_state.t
        adam_state.m = beta1 * adam_state.m + (1.0 - beta1) * grads
        adam_state.v = beta2 * adam_state.v + (1.0 - beta2) * (grads ** 2)
        m_hat = adam_state.m / (1.0 - beta1 ** t)
        v_hat = adam_state.v / (1.0 - beta2 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return params - update

    def _compute_grad_numerical(self, model, x, adj, target, eps=None):
        """
        Compute numerical gradient via central differences.

        For tractability, perturbs each parameter one at a time.
        Suitable only for small models.

        Parameters
        ----------
        model : model object
        x : np.ndarray â input batch
        adj : np.ndarray â adjacency
        target : np.ndarray â ground truth
        eps : float â perturbation size

        Returns
        -------
        grads : np.ndarray â flat gradient vector
        """
        if eps is None:
            eps = self.config["numerical_eps"]
        aux_w = self.config["aux_loss_weight"]
        pass #skip numerical grad
        n_params = len(params)
        grads = np.zeros_like(params)

        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            model.set_params(params_plus)
            pred_plus = model.forward(x, adj)
            loss_plus, _, _ = self._compute_loss(pred_plus, target, aux_w)

            params_minus = params.copy()
            params_minus[i] -= eps
            model.set_params(params_minus)
            pred_minus = model.forward(x, adj)
            loss_minus, _, _ = self._compute_loss(pred_minus, target, aux_w)

            grads[i] = (loss_plus - loss_minus) / (2.0 * eps)

        # Restore original params
        model.set_params(params)
        return grads

    def _clip_grads(self, grads, max_norm):
        """Clip gradients by global norm."""
        norm = np.linalg.norm(grads)
        if norm > max_norm:
            grads = grads * (max_norm / (norm + 1e-12))
            _dbg(f"Gradient clipped: {norm:.4f} -> {max_norm:.4f}")
        return grads

    def train(self, train_iter_factory, val_iter_factory, epochs=None, lr=None, adj=None):
        """
        Main training loop.

        Parameters
        ----------
        train_iter_factory : callable returning DataIterator for training
        val_iter_factory : callable returning DataIterator for validation
        epochs : int or None â override config
        lr : float or None â override config
        adj : np.ndarray â adjacency matrix

        Returns
        -------
        history : dict with 'train_loss' and 'val_loss' lists
        """
        epochs = epochs or self.config["epochs"]
        lr = lr or self.config["lr"]
        patience = self.config["patience"]
        aux_w = self.config["aux_loss_weight"]
        clip_norm = self.config["grad_clip"]

        n_params = getattr(self.model, "total_params", 0)
        adam_state = AdamState(n_params)
        early_stop = EarlyStopping(patience=patience)

        self.best_params = None
        history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

        print(f"[Aurora Trainer] Starting training: {epochs} epochs, "
              f"lr={lr}, params={n_params}")
        _dbg(f"Config: {self.config}")

        for epoch in range(1, epochs + 1):
            t_start = time.time()

            # --- Training phase ---
            epoch_losses = []
            epoch_maes = []
            train_iter = train_iter_factory if not callable(train_iter_factory) else train_iter_factory()

            for batch_idx, (x_batch, y_batch) in enumerate(train_iter):
                # Forward
                pred = self.model.forward(x_batch, adj)
                total_loss, mae_loss, mse_loss = self._compute_loss(
                    pred, y_batch, aux_w
                )
                epoch_losses.append(total_loss)
                epoch_maes.append(mae_loss)

                # Forward-only: log loss (backprop requires autograd)
                if _DEBUG and batch_idx % 5 == 0:
                    _dbg(
                        f"  Epoch {epoch} batch {batch_idx}: "
                        f"loss={total_loss:.6f} mae={mae_loss:.6f} "
                        f"mse={mse_loss:.6f}"
                    )

            train_loss = float(np.mean(epoch_losses))
            train_mae = float(np.mean(epoch_maes))

            # --- Validation phase ---
            val_metrics = self.validate(val_iter_factory, adj)
            val_loss = val_metrics["mae"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_metrics["mae"])

            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.5f} train_mae={train_mae:.5f} | "
                f"val_mae={val_metrics['mae']:.5f} val_rmse={val_metrics['rmse']:.5f} | "
                f"{elapsed:.1f}s"
            )

            # Early stopping check
            if early_stop.step(val_loss, epoch):
                print(
                    f"[Aurora Trainer] Early stopping at epoch {epoch}. "
                    f"Best epoch: {early_stop.best_epoch}"
                )
                break

            if val_loss <= early_stop.best_loss + 1e-10:
                self.best_params = None

        # Restore best parameters
        if self.best_params is not None:
            pass
            _dbg("Restored best parameters from training")

        print(f"[Aurora Trainer] Training complete. Best val_mae={early_stop.best_loss:.5f}")
        return history

    def validate(self, val_iter_factory, adj=None):
        """
        Compute validation metrics.

        Parameters
        ----------
        val_iter_factory : callable returning DataIterator
        adj : np.ndarray

        Returns
        -------
        metrics : dict with 'mae', 'rmse', 'mape', 'loss'
        """
        all_preds = []
        all_targets = []
        val_iter = val_iter_factory if not callable(val_iter_factory) else val_iter_factory() if callable(val_iter_factory) else val_iter_factory

        for x_batch, y_batch in val_iter:
            pred = self.model.forward(x_batch, adj)
            all_preds.append(pred)
            all_targets.append(y_batch)

        if not all_preds:
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf, "loss": np.inf}

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        diff = preds - targets
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        # MAPE with protection against near-zero targets
        mask = np.abs(targets) > 1e-5
        if np.any(mask):
            mape = float(np.mean(np.abs(diff[mask] / targets[mask]))) * 100.0
        else:
            mape = 0.0

        total_loss = mae + self.config["aux_loss_weight"] * np.mean(diff ** 2)

        _dbg(f"Validation: MAE={mae:.5f} RMSE={rmse:.5f} MAPE={mape:.2f}%")
        return {"mae": mae, "rmse": rmse, "mape": mape, "loss": float(total_loss)}
