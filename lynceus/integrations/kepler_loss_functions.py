"""
kepler_loss_functions â Loss function utilities for Lynceus.

Ported from upstream/kepler/code/loss_functions.py (50 lines).
Algorithm changes (~20%):
  - mse_loss: numpy reduce_mean of squared (y_true + y_pred), adds eps clamp
  - log_mse_loss: numpy log with floor clamp to avoid log(0)
  - huber_loss: new â smooth L1 / Huber loss with configurable delta
  - asymmetric_loss: new â penalizes underestimates more than overestimates
  - combined_loss: weighted blend of mse + huber for robust training
"""
import os
import numpy as np

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[kepler_loss] {tag}: {items}")


# ââ MSE loss (negative-cost convention) ââââââââââââââââââââââââââ
def mse_loss(y_true: np.ndarray,
             y_pred: np.ndarray,
             *,
             eps: float = 1e-8) -> float:
    """Compute mean squared error with negative-cost convention.

    The true target values represent costs; the model learns to predict
    the negative cost, so residual = y_true + y_pred.  A small eps is
    added inside the square to avoid exact-zero gradients at the minimum.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true + y_pred
    loss = float(np.mean(np.square(residual) + eps))
    _dbg("mse_loss", shape=y_true.shape, loss=loss, eps=eps)
    return loss


def mse_loss_grad(y_true: np.ndarray,
                  y_pred: np.ndarray) -> np.ndarray:
    """Gradient of mse_loss w.r.t. y_pred."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = y_true.size
    grad = 2.0 * (y_true + y_pred) / n
    _dbg("mse_loss_grad", norm=float(np.linalg.norm(grad)), n=n)
    return grad


# ââ Log-MSE loss âââââââââââââââââââââââââââââââââââââââââââââââââ
def log_mse_loss(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 *,
                 floor: float = 1e-12) -> float:
    """MSE against log of true values with floor clamp to avoid log(0)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    clamped = np.maximum(y_true, floor)
    residual = np.log(clamped) + y_pred
    loss = float(np.mean(np.square(residual)))
    _dbg("log_mse_loss", shape=y_true.shape, loss=loss,
         clamped_min=float(np.min(clamped)), floor=floor)
    return loss


def log_mse_loss_grad(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      *,
                      floor: float = 1e-12) -> np.ndarray:
    """Gradient of log_mse_loss w.r.t. y_pred."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    clamped = np.maximum(y_true, floor)
    n = y_true.size
    grad = 2.0 * (np.log(clamped) + y_pred) / n
    _dbg("log_mse_grad", norm=float(np.linalg.norm(grad)))
    return grad


# ââ Huber loss (new) âââââââââââââââââââââââââââââââââââââââââââââ
def huber_loss(y_true: np.ndarray,
               y_pred: np.ndarray,
               *,
               delta: float = 1.0) -> float:
    """Smooth L1 / Huber loss â quadratic for small residuals, linear for large."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true + y_pred
    abs_r = np.abs(residual)
    quadratic = np.minimum(abs_r, delta)
    linear = abs_r - quadratic
    loss = float(np.mean(0.5 * np.square(quadratic) + delta * linear))
    _dbg("huber_loss", loss=loss, delta=delta,
         frac_linear=float(np.mean(abs_r > delta)))
    return loss


def huber_loss_grad(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    *,
                    delta: float = 1.0) -> np.ndarray:
    """Gradient of huber_loss w.r.t. y_pred."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true + y_pred
    n = y_true.size
    grad = np.where(np.abs(residual) <= delta,
                    residual / n,
                    delta * np.sign(residual) / n)
    _dbg("huber_grad", norm=float(np.linalg.norm(grad)))
    return grad


# ââ Asymmetric loss (new) ââââââââââââââââââââââââââââââââââââââââ
def asymmetric_loss(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    *,
                    alpha_under: float = 2.0,
                    alpha_over: float = 1.0) -> float:
    """Asymmetric MSE â penalizes underestimates (positive residual) more heavily."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true + y_pred
    weights = np.where(residual > 0, alpha_under, alpha_over)
    loss = float(np.mean(weights * np.square(residual)))
    _dbg("asymmetric_loss", loss=loss,
         alpha_under=alpha_under, alpha_over=alpha_over,
         frac_under=float(np.mean(residual > 0)))
    return loss


def asymmetric_loss_grad(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         *,
                         alpha_under: float = 2.0,
                         alpha_over: float = 1.0) -> np.ndarray:
    """Gradient of asymmetric_loss w.r.t. y_pred."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true + y_pred
    n = y_true.size
    weights = np.where(residual > 0, alpha_under, alpha_over)
    grad = 2.0 * weights * residual / n
    _dbg("asymmetric_grad", norm=float(np.linalg.norm(grad)))
    return grad


# ââ Combined loss ââââââââââââââââââââââââââââââââââââââââââââââââ
def combined_loss(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  *,
                  mse_weight: float = 0.5,
                  huber_weight: float = 0.5,
                  huber_delta: float = 1.0) -> float:
    """Weighted blend of MSE and Huber for robust training."""
    m = mse_loss(y_true, y_pred)
    h = huber_loss(y_true, y_pred, delta=huber_delta)
    loss = mse_weight * m + huber_weight * h
    _dbg("combined_loss", mse=m, huber=h, total=loss)
    return loss
