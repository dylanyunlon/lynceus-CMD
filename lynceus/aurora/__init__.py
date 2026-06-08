"""
lynceus.aurora — D2STGNN Aurora variant for spatio-temporal forecasting.

Modules:
  config              AuroraConfig dataclass
  graph_conv          Diffusion convolution, Chebyshev basis, GCN
  dynamic_graph       Dynamic graph learner (node embeddings, KNN)
  temporal_attention  Multi-head attention, positional encoding, gated conv
  d2stgnn       Main D2STGNN model
  data_loader         Dataset, synthetic data, scaler, iterator
  trainer             Training loop with Adam optimizer
  evaluator           MAE/RMSE/MAPE metrics
"""
from lynceus.aurora.config import AuroraConfig
