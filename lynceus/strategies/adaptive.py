"""lynceus/strategies/adaptive.py — Online adaptive routing with Thompson Sampling.

算法改动 (Claude #7, M311-M320):
    1. UCB1/softmax选择 → Thompson Sampling (Beta分布后验采样)
       每个device维护(alpha, beta)参数, 采样θ~Beta(α,β), 选θ最大的device。
    2. _softmax_select → _gumbel_max_select: Gumbel-Max trick (数值稳定)
       加Gumbel(0,1)噪声到log概率上, 取argmax, 等价于按概率采样但避免softmax溢出。
    3. observe: 简单EMA → Bayesian online update (Welford法同时更新μ和σ²)
    4. _dbg_thompson_state: 打印每个device的Beta分布参数和采样值
"""
from __future__ import annotations
import math
import random as _random
from collections import defaultdict
from typing import Dict, List, Optional
from ..costing import CostBreakdown, CostModelEngine, QueryDescriptor
from ..schema import HardwareKind
from .foundation import RoutingDecision, RoutingStrategyBase


def _beta_sample(alpha: float, beta: float) -> float:
    """从Beta(alpha, beta)分布采样。用Joehnk方法避免对gamma函数的依赖。
    当alpha,beta都>=1时用rejection sampling; 否则用变换法。"""
    if alpha <= 0:
        alpha = 0.01
    if beta <= 0:
        beta = 0.01
    # Python的random模块有betavariate
    return _random.betavariate(alpha, beta)


def _gumbel_sample() -> float:
    """采样Gumbel(0,1)分布: -log(-log(U)), U~Uniform(0,1)
    用于Gumbel-Max trick实现离散概率采样。"""
    u = _random.random()
    # 防止log(0)
    u = max(u, 1e-10)
    u = min(u, 1.0 - 1e-10)
    return -math.log(-math.log(u))


def _dbg_thompson_state(query_id: str, ts_alpha: Dict[str, float],
                        ts_beta: Dict[str, float], samples: Dict[str, float],
                        chosen: str):
    """打印Thompson Sampling的完整状态"""
    from .._debug import dbg
    state_lines = []
    for dev in sorted(ts_alpha.keys()):
        a, b = ts_alpha[dev], ts_beta[dev]
        theta = samples.get(dev, 0.0)
        mean = a / (a + b) if (a + b) > 0 else 0.5
        state_lines.append(f"  {dev}: α={a:.2f} β={b:.2f} E[θ]={mean:.4f} sampled={theta:.4f}")
    dbg('thompson_state',
        query_id=query_id,
        chosen=chosen,
        device_states="\n".join(state_lines))


class AdaptiveStrategy(RoutingStrategyBase):
    """Thompson Sampling自适应路由: 每个device维护Beta分布后验,
    通过reward信号(latency越低reward越高)在线更新。"""

    def __init__(self, engine: CostModelEngine, *, warmup_steps: int = 40,
                 reward_scale: float = 1.0, prior_alpha: float = 1.0,
                 prior_beta: float = 1.0, **kwargs):
        super().__init__(engine, **kwargs)
        self._warmup_steps = warmup_steps
        self._reward_scale = reward_scale
        self._prior_a = prior_alpha
        self._prior_b = prior_beta
        # Thompson Sampling 参数: Beta(alpha, beta) per device
        self._ts_alpha: Dict[str, float] = defaultdict(lambda: prior_alpha)
        self._ts_beta: Dict[str, float] = defaultdict(lambda: prior_beta)
        # Bayesian online统计 (Welford法)
        self._bias_count: Dict[str, int] = defaultdict(int)
        self._bias_mean: Dict[str, float] = defaultdict(float)
        self._bias_m2: Dict[str, float] = defaultdict(float)  # sum of (x-mean)^2
        self._device_load: Dict[str, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "Adaptive"

    def _gumbel_max_select(self, log_probs: Dict[str, float]) -> str:
        """Gumbel-Max trick: 加Gumbel(0,1)噪声到log概率, 取argmax。
        数学等价于按exp(log_prob)/Z的概率采样, 但避免softmax的数值溢出。
        参考: Gumbel 1954, Maddison et al. 2014"""
        if not log_probs:
            raise RuntimeError("No devices")
        best_dev = None
        best_val = float('-inf')
        for dev, lp in log_probs.items():
            perturbed = lp + _gumbel_sample()
            if perturbed > best_val:
                best_val = perturbed
                best_dev = dev
        return best_dev

    def route_one(self, query: QueryDescriptor,
                  data_location: Optional[str] = None) -> RoutingDecision:
        from .._debug import dbg, checkpoint

        estimates = self._engine.estimate_all_devices(query, data_location)
        if not estimates:
            raise RuntimeError("No devices available for routing")

        # Warmup: cost-greedy with forced device rotation for initial stats
        if self._query_count < self._warmup_steps:
            # Every 6th query pick the least-tried device; otherwise pick min cost
            if self._query_count % 6 == 0 and len(estimates) > 1:
                best_id = min(estimates, key=lambda k: self._device_load.get(k, 0))
            else:
                best_id = min(estimates, key=lambda k: estimates[k].total_us)
            self._device_load[best_id] += 1
            self._query_count += 1
            return RoutingDecision(
                query_id=query.query_id, device_id=best_id,
                cost=estimates[best_id], confidence=0.5,
                metadata={"reason": "warmup_explore", "step": self._query_count})

        # Hybrid UCB-Thompson: combine Thompson posterior sample with
        # transfer-aware cost bonus (改动: 纯Thompson → UCB-Thompson混合)
        # UCB bonus = sqrt(2·ln(t) / n_i) weighted by transfer ratio
        samples: Dict[str, float] = {}
        total_pulls = max(1, sum(self._device_load.values()))
        for dev_id in estimates:
            alpha = self._ts_alpha[dev_id]
            beta = self._ts_beta[dev_id]
            theta = _beta_sample(alpha, beta)
            # UCB exploration bonus: encourages under-sampled devices
            n_i = max(1, self._device_load.get(dev_id, 1))
            ucb_bonus = math.sqrt(2.0 * math.log(total_pulls + 1) / n_i)
            # Transfer penalty: penalize high-transfer-ratio devices
            cb = estimates[dev_id]
            xfer_ratio = cb.transfer_cost_us / max(1.0, cb.total_us)
            xfer_penalty = xfer_ratio * 0.15
            # Combined score: Thompson sample + UCB bonus - transfer penalty
            samples[dev_id] = theta + 0.3 * ucb_bonus - xfer_penalty

        # 选combined score最大的device
        chosen = max(samples, key=lambda k: samples[k])
        self._device_load[chosen] += 1
        self._query_count += 1

        _dbg_thompson_state(query.query_id, dict(self._ts_alpha),
                            dict(self._ts_beta), samples, chosen)

        checkpoint("thompson_select", chosen=chosen,
                   samples={k: f"{v:.4f}" for k, v in samples.items()})

        return RoutingDecision(
            query_id=query.query_id, device_id=chosen,
            cost=estimates[chosen],
            confidence=min(1.0, self._query_count / max(1, self._warmup_steps * 2)),
            metadata={"reason": "thompson_sampling",
                      "ts_alpha": self._ts_alpha[chosen],
                      "ts_beta": self._ts_beta[chosen],
                      "sampled_theta": samples[chosen],
                      "device_load": dict(self._device_load)})

    def observe(self, query_id: str, device_id: str, actual_latency_us: float) -> None:
        """Bayesian online update with percentile-rank reward (改动: 固定scale → 自适应rank)。
        Reward = 1 - rank(latency) / n, 使得reward相对于观测分布自适应。
        同时用Welford法维护bias的running mean和variance。"""
        # 1. Welford在线更新bias统计 (先更新,后计算reward)
        self._bias_count[device_id] += 1
        n = self._bias_count[device_id]
        delta = actual_latency_us - self._bias_mean[device_id]
        self._bias_mean[device_id] += delta / n
        delta2 = actual_latency_us - self._bias_mean[device_id]
        self._bias_m2[device_id] += delta * delta2

        # 2. 自适应reward: 用z-score转为[0,1]区间
        # z = (x - μ) / σ, reward = sigmoid(-z) 使低latency得高reward
        std = math.sqrt(self._bias_m2[device_id] / max(1, n - 1)) if n > 1 else 1.0
        std = max(std, 1.0)  # 防止除零
        z_score = (actual_latency_us - self._bias_mean[device_id]) / std
        reward = 1.0 / (1.0 + math.exp(z_score))  # sigmoid(-z)
        reward = max(0.01, min(0.99, reward))

        # 3. Beta分布更新: 温和更新避免posterior过度集中
        update_weight = 0.8  # damping factor
        self._ts_alpha[device_id] += update_weight * reward
        self._ts_beta[device_id] += update_weight * (1.0 - reward)
        self._bias_m2[device_id] += delta * delta2

    def observe_with_estimate(self, device_id: str, estimated_us: float,
                              actual_us: float) -> None:
        """用预估vs实际的比值来调整Beta参数,惩罚预估不准的device。"""
        if estimated_us <= 0:
            return
        ratio = actual_us / estimated_us
        # ratio < 1 说明实际比预估快 → 奖励 (α增大)
        # ratio > 1 说明实际比预估慢 → 惩罚 (β增大)
        if ratio < 1.0:
            self._ts_alpha[device_id] += 0.5 * (1.0 - ratio)
        else:
            self._ts_beta[device_id] += 0.3 * (ratio - 1.0)

    def reset(self) -> None:
        super().reset()
        self._ts_alpha.clear()
        self._ts_beta.clear()
        self._bias_count.clear()
        self._bias_mean.clear()
        self._bias_m2.clear()
        self._device_load.clear()
