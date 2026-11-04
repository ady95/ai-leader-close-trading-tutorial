"""다음 날 아침 청산 강화학습 환경 (17장). 명세는 실험 전에 고정했다.

에피소드 = 진입 포지션 하나의 다음 날 아침.
  단계 0      장 시작 전. 전날까지의 정보(15:18 특징)만 보고 시가 단일가에 팔 비율을 정한다 (스프레드 없음).
  단계 1~30   09:00~09:29 각 1분봉이 끝난 시점. 남은 수량 중 팔 비율을 정하고 그 봉 종가 - 반 호가에 판다.
  09:30       남은 수량은 09:29 봉 종가 - 반 호가에 강제 청산.
보상  = 판 비율 × (매도가 ÷ 진입가 - 1 - 왕복 비용률).  위험 제약(17-2): lam × max(0, -손익 - 2%)를 마지막에 뺀다.
관측  = [정적 특징 …, 현재가/진입가-1, 시가/진입가-1, 경과 단계/30, 남은 비율, 현재가/당일 VWAP-1]  (모두 그 시점까지의 값)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

N_STEPS = 30


def half_tick(p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    bins = np.array([2_000, 5_000, 20_000, 50_000, 200_000, 500_000])
    ticks = np.array([1, 5, 10, 50, 100, 500, 1_000], dtype=float)
    return ticks[np.searchsorted(bins, p, side="right")] / 2


class ExitEnv(gym.Env):
    """episodes: dict(entry[N], static[N,F], open[N], close[N,30], vwap[N,30], cost[N])"""

    def __init__(self, episodes: dict, continuous: bool = False, lam: float = 0.0, seed: int = 0):
        super().__init__()
        self.ep, self.continuous, self.lam = episodes, continuous, lam
        self.n, f = episodes["static"].shape
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(f + 5,), dtype=np.float32)
        self.action_space = (gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32) if continuous
                             else gym.spaces.Discrete(3))         # 0 보유, 1 남은 수량 절반, 2 전량
        self.order = None
        self.cursor = 0

    def set_sequential(self) -> None:
        """평가용: 에피소드를 순서대로 한 번씩."""
        self.order, self.cursor = np.arange(self.n), 0

    def _obs(self) -> np.ndarray:
        i, t = self.i, self.t
        e = self.ep["entry"][i]
        px = self.ep["open"][i] if t == 0 else self.ep["close"][i, t - 1]
        vw = self.ep["open"][i] if t == 0 else self.ep["vwap"][i, t - 1]
        dyn = [px / e - 1, self.ep["open"][i] / e - 1 if t > 0 else 0.0, t / N_STEPS, self.left, px / vw - 1]
        if t == 0:
            dyn[0] = 0.0          # 장 시작 전에는 시가를 모른다
            dyn[4] = 0.0
        return np.concatenate([self.ep["static"][i], dyn]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.order is not None:
            self.i = int(self.order[self.cursor % self.n])
            self.cursor += 1
        else:
            self.i = int(self.rng.integers(self.n))
        self.t, self.left, self.pnl = 0, 1.0, 0.0
        return self._obs(), {}

    def _frac(self, a) -> float:
        if self.continuous:
            return float(np.clip(np.asarray(a).ravel()[0], 0.0, 1.0))
        return (0.0, 0.5, 1.0)[int(a)]

    def step(self, a):
        i, t = self.i, self.t
        e, cost = self.ep["entry"][i], self.ep["cost"][i]
        sell = self.left * self._frac(a)
        px = self.ep["open"][i] if t == 0 else self.ep["close"][i, t - 1] - half_tick(self.ep["close"][i, t - 1])
        r = sell * (px / e - 1 - cost)
        self.left -= sell
        self.t += 1
        done = self.left <= 1e-9 or self.t > N_STEPS
        if self.t > N_STEPS and self.left > 1e-9:            # 09:30 강제 청산
            last = self.ep["close"][i, N_STEPS - 1]
            r += self.left * ((last - half_tick(last)) / e - 1 - cost)
            self.left = 0.0
        self.pnl += r
        if done and self.lam > 0:
            r -= self.lam * max(0.0, -self.pnl - 0.02)
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(r), done, False, {"pnl": self.pnl}


def evaluate(model, episodes: dict, continuous: bool) -> np.ndarray:
    """에피소드마다 결정적 정책으로 한 번씩 돌려 손익(비용 후)을 돌려준다."""
    env = ExitEnv(episodes, continuous=continuous)
    env.set_sequential()
    out = np.zeros(env.n)
    for k in range(env.n):
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(a)
        out[k] = info["pnl"]
    return out
