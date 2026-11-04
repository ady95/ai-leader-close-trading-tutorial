"""17장 실험 — PPO·SAC 동적 청산 vs S1·S5·S7 (1분봉 기간, 같은 진입 포지션).

사용법 (먼저 build1518)
  python -m experiments.ch17_rl paths      # 진입 포지션별 다음 날 09:00~09:29 1분봉 경로 → research/rl_paths.npz
  python -m experiments.ch17_rl train      # 학습(2026-06 이전) → 평가(2026-07 이후)

학습·평가 분할은 한 번만 한다(계산량 때문). 하이퍼파라미터는 stable-baselines3 기본값에 가깝게 고정했다.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from backtest.costs import round_trip_cost
from backtest.metrics import fmt_table, summarize
from data.store import path
from experiments.ch12_minute import frame
from models.reinforcement_learning.exit_env import N_STEPS, ExitEnv, evaluate, half_tick

pd.set_option("display.width", 200)
NPZ = path("research", "rl_paths.npz")
STATIC = ["ret_1518", "clv_1518", "upper_wick_1518", "vs_vwap", "vs_ma20", "mom_30m", "peer_ret_1518",
          "n_peers_top", "near_high_prev", "tv_ratio20_1518"]
SPLIT = "20260701"


def _job(args):
    symbol, rows = args
    m = pd.read_parquet(path("minute", f"{symbol}.parquet"), columns=["ts", "open", "high", "low", "close", "volume"])
    m["dt"], m["hm"] = m["ts"].str[:8], m["ts"].str[8:12]
    m = m[(m["hm"] >= "0900") & (m["hm"] <= "0929")]
    days = {d: g.sort_values("hm") for d, g in m.groupby("dt")}
    out = []
    for r in rows:
        g = days.get(r["nxt_dt"])
        if g is None or len(g) < 5 or g["hm"].iloc[0] != "0900":
            continue
        g = g.set_index("hm").reindex([f"09{k:02d}" for k in range(30)]).ffill()   # 거래 없는 분은 직전 값
        c = g["close"].to_numpy(float)
        tp = ((g["high"] + g["low"] + g["close"]) / 3).to_numpy(float)
        v = g["volume"].fillna(0).to_numpy(float)
        cv = np.cumsum(v)
        vwap = np.where(cv > 0, np.cumsum(tp * v) / np.where(cv > 0, cv, 1), c)
        out.append({"symbol": symbol, "dt": r["dt"], "close": c, "vwap": vwap, "open": float(g["open"].iloc[0])})
    return out


def cmd_paths() -> None:
    x = frame()
    x = x.dropna(subset=STATIC[:6] + ["nxt_open"])
    jobs = [(s, g.loc[:, ["dt", "nxt_dt"]].to_dict("records")) for s, g in x.groupby("symbol")]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = [r for part in ex.map(_job, jobs, chunksize=4) for r in part]
    p = pd.DataFrame(res).merge(x, on=["symbol", "dt"])
    p = p[np.isfinite(np.stack(p["close"].to_numpy())).all(axis=1)]
    st = p.loc[:, STATIC].astype(float).clip(-5, 5).fillna(0.0).to_numpy()
    np.savez(NPZ, entry=p["entry"].to_numpy(float), static=st, open=p["open"].to_numpy(float),
             close=np.stack(p["close"].to_numpy()), vwap=np.stack(p["vwap"].to_numpy()),
             cost=round_trip_cost(p["nxt_dt"]).to_numpy(float), dt=p["dt"].to_numpy(str),
             nxt_dt=p["nxt_dt"].to_numpy(str), symbol=p["symbol"].to_numpy(str))
    print(f"{len(p):,} 에피소드 → {NPZ}")


def subset(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def fixed_policy(ep: dict, step: int) -> np.ndarray:
    """정해진 단계에서 전량 매도 (0 = 시가, k = 09:0(k-1) 봉 종가)."""
    if step == 0:
        px = ep["open"]
    else:
        c = ep["close"][:, step - 1]
        px = c - half_tick(c)
    return px / ep["entry"] - 1 - ep["cost"]


def cmd_train() -> None:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env
    d = dict(np.load(NPZ, allow_pickle=False))
    tr, te = subset(d, d["dt"] < SPLIT), subset(d, d["dt"] >= SPLIT)
    print(f"학습 {len(tr['entry']):,}건 ({min(tr['dt'])}~{max(tr['dt'])}), 평가 {len(te['entry']):,}건 "
          f"({min(te['dt'])}~{max(te['dt'])})")
    results = {"S1 시가 전량": fixed_policy(te, 0), "09:01 전량": fixed_policy(te, 1),
               "09:10 전량 (S5)": fixed_policy(te, 10), "09:30 전량 (S5)": fixed_policy(te, N_STEPS)}
    for lam in (0.0, 1.0):
        venv = make_vec_env(lambda: ExitEnv(tr, continuous=False, lam=lam, seed=1), n_envs=8, seed=1)
        ppo = PPO("MlpPolicy", venv, seed=1, verbose=0, n_steps=256, batch_size=512, learning_rate=3e-4)
        ppo.learn(total_timesteps=200_000)
        results[f"PPO (위험 제약 λ={lam:g})"] = evaluate(ppo, te, continuous=False)
    sac = SAC("MlpPolicy", ExitEnv(tr, continuous=True, seed=2), seed=2, verbose=0, buffer_size=200_000,
              learning_starts=5_000, batch_size=256)
    sac.learn(total_timesteps=60_000)
    results["SAC (연속 매도 비율)"] = evaluate(sac, te, continuous=True)
    rows = {k: summarize(pd.DataFrame({"r": v, "dt": te["dt"]}), "r") for k, v in results.items()}
    print(fmt_table(rows).iloc[:, :8].to_string())
    base = results["S1 시가 전량"]
    for k, v in results.items():
        if k.startswith("S1"):
            continue
        dlt = v - base
        print(f"  {k} − S1: {dlt.mean() * 100:+.3f}%p (t={dlt.mean() / dlt.std() * np.sqrt(len(dlt)):+.2f})")
    # 정책이 실제로 무엇을 했나: 시가에 판 비율
    env = ExitEnv(te, continuous=False)
    env.set_sequential()
    first = []
    for _ in range(env.n):
        obs, _ = env.reset()
        a, _ = ppo.predict(obs, deterministic=True)
        first.append(int(a))
    print("PPO(λ=1) 장 시작 전 행동 분포 (0 보유, 1 절반, 2 전량):", pd.Series(first).value_counts(normalize=True)
          .round(3).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["paths", "train"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
