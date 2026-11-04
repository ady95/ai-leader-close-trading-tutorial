"""LightGBM Walk-Forward 학습 (12장).

원칙
  - 하이퍼파라미터는 실험 전에 고정한다(PARAMS). 테스트 결과를 보고 바꾸지 않는다.
  - 학습은 테스트 구간보다 과거만 쓴다(확장 창). 라벨이 미래 N일을 보므로 학습 끝과 테스트 시작
    사이에 EMBARGO 거래일을 비운다.
  - 모든 예측은 '그 날짜 이전 데이터로만 학습한 모델'이 낸 값이다.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
              seed=42, verbose=-1, num_threads=4)
CLS_PARAMS = {**PARAMS, "objective": "binary"}
NUM_ROUNDS = 300
EMBARGO = 5


def folds(dates: pd.Series, test_starts: list[str], test_ends: list[str], train_start: str) -> list[dict]:
    days = np.sort(dates.unique())
    out = []
    for a, b in zip(test_starts, test_ends):
        test = days[(days >= a) & (days <= b)]
        if len(test) == 0:
            continue
        prior = days[days < test[0]]
        train_end = prior[-(EMBARGO + 1)] if len(prior) > EMBARGO else None
        if train_end is None:
            continue
        out.append({"train": (train_start, train_end), "test": (test[0], test[-1])})
    return out


def walk_forward(df: pd.DataFrame, features: list[str], target: str, fold_list: list[dict],
                 categorical: list[str] | None = None, classifier: bool = False,
                 clip: float | None = 0.15) -> tuple[pd.Series, pd.DataFrame]:
    """폴드마다 학습 → 테스트 예측. (예측 Series, 폴드별 특징 중요도) 를 돌려준다."""
    pred = pd.Series(np.nan, index=df.index)
    imps = []
    X = df.loc[:, features].copy()
    for c in categorical or []:
        X[c] = X[c].astype("category")
    for i, f in enumerate(fold_list):
        tr = df["dt"].between(*f["train"]) & df[target].notna()
        te = df["dt"].between(*f["test"])
        y = df.loc[tr, target]
        if clip is not None and not classifier:
            y = y.clip(-clip, clip)                      # 극단값 몇 개가 학습을 끌고 가지 않게
        ds = lgb.Dataset(X.loc[tr], y, categorical_feature=categorical or "auto", free_raw_data=True)
        booster = lgb.train(CLS_PARAMS if classifier else PARAMS, ds, num_boost_round=NUM_ROUNDS)
        pred.loc[te] = booster.predict(X.loc[te])
        imps.append(pd.Series(booster.feature_importance("gain"), index=features, name=f"fold{i}"))
    return pred, pd.concat(imps, axis=1)
