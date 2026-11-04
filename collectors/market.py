"""키움 REST 조회 → 데이터프레임 변환 (종목목록·일봉·분봉·틱·수급·프로그램매매·테마·순위)."""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from collectors.kiwoom_rest import KiwoomREST, normalize_row
from data.store import now_iso

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ 종목 목록
def stock_list(api: KiwoomREST) -> pd.DataFrame:
    """코스피(0)·코스닥(10) 전 종목. 연구 대상 여부(is_common)를 표시한다."""
    frames = []
    for mrkt in ("0", "10"):
        rows = []
        for page in api.paged("/api/dostk/stkinfo", "ka10099", {"mrkt_tp": mrkt}, list_key="list"):
            rows += page
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df["listCount"] = pd.to_numeric(df["listCount"], errors="coerce")
    df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce")
    df["is_common"] = (
        df["marketName"].isin(["거래소", "코스닥"])
        & (df["companyClassName"] != "스팩")
        & df["code"].str.fullmatch(r"\d{5}0")          # 우선주(끝자리 5·7·9 등) 제외
        & ~df["name"].str.contains("스팩")
    )
    df["is_halted"] = df["state"].str.contains("거래정지", na=False)
    df["received_at"] = now_iso()
    return df


# ---------------------------------------------------------------------- 일봉
def daily_bars(api: KiwoomREST, symbol: str, base_dt: str, start_dt: str | None = None) -> pd.DataFrame:
    """ka10081 일봉. base_dt 부터 과거로, start_dt(YYYYMMDD) 이전이 나오면 멈춘다."""
    rows: list[dict] = []
    body = {"stk_cd": symbol, "base_dt": base_dt, "upd_stkpc_tp": "1"}
    stop = (lambda r: r[-1]["dt"] < start_dt) if start_dt else None
    for page in api.paged("/api/dostk/chart", "ka10081", body, "stk_dt_pole_chart_qry", stop=stop):
        rows += page
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if df.empty:
        return df
    if start_dt:
        df = df[df["dt"] >= start_dt]
    df = df.rename(columns={"cur_prc": "close", "open_pric": "open", "high_pric": "high",
                            "low_pric": "low", "trde_qty": "volume", "trde_prica": "value_mil"})
    df.insert(0, "symbol", symbol)
    df["received_at"] = now_iso()
    return df


# ---------------------------------------------------------------------- 분봉
def minute_bars(api: KiwoomREST, symbol: str, since: str | None = None, scope: str = "1") -> pd.DataFrame:
    """ka10080 분봉 (최근 → 과거). since(YYYYMMDDHHMMSS) 이하가 나오면 멈춘다.

    키움은 최근 약 1년치만 제공하므로 매일 누적 저장해야 한다.
    cntr_tm 은 봉의 시작 시각이다 (예: 152000 = 15:20:00~15:20:59).
    """
    rows: list[dict] = []
    body = {"stk_cd": symbol, "tic_scope": scope, "upd_stkpc_tp": "1"}
    stop = (lambda r: r[-1]["cntr_tm"] <= since) if since else None
    for page in api.paged("/api/dostk/chart", "ka10080", body, "stk_min_pole_chart_qry", stop=stop):
        rows += page
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if df.empty:
        return df
    df = df.rename(columns={"cntr_tm": "ts", "cur_prc": "close", "open_pric": "open",
                            "high_pric": "high", "low_pric": "low", "trde_qty": "volume",
                            "acc_trde_qty": "acc_volume"})
    df.insert(0, "symbol", symbol)
    df["received_at"] = now_iso()
    return df


# ------------------------------------------------------------------------ 틱
def ticks(api: KiwoomREST, symbol: str, until_ts: str, max_pages: int = 400) -> pd.DataFrame:
    """ka10079 1틱 차트 (최근 → 과거). until_ts(YYYYMMDDHHMMSS) 이전까지 수집.

    키움은 최근 약 하루치 틱만 제공하므로 장중·장마감 직후에 구간을 정해 받는다.
    """
    rows: list[dict] = []
    body = {"stk_cd": symbol, "tic_scope": "1", "upd_stkpc_tp": "1"}
    for page in api.paged("/api/dostk/chart", "ka10079", body, "stk_tic_chart_qry",
                          stop=lambda r: r[-1]["cntr_tm"] < until_ts, max_pages=max_pages):
        rows += page
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if df.empty:
        return df
    df = df[df["cntr_tm"] >= until_ts]
    df = df.rename(columns={"cntr_tm": "ts", "cur_prc": "price", "trde_qty": "qty"})
    # 같은 초에 여러 틱이 있으므로 응답 순서(최근→과거)를 seq 로 보존한다.
    df = df.iloc[::-1].reset_index(drop=True)
    df["seq"] = range(len(df))
    df.insert(0, "symbol", symbol)
    df["received_at"] = now_iso()
    return df[["symbol", "ts", "seq", "price", "qty", "received_at"]]


# -------------------------------------------------------------------- 수급
def investor_daily(api: KiwoomREST, symbol: str, dt: str, pages: int = 1) -> pd.DataFrame:
    """ka10059 종목별 투자자·기관별 일별 순매수 (금액, 백만원 단위)."""
    rows: list[dict] = []
    body = {"dt": dt, "stk_cd": symbol, "amt_qty_tp": "1", "trde_tp": "0", "unit_tp": "1000"}
    for page in api.paged("/api/dostk/stkinfo", "ka10059", body, "stk_invsr_orgn", max_pages=pages):
        rows += page
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if not df.empty:
        df.insert(0, "symbol", symbol)
        df["received_at"] = now_iso()
    return df


def investor_intraday(api: KiwoomREST, symbol: str) -> pd.DataFrame:
    """ka10064 장중 투자자별 매매 (잠정치, 당일만 제공). 수집 시각이 곧 관측 시각이다."""
    j, _ = api.request("/api/dostk/chart", "ka10064",
                       {"mrkt_tp": "000", "amt_qty_tp": "1", "trde_tp": "0", "stk_cd": symbol})
    rows = next((v for v in j.values() if isinstance(v, list)), [])
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if not df.empty:
        df.insert(0, "symbol", symbol)
        df["received_at"] = now_iso()
    return df


def program_intraday(api: KiwoomREST, symbol: str, date: str) -> pd.DataFrame:
    """ka90008 종목 시간별 프로그램매매 추이 (지정일)."""
    rows: list[dict] = []
    body = {"amt_qty_tp": "1", "stk_cd": symbol, "date": date}
    for page in api.paged("/api/dostk/mrkcond", "ka90008", body):
        rows += page
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if not df.empty:
        df.insert(0, "symbol", symbol)
        df.insert(1, "date", date)
        df["received_at"] = now_iso()
    return df


# --------------------------------------------------------------------- 순위·테마
def trade_value_rank(api: KiwoomREST, pages: int = 1) -> pd.DataFrame:
    """ka10032 거래대금 상위 (통합). 종목코드의 _AL 접미사를 떼어 symbol 로 둔다."""
    rows: list[dict] = []
    body = {"mrkt_tp": "000", "mang_stk_incls": "0", "stex_tp": "3"}
    for page in api.paged("/api/dostk/rkinfo", "ka10032", body, "trde_prica_upper", max_pages=pages):
        rows += page
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if not df.empty:
        df["symbol"] = df["stk_cd"].str.split("_").str[0]
        df["received_at"] = now_iso()
    return df


def theme_groups(api: KiwoomREST) -> pd.DataFrame:
    rows: list[dict] = []
    body = {"qry_tp": "0", "stk_cd": "", "date_tp": "1", "thema_nm": "", "flu_pl_amt_tp": "1", "stex_tp": "1"}
    for page in api.paged("/api/dostk/thme", "ka90001", body, "thema_grp"):
        rows += page
    df = pd.DataFrame(rows)
    if not df.empty:
        df["received_at"] = now_iso()
    return df


def theme_members(api: KiwoomREST, theme_code: str) -> pd.DataFrame:
    j, _ = api.request("/api/dostk/thme", "ka90002",
                       {"date_tp": "1", "thema_grp_cd": theme_code, "stex_tp": "1"})
    rows = next((v for v in j.values() if isinstance(v, list)), [])
    df = pd.DataFrame([{"thema_grp_cd": theme_code, "stk_cd": r.get("stk_cd"), "stk_nm": r.get("stk_nm")}
                       for r in rows])
    return df


# ------------------------------------------------------------------ 업종·지수
# 업종 분류로 쓰지 않는 지수 (종합·규모별·스타일·파생). '제조'는 하위 업종을 묶은 상위 분류다.
NON_SECTOR = {"001", "002", "003", "004", "027", "101", "106"}


def sector_codes(api: KiwoomREST) -> pd.DataFrame:
    """ka10101 업종코드 목록. market: 0=코스피, 1=코스닥. is_sector=업종 분류용 여부."""
    frames = []
    for mrkt in ("0", "1"):
        j, _ = api.request("/api/dostk/stkinfo", "ka10101", {"mrkt_tp": mrkt})
        df = pd.DataFrame(j.get("list", []))
        df["market"] = mrkt
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).rename(columns={"code": "inds_cd", "name": "inds_nm"})
    c = df["inds_cd"].astype(int)
    df["is_sector"] = ~df["inds_cd"].isin(NON_SECTOR) & ((c <= 30) | c.between(103, 137) | (c == 141))
    return df.loc[:, ["market", "inds_cd", "inds_nm", "is_sector"]]


def sector_members(api: KiwoomREST, market_tp: str, inds_cd: str) -> list[str]:
    """ka20002 업종별 주가 → 그 업종에 속한 종목코드 목록 (조회 시점 기준)."""
    codes: list[str] = []
    body = {"mrkt_tp": market_tp, "inds_cd": inds_cd, "stex_tp": "1"}
    for page in api.paged("/api/dostk/sect", "ka20002", body, "inds_stkpc"):
        codes += [r["stk_cd"] for r in page]
    return codes


def index_daily(api: KiwoomREST, inds_cd: str, base_dt: str, start_dt: str) -> pd.DataFrame:
    """ka20006 업종(지수) 일봉. 키움은 지수 값을 100배 정수로 주므로 100으로 나눈다."""
    rows: list[dict] = []
    body = {"inds_cd": inds_cd, "base_dt": base_dt}
    for page in api.paged("/api/dostk/chart", "ka20006", body, "inds_dt_pole_qry",
                          stop=lambda r: r[-1]["dt"] < start_dt):
        rows += page
    df = pd.DataFrame([normalize_row(r) for r in rows])
    if df.empty:
        return df
    df = df[df["dt"] >= start_dt].rename(columns={
        "cur_prc": "close", "open_pric": "open", "high_pric": "high", "low_pric": "low",
        "trde_qty": "volume", "trde_prica": "value_mil"})
    for c in ("close", "open", "high", "low"):
        df[c] = df[c] / 100
    df.insert(0, "inds_cd", inds_cd)
    df["received_at"] = now_iso()
    return df.loc[:, ["inds_cd", "dt", "open", "high", "low", "close", "volume", "value_mil", "received_at"]]


def today() -> str:
    return datetime.now().strftime("%Y%m%d")
