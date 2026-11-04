# ai-leader-close-trading-tutorial

위키독스 「AI로 검증하는 주도주 종가배팅 자동매매 따라하기」의 예제 코드 저장소입니다.

- 책: https://wikidocs.net/book/21983
- 데이터·주문: **키움증권 REST API** 단일 구성 (공시는 DART 오픈API 보조)
- 목적: 실전 트레이더의 종가배팅 경험칙을 데이터로 검증하고, LightGBM·강화학습의 추가 가치를 측정하는 **연구 코드**입니다. 수익을 보장하지 않으며 투자 권유가 아닙니다.

## 빠른 시작

```bash
git clone https://github.com/ady95/ai-leader-close-trading-tutorial.git
cd ai-leader-close-trading-tutorial
uv venv -p 3.12 .venv            # 또는 python -m venv .venv
uv pip install -p .venv -r requirements.txt
cp .env.example .env              # 키움 앱키·시크릿, DART 키 입력
python -m pytest -q tests
```

키움 REST API는 [키움 OpenAPI 포털](https://openapi.kiwoom.com)에서 신청하고 **접속 IP를 등록**해야 호출됩니다.
시세 수집은 실전 키(조회 전용)를 쓰고, 주문 예제는 `TRADING_MODE=paper`(모의투자)를 기본으로 합니다.

## 데이터 수집

키움 REST API는 **1분봉을 최근 약 1년, 틱을 약 하루치**만 제공합니다. 과거 호가와 장중 잠정 수급 이력은 제공하지 않습니다.
그래서 연구를 시작하는 날부터 매일 쌓아 두어야 합니다.

```bash
# 1) 최초 1회: 종목목록 → 일봉(2023~) → 분봉 후보 선정 → 후보 1분봉 1년치
python -m scripts.backfill all

# 2) 상시 실행: 실시간 체결·호가, 장중 잠정 수급, DART 공시 최초 수신 시각, 장마감 후 수집과 증분 백필
python -m scripts.collect_daemon
```

| 데이터 | 수집 방법 | 저장 위치 (`data/lake/`) |
|---|---|---|
| 전 종목 목록 | ka10099 | `universe/` |
| 업종 분류·지수 일봉 | ka10101·ka20002·ka20006 | `universe/sectors_*.parquet`, `index/` |
| 일봉 | ka10081 (수정주가) | `daily/{종목}.parquet` |
| 1분봉 | ka10080, 매일 증분 | `minute/{종목}.parquet` |
| 틱 | ka10079, 오전·장마감 후 구간 | `ticks/YYYYMMDD/` |
| 실시간 체결·호가 | WebSocket 0B·0D 원본 | `realtime/YYYYMMDD/*.jsonl.gz` |
| 장중 잠정 수급 | ka10064, 10분 폴링 | `realtime/YYYYMMDD/investor_intraday.jsonl.gz` |
| 일별 수급(확정) | ka10059 | `investor_daily/` |
| 시간별 프로그램매매 | ka90008 | `program_intraday/YYYYMMDD/` |
| 테마 | ka90001·ka90002 | `themes/` |
| 공시 | DART list.json, 60초 폴링 | `dart/YYYYMMDD.jsonl` |

모든 레코드에 **수집 시각(`received_at`, `rt`, `first_seen`)** 을 남깁니다. 백테스트에서 "그 시각에 알 수 있었던 정보"만 쓰기 위한 기준입니다.
수집 데이터는 용량이 크고 재배포 대상이 아니므로 저장소에 포함하지 않습니다.

## 연구 실험

연구 기간은 실험 전에 나누어 두었습니다. 개발 기간(2023-01~2025-12)으로 탐색하고, **2026-01 이후 홀드아웃은 봉인**되어
`data.dataset.load_daily("holdout")` 이 예외를 냅니다(최종 검증 때만 `RESEARCH_UNSEAL_HOLDOUT=1`).

```bash
python -m scripts.backfill sectors                 # 업종 매핑 + 지수 일봉
python -m experiments.ch05_leaders build           # 5장 특징 계산
python -m experiments.ch05_leaders e0              # E0 기준선 (그 밖에 grid / sector / regime / ranker)
```

## 저장소 구조

```text
config/        환경설정 (.env 로더)
collectors/    키움 REST·WebSocket·DART 수집기
data/          저장 유틸 (수집 데이터는 data/lake/, git 제외)
features/      수급·캔들·VWAP·호가 특징
strategies/    leader_ranker / entry_patterns(P1·P2) / entry_timing(T-1~T4) / exit_policies(S0~S8)
models/        lightgbm / reinforcement_learning
backtest/      시점 재현·비용·체결 시뮬레이션
execution/     주문·정정·취소·포지션 동기화
risk/          독립 위험관리
experiments/   실험 설정과 메타데이터 (experiment_id)
reports/       실험 결과 보고서
scripts/       backfill.py, collect_daemon.py
tests/         단위 테스트
```

## 면책

이 코드는 교육·연구 목적입니다. 실거래에 사용해 발생한 손실에 대해 저자는 책임지지 않습니다.
