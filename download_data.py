# coding: utf-8
"""
주가 데이터 다운로드: S&P 500 + NASDAQ 100 → 2020-01-01부터 일별 OHLCV.
한 번 실행하면 데이터/주가/{TICKER}.parquet 로 저장. 재실행 시 새로 받음(증분 갱신 X).

사용: python download_data.py
의존성: yfinance, pandas, lxml(또는 html5lib)
"""
import os
import sys
import time
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def read_html(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text), header=0)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '데이터')
PRICE_DIR = os.path.join(DATA_DIR, '주가')
TICKERS_PATH = os.path.join(DATA_DIR, 'tickers.csv')
# 통합 캐시: 개별 parquet은 .gitignore 대상이라 Streamlit Cloud에 안 올라간다.
# 전 종목을 1개 파일로 묶어 git에 커밋 → 클라우드에서도 로컬 데이터 사용(라이브 페치 차단 회피).
COMBINED_PATH = os.path.join(DATA_DIR, '주가_통합.parquet')
os.makedirs(PRICE_DIR, exist_ok=True)

START = '2020-01-01'
END = datetime.now().strftime('%Y-%m-%d')


def fetch_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = read_html(url)[0]
    df = df.rename(columns={'Symbol': 'ticker', 'Security': 'name', 'GICS Sector': 'sector'})
    df['ticker'] = df['ticker'].astype(str).str.replace('.', '-', regex=False)
    df['index'] = 'S&P500'
    return df[['ticker', 'name', 'sector', 'index']]


def fetch_nasdaq100():
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    tables = read_html(url)
    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if 'ticker' in cols and any(c in cols for c in ['company', 'security']):
            target = t.copy()
            break
    if target is None:
        raise RuntimeError('NASDAQ-100 표 못 찾음 — Wikipedia 구조 변경 가능')
    target.columns = [str(c).lower() for c in target.columns]
    name_col = 'company' if 'company' in target.columns else 'security'
    df = target.rename(columns={'ticker': 'ticker', name_col: 'name'})
    df['ticker'] = df['ticker'].astype(str).str.replace('.', '-', regex=False)
    df['index'] = 'NASDAQ100'
    if 'gics sector' in df.columns:
        df = df.rename(columns={'gics sector': 'sector'})
    if 'sector' not in df.columns:
        df['sector'] = ''
    return df[['ticker', 'name', 'sector', 'index']]


def build_ticker_list():
    print('S&P 500 티커 수집...')
    sp = fetch_sp500()
    print(f'  S&P 500: {len(sp)}개')
    print('NASDAQ 100 티커 수집...')
    nq = fetch_nasdaq100()
    print(f'  NASDAQ 100: {len(nq)}개')
    merged = pd.concat([sp, nq], ignore_index=True)
    grouped = merged.groupby('ticker').agg({
        'name': 'first',
        'sector': 'first',
        'index': lambda x: '+'.join(sorted(set(x)))
    }).reset_index().sort_values('ticker').reset_index(drop=True)
    grouped.to_csv(TICKERS_PATH, index=False, encoding='utf-8-sig')
    print(f'  통합 고유 티커: {len(grouped)}개 → {TICKERS_PATH}')
    return grouped['ticker'].tolist()


def get_existing_last_dates():
    """기존 저장된 종목별 마지막 거래일 매핑 반환."""
    out = {}
    if not os.path.isdir(PRICE_DIR):
        return out
    for fn in os.listdir(PRICE_DIR):
        if not fn.endswith('.parquet'):
            continue
        t = fn[:-len('.parquet')]
        try:
            df = pd.read_parquet(os.path.join(PRICE_DIR, fn))
            if not df.empty:
                out[t] = pd.to_datetime(df.index.max())
        except Exception:
            pass
    return out


def _save_one(ticker, new_df, mode):
    """new_df를 기존 parquet에 병합(증분) 또는 덮어쓰기(풀)."""
    path = os.path.join(PRICE_DIR, f'{ticker}.parquet')
    new_df = new_df.copy()
    new_df.index = pd.to_datetime(new_df.index).normalize()
    if mode == 'incremental' and os.path.exists(path):
        old = pd.read_parquet(path)
        old.index = pd.to_datetime(old.index).normalize()
        merged = pd.concat([old, new_df])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        merged.to_parquet(path)
        return len(new_df)
    new_df.to_parquet(path)
    return len(new_df)


def _batch_download(tickers, start, end, batch_size, mode, threads=True, sleep_between=0.4):
    """tickers를 같은 기간으로 일괄 다운로드. 반환: (saved_rows, failed, inc_empty).
    - failed: 풀 모드 누락(다운로드 실패로 단정 가능)
    - inc_empty: 증분 모드 빈 결과(휴장이거나 yfinance 일시 오류 — 재시도 후보)
    """
    total = len(tickers)
    saved_rows, failed, inc_empty = 0, [], []
    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        tag = '풀' if mode == 'full' else '증분'
        print(f'  [{tag}|{i+1}-{min(i+batch_size, total)}/{total}] {start}~{end}', flush=True)
        try:
            df = yf.download(batch, start=start, end=end, progress=False,
                             auto_adjust=False, group_by='ticker', threads=threads)
        except Exception as e:
            print(f'    배치 실패: {e}')
            (failed if mode == 'full' else inc_empty).extend(batch)
            continue
        for t in batch:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(how='all')
                if sub.empty:
                    (failed if mode == 'full' else inc_empty).append(t)
                    continue
                if mode == 'full' and len(sub) < 50:
                    failed.append(t)
                    continue
                saved_rows += _save_one(t, sub, mode)
            except Exception:
                (failed if mode == 'full' else inc_empty).append(t)
        time.sleep(sleep_between)
    return saved_rows, failed, inc_empty


def download_prices(tickers, mode='auto', batch_size=40):
    """
    mode:
      'auto' — 기존 파일 있으면 증분, 없으면 풀 (권장, 매일 갱신)
      'full' — 강제로 전체 재다운로드
    """
    today_ts = pd.Timestamp(datetime.now().date())
    existing = get_existing_last_dates() if mode == 'auto' else {}

    to_full = []
    to_inc = {}      # start_date(str) → [tickers]
    up_to_date = 0

    for t in tickers:
        last = existing.get(t)
        if mode == 'full' or last is None:
            to_full.append(t)
            continue
        # 마지막 거래일이 오늘 또는 어제(주말 포함) 이상이면 스킵
        if last >= today_ts - pd.Timedelta(days=1):
            up_to_date += 1
            continue
        inc_start = (last + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        to_inc.setdefault(inc_start, []).append(t)

    print(f'\n총 {len(tickers)}개 · 이미 최신 {up_to_date}개 · '
          f'증분 대상 {sum(len(v) for v in to_inc.values())}개 · '
          f'풀 다운로드 {len(to_full)}개')

    all_failed = []
    inc_empty_by_range = {}      # (start, end) → [tickers] : 증분 빈 결과 → 재시도 대상
    if to_full:
        print(f'\n[풀 다운로드] {len(to_full)}개 ({START}~{END})')
        _, fail, _ = _batch_download(to_full, START, END, batch_size, mode='full')
        all_failed.extend(fail)
    for inc_start in sorted(to_inc):
        ts = to_inc[inc_start]
        print(f'\n[증분 다운로드] {len(ts)}개 ({inc_start}~{END})')
        _, fail, empty = _batch_download(ts, inc_start, END, batch_size, mode='incremental')
        all_failed.extend(fail)
        if empty:
            inc_empty_by_range[(inc_start, END)] = empty

    # 증분 빈 결과 재시도 — 작은 배치 + 단일 스레드 + 긴 sleep (yfinance rate limit 회피).
    # 첫 라운드의 큰 배치(40·threads=True)에서 빈 결과로 떨어진 종목들은 대부분 휴장이 아니라
    # 일시적 throttle이라, 작게 다시 던지면 받아짐.
    for (start, end), tickers in list(inc_empty_by_range.items()):
        remaining = tickers
        for attempt in range(2):
            if not remaining:
                break
            print(f'\n[증분 재시도 {attempt+1}/2] {len(remaining)}개 ({start}~{end}) — 배치10·단일스레드')
            _, _, still_empty = _batch_download(
                remaining, start, end, batch_size=10, mode='incremental',
                threads=False, sleep_between=2.0)
            recovered = len(remaining) - len(still_empty)
            print(f'  → {recovered}개 회수, {len(still_empty)}개 남음')
            remaining = still_empty
            if remaining and attempt == 0:
                time.sleep(3)
        if remaining:
            sample = ', '.join(remaining[:10])
            more = f' 외 {len(remaining)-10}개' if len(remaining) > 10 else ''
            print(f'  ⚠ 재시도 후에도 빈 결과: {len(remaining)}개 — {sample}{more}')
            print(f'    (휴장일 가능성이 높지만, 다른 종목과 비교해 누락 의심되면 재실행 권장)')

    print(f'\n실패: {len(all_failed)}개')
    if all_failed:
        sample = ', '.join(all_failed[:15])
        more = f' 외 {len(all_failed)-15}개' if len(all_failed) > 15 else ''
        print(f'실패 티커: {sample}{more}')
    return all_failed


def build_combined_parquet():
    """개별 종목 parquet → 전 종목 1개 통합 파일(zstd).
    컬럼: ticker, date, open, high, low, close, adj close, volume.
    Streamlit Cloud에 git으로 올려 라이브 페치 의존을 없애고 스크리너 로딩도 빠르게."""
    required = {'open', 'high', 'low', 'close', 'volume'}
    frames, skipped = [], []
    if not os.path.isdir(PRICE_DIR):
        print('주가 폴더가 없어 통합 파일을 만들지 못했습니다.')
        return
    for fn in sorted(os.listdir(PRICE_DIR)):
        if not fn.endswith('.parquet'):
            continue
        ticker = fn[:-len('.parquet')]
        try:
            df = pd.read_parquet(os.path.join(PRICE_DIR, fn))
        except Exception as e:
            skipped.append(f'{ticker}(읽기실패: {e})')
            continue
        if df is None or df.empty:
            skipped.append(f'{ticker}(빈 데이터)')
            continue
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        missing = required - set(df.columns)
        if missing:
            skipped.append(f'{ticker}(컬럼누락: {sorted(missing)})')
            continue
        df.index = pd.to_datetime(df.index).normalize()
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})
        df.insert(0, 'ticker', ticker)
        frames.append(df)
    if skipped:
        print(f'  ⚠ 통합 제외 {len(skipped)}종목: ' + ', '.join(skipped[:15])
              + (f' 외 {len(skipped)-15}개' if len(skipped) > 15 else ''))
    if not frames:
        print('통합할 종목 데이터가 없습니다.')
        return
    combined = pd.concat(frames, ignore_index=True)
    combined['ticker'] = combined['ticker'].astype('category')
    combined.to_parquet(COMBINED_PATH, compression='zstd', index=False)
    size_mb = os.path.getsize(COMBINED_PATH) / 1e6
    n_tickers = combined['ticker'].nunique()
    print(f'  통합 파일: {n_tickers}종목 · {len(combined):,}행 · '
          f'{size_mb:.1f}MB → {COMBINED_PATH}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='S&P500+NASDAQ100 주가 다운로드 (증분/풀)')
    parser.add_argument('--full', action='store_true',
                        help='기존 데이터 무시하고 전체 재다운로드 (월 1회 권장)')
    args = parser.parse_args()

    tickers = build_ticker_list()
    failed = download_prices(tickers, mode='full' if args.full else 'auto')

    # 실패율이 높으면 일시적 throttle/네트워크 문제일 수 있다. 개별 parquet은 그대로라
    # 통합 재생성 자체는 무해하므로(건너뛰면 오히려 더 오래된 통합 파일이 남음) 진행하되,
    # 커밋·배포 전 재실행을 권하는 경고를 띄운다.
    fail_ratio = len(failed) / max(len(tickers), 1)
    if fail_ratio > 0.2:
        print(f'\n⚠ 다운로드 실패 {len(failed)}/{len(tickers)}종목 ({fail_ratio:.0%}) — '
              f'네트워크/throttle 가능성이 큽니다.\n'
              f'  통합 파일은 기존 종목 데이터로 재생성하지만, 커밋·배포 전 재실행을 권장합니다.')

    print('\n[통합 파일 생성]')
    build_combined_parquet()
    print(f'\n완료. 저장 위치: {PRICE_DIR}')
    print('다음: streamlit run "ma_app.py"')
