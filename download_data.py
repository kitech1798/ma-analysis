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


def _batch_download(tickers, start, end, batch_size, mode):
    """tickers를 같은 기간으로 일괄 다운로드."""
    total = len(tickers)
    saved_rows, failed = 0, []
    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        tag = '풀' if mode == 'full' else '증분'
        print(f'  [{tag}|{i+1}-{min(i+batch_size, total)}/{total}] {start}~{end}', flush=True)
        try:
            df = yf.download(batch, start=start, end=end, progress=False,
                             auto_adjust=False, group_by='ticker', threads=True)
        except Exception as e:
            print(f'    배치 실패: {e}')
            failed.extend(batch)
            continue
        for t in batch:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(how='all')
                if sub.empty:
                    if mode == 'full':
                        failed.append(t)
                    # 증분 모드에서 빈 결과는 정상(거래일 아님 또는 휴장)
                    continue
                if mode == 'full' and len(sub) < 50:
                    failed.append(t)
                    continue
                saved_rows += _save_one(t, sub, mode)
            except Exception:
                failed.append(t)
        time.sleep(0.4)
    return saved_rows, failed


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
    if to_full:
        print(f'\n[풀 다운로드] {len(to_full)}개 ({START}~{END})')
        _, fail = _batch_download(to_full, START, END, batch_size, mode='full')
        all_failed.extend(fail)
    for inc_start in sorted(to_inc):
        ts = to_inc[inc_start]
        print(f'\n[증분 다운로드] {len(ts)}개 ({inc_start}~{END})')
        _, fail = _batch_download(ts, inc_start, END, batch_size, mode='incremental')
        all_failed.extend(fail)

    print(f'\n실패: {len(all_failed)}개')
    if all_failed:
        sample = ', '.join(all_failed[:15])
        more = f' 외 {len(all_failed)-15}개' if len(all_failed) > 15 else ''
        print(f'실패 티커: {sample}{more}')
    return all_failed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='S&P500+NASDAQ100 주가 다운로드 (증분/풀)')
    parser.add_argument('--full', action='store_true',
                        help='기존 데이터 무시하고 전체 재다운로드 (월 1회 권장)')
    args = parser.parse_args()

    tickers = build_ticker_list()
    download_prices(tickers, mode='full' if args.full else 'auto')
    print(f'\n완료. 저장 위치: {PRICE_DIR}')
    print('다음: streamlit run "ma_app.py"')
