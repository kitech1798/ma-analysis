# coding: utf-8
"""
이동평균선 분석 로직. 지식.md §3 7단계 체크리스트 + §2 월가 정설 8개 구현.
방어 우선 톤(feedback_investment_philosophy) — 매수 권고가 아닌 위험 신호·일관성 평가에 집중.
"""
import numpy as np
import pandas as pd

MA_PERIODS = [10, 20, 50, 150, 200]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 이평선·기울기·이격도 컬럼 추가."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    close = out['close']
    for p in MA_PERIODS:
        out[f'ma{p}'] = close.rolling(p).mean()
    out['ma200_slope_20d'] = (out['ma200'] - out['ma200'].shift(20)) / out['ma200'].shift(20) * 100
    for p in [20, 50, 200]:
        out[f'dist_ma{p}'] = (close - out[f'ma{p}']) / out[f'ma{p}'] * 100
    return out


def latest_state(df_ind: pd.DataFrame) -> dict:
    last = df_ind.dropna(subset=['ma200']).iloc[-1]
    return {
        'date': df_ind.index[-1],
        'close': float(last['close']),
        'ma10': float(last.get('ma10', np.nan)),
        'ma20': float(last['ma20']),
        'ma50': float(last['ma50']),
        'ma150': float(last['ma150']),
        'ma200': float(last['ma200']),
        'ma200_slope': float(last['ma200_slope_20d']),
        'dist_ma20': float(last['dist_ma20']),
        'dist_ma50': float(last['dist_ma50']),
        'dist_ma200': float(last['dist_ma200']),
        'high_52w': float(df_ind['close'].tail(252).max()),
        'low_52w': float(df_ind['close'].tail(252).min()),
    }


def ma_alignment(s: dict):
    c, m20, m50, m200 = s['close'], s['ma20'], s['ma50'], s['ma200']
    if c > m20 > m50 > m200:
        return ('정배열', '🟢', '가격 > 20 > 50 > 200 — 모든 시간 프레임 상승 정렬')
    if c < m20 < m50 < m200:
        return ('역배열', '🔴', '가격 < 20 < 50 < 200 — 모든 시간 프레임 하락 정렬')
    return ('혼조', '🟡', '일부 정렬 깨짐 — 추세 전환기 또는 박스권')


def weinstein_stage(s: dict):
    """Stan Weinstein 4단계 — 200일선 위/아래 × 200일선 기울기."""
    above = s['close'] > s['ma200']
    slope = s['ma200_slope']
    if abs(slope) < 1.5:
        if above:
            return ('Stage 3 (천장 다지기)', '🟡',
                    '200일선 평탄·가격은 위 — 추세 약화 가능, 관망')
        return ('Stage 1 (바닥 다지기)', '🟡',
                '200일선 평탄·가격은 아래 — 반등 후보 또는 추가 하락 가능')
    if slope > 0 and above:
        return ('Stage 2 (상승)', '🟢',
                '200일선 우상향·가격은 위 — Weinstein 정석 매수 영역')
    if slope < 0 and not above:
        return ('Stage 4 (하락)', '🔴',
                '200일선 우하향·가격은 아래 — Weinstein 절대 매수 금지 영역')
    if slope > 0 and not above:
        return ('Stage 1→2 전환', '🟡',
                '200일선은 상승 시작했으나 가격이 아직 아래 — 돌파 대기')
    return ('Stage 3→4 전환', '🔴',
            '200일선이 꺾이기 시작 — 비중 축소 검토')


def minervini_trend_template(s: dict):
    """Mark Minervini 8개 조건."""
    c = s['close']
    checks = [
        ('가격 > 150일선', c > s['ma150']),
        ('가격 > 200일선', c > s['ma200']),
        ('150일선 > 200일선', s['ma150'] > s['ma200']),
        ('200일선 우상향', s['ma200_slope'] > 0),
        ('50 > 150 > 200 (정배열)', s['ma50'] > s['ma150'] > s['ma200']),
        ('가격 > 50일선', c > s['ma50']),
        ('가격 ≥ 52주 저점 +30%', c >= s['low_52w'] * 1.30),
        ('가격 ≥ 52주 고점 -25%', c >= s['high_52w'] * 0.75),
    ]
    passed = sum(1 for _, ok in checks if ok)
    return checks, passed


def paul_tudor_jones(s: dict):
    if s['close'] > s['ma200']:
        return ('통과', '🟢', '200일선 위 — Paul Tudor Jones 방어 룰 클리어')
    return ('위반', '🔴', '200일선 아래 — PTJ 룰상 비중 축소·청산 영역')


def golden_death_cross(df_ind: pd.DataFrame, lookback: int = 60):
    recent = df_ind.tail(lookback).dropna(subset=['ma50', 'ma200'])
    if len(recent) < 2:
        return ('데이터 부족', '⚪', '교차 판단 불가')
    above = recent['ma50'] > recent['ma200']
    flips = above.ne(above.shift())
    flips.iloc[0] = False
    if not flips.any():
        cur = '위' if above.iloc[-1] else '아래'
        return ('교차 없음', '⚪',
                f'최근 {lookback}거래일 내 교차 없음 — 현재 50일선이 200일선 {cur}에 위치')
    last_flip = flips[flips].index[-1]
    days = (df_ind.index[-1] - last_flip).days
    date_str = last_flip.strftime('%Y년 %m월 %d일')
    if above.loc[last_flip]:
        return ('골든크로스', '🟢',
                f'{date_str}에 50일선이 200일선을 상향 돌파했어요 ({days}일 경과). '
                f'장기 상승 추세로의 전환을 알리는 강세 신호입니다.')
    return ('데드크로스', '🔴',
            f'{date_str}에 50일선이 200일선을 하향 돌파했어요 ({days}일 경과). '
            f'장기 추세가 약세로 꺾였다는 경고입니다.')


def extension_warning(s: dict):
    d = s['dist_ma50']
    if d > 20:
        return ('과열', '🔴', f'50일선 +{d:.1f}% 이격 — 평균회귀 위험 큼')
    if d > 15:
        return ('주의', '🟡', f'50일선 +{d:.1f}% 이격 — 단기 조정 가능성')
    if d < -15:
        return ('과매도', '🟡', f'50일선 {d:.1f}% 이격 — 단기 반등 후보(추세 약세 시 추가 하락 가능)')
    return ('정상', '🟢', f'50일선 {d:+.1f}% 이격 — 정상 범위')


def slope_signal(s: dict):
    slope = s['ma200_slope']
    if slope > 1.5:
        return ('🟢', f'최근 20거래일간 +{slope:.2f}% — 200일선 우상향 견고')
    if slope > 0:
        return ('🟡', f'최근 20거래일간 +{slope:.2f}% — 약한 상승 / 평탄에 가까움')
    if slope > -1.5:
        return ('🟡', f'최근 20거래일간 {slope:.2f}% — 약한 하락 / 평탄에 가까움')
    return ('🔴', f'최근 20거래일간 {slope:.2f}% — 200일선 우하향')


def overall_verdict(s, weinstein, m_passed, ptj, cross, extension):
    """방어 우선 종합. 위험·강세 신호를 가중치+근거 리스트로 누적."""
    risk_reasons, bull_reasons = [], []
    risk = 0

    if ptj[1] == '🔴':
        risk += 2
        risk_reasons.append(
            f"가격(${s['close']:,.2f})이 200일선(${s['ma200']:,.2f}) 아래에 있어요 "
            f"({s['dist_ma200']:+.1f}% 이격) — Paul Tudor Jones 방어 룰 위반"
        )
    if weinstein[1] == '🔴':
        risk += 2
        risk_reasons.append(f"Weinstein {weinstein[0]} — {weinstein[2]}")
    if m_passed <= 3:
        risk += 1
        risk_reasons.append(f"Minervini 강세필터 {m_passed}/8 통과 — 강세 추세 종목 조건 미달")
    if s['dist_ma50'] > 20:
        risk += 1
        risk_reasons.append(
            f"50일선(${s['ma50']:,.2f})보다 +{s['dist_ma50']:.1f}% 위 — 단기 과열, 평균회귀 위험"
        )
    if s['ma200_slope'] < -1.5:
        risk += 1
        risk_reasons.append(
            f"200일선이 최근 20거래일간 {s['ma200_slope']:+.2f}% — 장기 추세 하락 중"
        )
    if cross[1] == '🔴':
        risk += 1
        risk_reasons.append(cross[2])
    if extension[0] == '주의' and s['dist_ma50'] > 0:
        risk_reasons.append(extension[2])

    bull = 0
    if ptj[1] == '🟢':
        bull += 1
        bull_reasons.append(
            f"가격(${s['close']:,.2f})이 200일선(${s['ma200']:,.2f}) 위에 있어요 "
            f"(+{s['dist_ma200']:.1f}% 이격) — Paul Tudor Jones 방어 룰 통과"
        )
    if weinstein[0].startswith('Stage 2'):
        bull += 1
        bull_reasons.append(
            "Weinstein Stage 2 — 200일선이 우상향 중이고 가격도 그 위에 있어 정석 매수 영역"
        )
    if m_passed >= 7:
        bull += 1
        bull_reasons.append(f"Minervini 강세필터 {m_passed}/8 통과 — 강세 추세 종목 조건 충족")
    if cross[1] == '🟢':
        bull += 1
        bull_reasons.append(cross[2])

    if risk >= 3:
        return ('🔴 고위험', '#ef4444',
                '여러 위험 신호가 중첩됐어요 — 신규 매수는 보류, 보유 중이라면 비중·손절선을 재점검하세요.',
                risk_reasons, bull_reasons)
    if risk >= 1:
        return ('🟡 주의', '#f59e0b',
                '약한 경계 신호가 보여요 — 분할 대응·현금 비중 점검, 단기 변동성 확대 가능성에 대비하세요.',
                risk_reasons, bull_reasons)
    if bull >= 2:
        return ('🟢 양호', '#10b981',
                '강세 추세가 유지되고 있어요 — 평소 운영을 지속하되 이격도 점검은 습관화하세요.',
                risk_reasons, bull_reasons)
    return ('🟢 정상', '#71717a',
            '특별한 위험 신호가 없어요 — 평소 운영을 유지하세요.',
            risk_reasons, bull_reasons)


def analyze(df: pd.DataFrame) -> dict:
    df_ind = compute_indicators(df)
    s = latest_state(df_ind)
    align = ma_alignment(s)
    stage = weinstein_stage(s)
    m_checks, m_passed = minervini_trend_template(s)
    ptj = paul_tudor_jones(s)
    cross = golden_death_cross(df_ind)
    ext = extension_warning(s)
    slope = slope_signal(s)
    verdict = overall_verdict(s, stage, m_passed, ptj, cross, ext)
    return {
        'df': df_ind, 'state': s,
        'alignment': align, 'stage': stage,
        'minervini': (m_checks, m_passed),
        'paul_tudor_jones': ptj, 'cross': cross,
        'extension': ext, 'slope': slope,
        'verdict': verdict,
    }
