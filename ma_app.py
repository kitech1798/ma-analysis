# coding: utf-8
"""
이동평균선 분석 — Streamlit 인터페이스 (다크 테크 톤).
실행: streamlit run "ma_app.py"
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_logic import analyze

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '데이터')
PRICE_DIR = os.path.join(DATA_DIR, '주가')
TICKERS_PATH = os.path.join(DATA_DIR, 'tickers.csv')

# 다크 테크 톤 팔레트
BG = '#0a0a0a'
SURFACE = '#161618'
CARD = '#1c1c1f'
BORDER = '#2a2a2e'
TEXT = '#fafafa'
TEXT_MUTED = '#a1a1aa'
GREEN = '#10b981'
AMBER = '#f59e0b'
RED = '#ef4444'
NEUTRAL = '#71717a'

MA_COLORS = {10: '#71717a', 20: '#06b6d4', 50: '#f59e0b', 150: '#a855f7', 200: '#ef4444'}

st.set_page_config(page_title='이동평균선 분석', layout='wide', page_icon='📊')

st.markdown(f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.3px; }}
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 12px 16px; }}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-size: 12px !important; }}
[data-testid="stMetricValue"] {{ color: {TEXT} !important; font-size: 22px !important; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
hr {{ border-color: {BORDER}; }}
.stExpander {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; }}
.stExpander > details > summary {{ color: {TEXT}; }}
[data-testid="stDataFrame"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; }}
.stRadio label, .stCheckbox label, .stMultiSelect label, .stSelectbox label {{ color: {TEXT_MUTED}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_tickers():
    if not os.path.exists(TICKERS_PATH):
        return pd.DataFrame()
    return pd.read_csv(TICKERS_PATH, encoding='utf-8-sig')


@st.cache_data(ttl=21600)
def load_prices(ticker: str):
    """로컬 parquet 우선, 없으면 yfinance 라이브 페치 (6시간 캐시)."""
    path = os.path.join(PRICE_DIR, f'{ticker}.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        try:
            df = yf.download(ticker, start='2020-01-01', progress=False, auto_adjust=False)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).normalize()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def signal_card(label, badge, text):
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:14px;padding:14px 18px;'
        f'background:{CARD};border:1px solid {BORDER};border-radius:8px;margin:6px 0;">'
        f'<span style="font-size:20px;line-height:1.3;flex-shrink:0;">{badge}</span>'
        f'<div style="flex:1;">'
        f'<div style="font-weight:600;color:{TEXT};font-size:14px;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{TEXT_MUTED};font-size:13px;line-height:1.6;">{text}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def main():
    st.title('이동평균선 분석')
    st.markdown(
        f"<p style='color:{TEXT_MUTED};margin-top:-12px;font-size:13px;'>"
        f"20·50·200일선 기준 · S&P 500 + NASDAQ 100 · 큰 실수 회피 우선</p>",
        unsafe_allow_html=True,
    )

    tickers_df = load_tickers()
    if tickers_df.empty:
        st.error('티커 데이터가 없습니다. `python download_data.py` 를 먼저 실행하세요.')
        st.stop()

    with st.sidebar:
        st.header('종목 선택')
        options = [f"{r.ticker} — {r['name']}" for _, r in tickers_df.iterrows()]
        choice = st.selectbox('티커 검색 (입력하면 필터링)', options=options, index=0)
        ticker = choice.split(' — ')[0]

        st.divider()
        st.header('차트 설정')
        ma_to_show = st.multiselect('이평선 표시', [10, 20, 50, 150, 200], default=[20, 50, 200])
        show_volume = st.checkbox('거래량 표시', value=True)
        date_range = st.radio('기간', ['최근 6개월', '최근 1년', '최근 3년', '전체(2020~)'], index=1)

    df = load_prices(ticker)
    if df is None or df.empty:
        st.error(f'{ticker}: 주가 데이터 파일이 없습니다.')
        st.stop()

    result = analyze(df)
    s = result['state']
    meta = tickers_df[tickers_df['ticker'] == ticker].iloc[0]

    # ── 헤더
    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    with h1:
        st.subheader(f"{meta['name']} ({ticker})")
        st.markdown(
            f"<p style='color:{TEXT_MUTED};font-size:12.5px;margin-top:-8px;'>"
            f"{meta['sector']} · {meta['index']} · 기준일 {s['date'].strftime('%Y-%m-%d')}</p>",
            unsafe_allow_html=True,
        )
    h2.metric('현재가', f"${s['close']:,.2f}", help='기준일 종가')
    h3.metric('종합 판정', result['verdict'][0], help='7개 지표를 가중합한 방어 우선 판정')
    _, m_pass = result['minervini']
    h4.metric(
        '강세 필터', f'{m_pass}/8',
        help=(
            'Mark Minervini의 8조건 강세 추세 필터.\n'
            '가격이 50/150/200일선 위, 정배열, 200일선 우상향, 52주 저점 +30% 이상, 고점 -25% 이내 등.\n'
            '7개 이상 통과 시 강세 추세 종목으로 분류 (추세 매수 후보).'
        ),
    )

    # ── 종합 배너 + 근거
    v_label, v_color, v_text, risk_reasons, bull_reasons = result['verdict']

    def reasons_block(title, items, icon, accent):
        if not items:
            return ''
        rendered = ''.join(f'<li style="margin:5px 0;">{r}</li>' for r in items)
        return (
            f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid {BORDER};">'
            f'<div style="font-size:11px;color:{accent};font-weight:700;margin-bottom:6px;'
            f'text-transform:uppercase;letter-spacing:0.8px;">{icon} {title}</div>'
            f'<ul style="margin:0;padding-left:20px;color:{TEXT};font-size:13.5px;line-height:1.65;">'
            f'{rendered}</ul></div>'
        )

    reasons_html = reasons_block('왜 이런 판정이 나왔나', risk_reasons, '⚠️', v_color)
    reasons_html += reasons_block('동시에 보이는 강세 신호', bull_reasons, '✓', GREEN)

    st.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-left:4px solid {v_color};padding:20px 24px;border-radius:10px;'
        f'margin:16px 0 28px;">'
        f'<div style="font-size:22px;font-weight:700;color:{v_color};letter-spacing:-0.3px;">{v_label}</div>'
        f'<div style="font-size:14px;margin-top:8px;color:{TEXT};line-height:1.6;">{v_text}</div>'
        f'{reasons_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 차트
    df_chart = result['df'].copy()
    n = {'최근 6개월': 126, '최근 1년': 252, '최근 3년': 252 * 3}.get(date_range)
    if n:
        df_chart = df_chart.tail(n)

    rows = 2 if show_volume else 1
    heights = [0.75, 0.25] if show_volume else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=heights)
    fig.add_trace(go.Candlestick(
        x=df_chart.index, open=df_chart['open'], high=df_chart['high'],
        low=df_chart['low'], close=df_chart['close'],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
        name='가격', showlegend=False,
    ), row=1, col=1)
    for p in ma_to_show:
        col = f'ma{p}'
        if col in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart[col], mode='lines',
                name=f'{p}일선', line=dict(color=MA_COLORS[p], width=1.6),
            ), row=1, col=1)
    if show_volume:
        fig.add_trace(go.Bar(
            x=df_chart.index, y=df_chart['volume'],
            marker_color='#3f3f46', showlegend=False, name='거래량',
        ), row=2, col=1)

    axis_style = dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, color=TEXT_MUTED)
    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, size=12, family='sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                    bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT)),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    st.plotly_chart(fig, use_container_width=True)

    # ── 7단계 해석
    st.subheader('7단계 해석')

    signal_card('① 200일선 (Paul Tudor Jones 룰)', result['paul_tudor_jones'][1],
                f"{result['paul_tudor_jones'][0]} — {result['paul_tudor_jones'][2]}")
    signal_card('② 이평선 정렬 (정배열/역배열)', result['alignment'][1],
                f"{result['alignment'][0]} — {result['alignment'][2]}")
    signal_card('③ Weinstein Stage (사이클 단계)', result['stage'][1],
                f"{result['stage'][0]} — {result['stage'][2]}")
    signal_card('④ 골든크로스 / 데드크로스 (50일선 ↔ 200일선)',
                result['cross'][1],
                f"{result['cross'][0]} — {result['cross'][2]}")
    signal_card('⑤ 50일선 이격도 (단기 과열도)', result['extension'][1],
                f"{result['extension'][0]} — {result['extension'][2]}")
    signal_card('⑥ 200일선 기울기 (장기 추세 방향)', result['slope'][0], result['slope'][1])

    m_checks, m_passed = result['minervini']
    m_badge = '🟢' if m_passed >= 7 else ('🟡' if m_passed >= 5 else '🔴')
    signal_card(
        '⑦ Minervini 강세필터 (8조건 종합)', m_badge,
        f'{m_passed}/8 통과 — Mark Minervini가 만든 강세 추세 종목 체크리스트. '
        f'7개 이상 통과해야 "추세가 살아있는 종목"으로 분류해 매수 후보로 고려합니다.'
    )

    with st.expander(f'Minervini 강세 추세 8조건 상세 ({m_passed}/8 통과)'):
        st.caption('Mark Minervini의 SEPA 전략 핵심 — 8개 모두 통과하는 종목만 추세 매수 후보로 거래')
        for label, ok in m_checks:
            st.write(f"{'✅' if ok else '❌'} {label}")

    with st.expander('지표 수치 상세'):
        rows_data = pd.DataFrame([
            ('현재가', f"${s['close']:,.2f}", '-'),
            ('20일선', f"${s['ma20']:,.2f}", f"{s['dist_ma20']:+.2f}% 이격"),
            ('50일선', f"${s['ma50']:,.2f}", f"{s['dist_ma50']:+.2f}% 이격"),
            ('150일선', f"${s['ma150']:,.2f}", '-'),
            ('200일선', f"${s['ma200']:,.2f}", f"{s['dist_ma200']:+.2f}% 이격"),
            ('200일선 기울기', f"{s['ma200_slope']:+.2f}%", '최근 20거래일 변화율'),
            ('52주 고점', f"${s['high_52w']:,.2f}", f"{(s['close']/s['high_52w']-1)*100:+.1f}% 차이"),
            ('52주 저점', f"${s['low_52w']:,.2f}", f"{(s['close']/s['low_52w']-1)*100:+.1f}% 차이"),
        ], columns=['항목', '값', '해석'])
        st.dataframe(rows_data, hide_index=True, use_container_width=True)

    st.markdown(
        f"<p style='color:{TEXT_MUTED};font-size:11.5px;margin-top:24px;'>"
        f"본 분석은 보조 도구입니다. 매수·매도 답안이 아닌 위험 신호 점검용. "
        f"투자 판단의 책임은 본인에게.</p>",
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
