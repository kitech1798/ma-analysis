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
COMBINED_PATH = os.path.join(DATA_DIR, '주가_통합.parquet')

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
.stTabs [data-baseweb="tab-list"] {{
    gap: 10px; border-bottom: none; margin-bottom: 14px; padding: 4px 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_MUTED}; font-size: 16px; font-weight: 700;
    padding: 10px 26px; background: {CARD};
    border: 1px solid {BORDER}; border-radius: 10px;
    transition: color 0.15s, background 0.15s, border-color 0.15s;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT}; border-color: {TEXT_MUTED}; }}
.stTabs [aria-selected="true"] {{
    color: #0a0a0a !important; background: {GREEN} !important;
    border-color: {GREEN} !important;
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
    display: none;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_tickers():
    if not os.path.exists(TICKERS_PATH):
        return pd.DataFrame()
    return pd.read_csv(TICKERS_PATH, encoding='utf-8-sig')


@st.cache_resource
def load_all_prices():
    """통합 parquet → {ticker: OHLCV DataFrame}. 읽기 전용 공유 캐시(복사 없음).

    개별 parquet은 .gitignore 대상이라 Streamlit Cloud엔 없다. git으로 올라온
    통합 파일을 1회 읽어 종목별로 쪼개 둔다 → 라이브 페치 의존 제거 + 스크리너 가속.
    """
    if not os.path.exists(COMBINED_PATH):
        return {}
    combined = pd.read_parquet(COMBINED_PATH)
    combined.columns = [str(c).lower() for c in combined.columns]
    out = {}
    for ticker, g in combined.groupby('ticker', observed=True):
        g = g.drop(columns=['ticker']).copy()
        g['date'] = pd.to_datetime(g['date'])
        g = g.set_index('date').sort_index()
        out[str(ticker)] = g
    return out


@st.cache_data(ttl=21600)
def load_prices(ticker: str):
    """통합 캐시 → 개별 parquet → yfinance 라이브 페치 순으로 시도."""
    allp = load_all_prices()
    if ticker in allp and not allp[ticker].empty:
        # cache_resource 공유 객체 — 호출부 변형이 캐시를 오염시키지 않도록 복사본 반환.
        return allp[ticker].copy()

    path = os.path.join(PRICE_DIR, f'{ticker}.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df.columns = [str(c).lower() for c in df.columns]
        # 통합 경로와 동일하게 날짜 인덱스 정렬을 보장.
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

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


@st.cache_data
def data_reference_date():
    """전 종목 통합 데이터의 최신 거래일(기준일). 데이터 교체는 앱 재시작으로 반영."""
    allp = load_all_prices()
    latest = None
    for df in allp.values():
        if df.empty:
            continue
        d = df.index.max()
        if latest is None or d > latest:
            latest = d
    return latest


VERDICT_ORDER = {'🟢 양호': 0, '🟢 정상': 1, '🟡 주의': 2, '🔴 고위험': 3}


@st.cache_data
def build_screener():
    """전 종목 분석 결과 테이블 (프로세스당 1회 계산, 데이터 교체는 재시작으로 반영)."""
    allp = load_all_prices()
    tickers_df = load_tickers()
    meta = tickers_df.set_index('ticker') if not tickers_df.empty else pd.DataFrame()
    rows = []
    for ticker, df in allp.items():
        try:
            r = analyze(df)
        except Exception:
            continue
        s = r['state']
        _, m_pass = r['minervini']
        info = meta.loc[ticker] if ticker in meta.index else None

        def _meta(col, fallback):
            if info is None:
                return fallback
            val = info[col]
            return str(val) if pd.notna(val) else fallback

        rows.append({
            'ticker': ticker,
            'name': _meta('name', ticker),
            'sector': _meta('sector', ''),
            'index': _meta('index', ''),
            '종합판정': r['verdict'][0],
            '강세필터': int(m_pass),
            'Stage': r['stage'][0],
            '현재가': round(s['close'], 2),
            '200일이격%': round(s['dist_ma200'], 1),
            '50일이격%': round(s['dist_ma50'], 1),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['_정렬'] = out['종합판정'].map(VERDICT_ORDER).fillna(9)
    out = out.sort_values(['_정렬', '강세필터'], ascending=[True, False]).drop(columns='_정렬')
    return out.reset_index(drop=True)


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


def render_detail(ticker, tickers_df, ma_to_show, show_volume, date_range, key_prefix='detail'):
    """단일 종목 7단계 분석 화면."""
    df = load_prices(ticker)
    if df is None or df.empty:
        st.error(
            f'{ticker}: 주가 데이터를 불러오지 못했습니다. '
            f'`python download_data.py` 로 데이터를 갱신한 뒤 다시 시도하세요.'
        )
        return

    result = analyze(df)
    s = result['state']
    meta_rows = tickers_df[tickers_df['ticker'] == ticker]
    name = meta_rows.iloc[0]['name'] if not meta_rows.empty else ticker
    sector = meta_rows.iloc[0]['sector'] if not meta_rows.empty else ''
    index_name = meta_rows.iloc[0]['index'] if not meta_rows.empty else ''

    # ── 헤더
    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    with h1:
        st.subheader(f"{name} ({ticker})")
        st.markdown(
            f"<p style='color:{TEXT_MUTED};font-size:12.5px;margin-top:-8px;'>"
            f"{sector} · {index_name} · 기준일 {s['date'].strftime('%Y-%m-%d')}</p>",
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
    st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_chart_{ticker}')

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
        st.dataframe(rows_data, hide_index=True, use_container_width=True,
                     key=f'{key_prefix}_metrics_{ticker}')


def render_screener(tickers_df, ma_to_show, show_volume, date_range):
    """전 종목 모아보기 — 종합판정·강세필터·섹터·지수로 필터링."""
    table = build_screener()
    if table.empty:
        st.warning('분석할 종목 데이터가 없습니다. `python download_data.py` 를 먼저 실행하세요.')
        return

    st.markdown(
        f"<p style='color:{TEXT_MUTED};font-size:13px;margin-bottom:10px;'>"
        f"전 종목을 같은 7단계 기준으로 평가했어요. 아래 필터로 좁히고, "
        f"행을 클릭하면 해당 종목 상세 분석이 펼쳐집니다.</p>",
        unsafe_allow_html=True,
    )

    # ── 필터
    verdict_opts = [v for v in VERDICT_ORDER if v in set(table['종합판정'])]
    c1, c2 = st.columns([2, 1])
    with c1:
        sel_verdict = st.multiselect('종합 판정', verdict_opts, default=verdict_opts)
    with c2:
        min_bull = st.slider('강세필터 최소 점수', 0, 8, 0,
                             help='Minervini 8조건 중 통과 개수 하한')

    c3, c4, c5 = st.columns([1, 2, 2])
    with c3:
        index_opts = sorted(set(table['index']))
        sel_index = st.multiselect('지수', index_opts, default=index_opts)
    with c4:
        sector_opts = sorted(x for x in set(table['sector']) if x)
        sel_sector = st.multiselect('섹터', sector_opts, default=[])
    with c5:
        query = st.text_input('티커·종목명 검색', '').strip().lower()

    view = table.copy()
    if sel_verdict:
        view = view[view['종합판정'].isin(sel_verdict)]
    view = view[view['강세필터'] >= min_bull]
    if sel_index:
        view = view[view['index'].isin(sel_index)]
    if sel_sector:
        view = view[view['sector'].isin(sel_sector)]
    if query:
        mask = (view['ticker'].str.lower().str.contains(query, na=False)
                | view['name'].str.lower().str.contains(query, na=False))
        view = view[mask]

    # ── 요약 카운트
    counts = table['종합판정'].value_counts()
    chips = ' &nbsp; '.join(
        f"<span style='color:{TEXT};font-weight:600;'>{v}</span> "
        f"<span style='color:{TEXT_MUTED};'>{counts.get(v, 0)}</span>"
        for v in verdict_opts
    )
    st.markdown(
        f"<div style='margin:6px 0 12px;font-size:13px;'>"
        f"<span style='color:{TEXT_MUTED};'>필터 결과 </span>"
        f"<span style='color:{TEXT};font-weight:700;'>{len(view)}</span>"
        f"<span style='color:{TEXT_MUTED};'> / 전체 {len(table)}종목 &nbsp;·&nbsp; </span>{chips}</div>",
        unsafe_allow_html=True,
    )

    show_cols = ['ticker', 'name', 'sector', 'index', '종합판정', '강세필터',
                 'Stage', '현재가', '200일이격%', '50일이격%']
    event = st.dataframe(
        view[show_cols],
        hide_index=True,
        use_container_width=True,
        height=520,
        on_select='rerun',
        selection_mode='single-row',
        key='screener_table',
        column_config={
            'ticker': st.column_config.TextColumn('티커', width='small'),
            'name': st.column_config.TextColumn('종목명'),
            'sector': st.column_config.TextColumn('섹터'),
            'index': st.column_config.TextColumn('지수', width='small'),
            '종합판정': st.column_config.TextColumn('종합판정', width='small'),
            '강세필터': st.column_config.ProgressColumn(
                '강세필터', min_value=0, max_value=8, format='%d/8'),
            'Stage': st.column_config.TextColumn('Weinstein'),
            '현재가': st.column_config.NumberColumn('현재가', format='$%.2f'),
            '200일이격%': st.column_config.NumberColumn('200일이격', format='%+.1f%%'),
            '50일이격%': st.column_config.NumberColumn('50일이격', format='%+.1f%%'),
        },
    )

    sel = event.selection.rows if event and event.selection else []
    if sel:
        picked = view.iloc[sel[0]]['ticker']
        st.divider()
        st.markdown(
            f"<p style='color:{GREEN};font-size:13px;font-weight:600;margin-bottom:4px;'>"
            f"▼ {picked} 상세 분석</p>", unsafe_allow_html=True)
        render_detail(picked, tickers_df, ma_to_show, show_volume, date_range,
                      key_prefix='screen')
    else:
        st.caption('행을 클릭하면 그 종목의 7단계 상세 분석이 여기 펼쳐집니다.')


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

    ref_date = data_reference_date()
    ref_str = ref_date.strftime('%Y년 %m월 %d일') if ref_date is not None else '데이터 없음'

    # ── 데이터 기준일 (상단 가운데 크게)
    st.markdown(
        f"<div style='text-align:center;margin:6px 0 22px;'>"
        f"<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;"
        f"color:{TEXT_MUTED};margin-bottom:2px;'>데이터 기준일</div>"
        f"<div style='font-size:32px;font-weight:800;color:{TEXT};letter-spacing:0.5px;'>{ref_str}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header('종목 선택')
        options = [f"{r.ticker} — {r['name']}" for _, r in tickers_df.iterrows()]
        choice = st.selectbox('티커 검색 (입력하면 필터링)', options=options, index=0)
        ticker = choice.split(' — ')[0]

        st.divider()
        st.header('차트 설정')
        ma_to_show = st.multiselect('이평선 표시', [10, 20, 50, 150, 200],
                                    default=[20, 50, 200])
        show_volume = st.checkbox('거래량 표시', value=True)
        date_range = st.radio('기간', ['최근 6개월', '최근 1년', '최근 3년', '전체(2020~)'],
                              index=1)

    tab_detail, tab_screen = st.tabs(['개별 분석', '모아보기'])
    with tab_detail:
        render_detail(ticker, tickers_df, ma_to_show, show_volume, date_range)
    with tab_screen:
        render_screener(tickers_df, ma_to_show, show_volume, date_range)

    st.markdown(
        f"<p style='color:{TEXT_MUTED};font-size:11.5px;margin-top:24px;'>"
        f"본 분석은 보조 도구입니다. 매수·매도 답안이 아닌 위험 신호 점검용. "
        f"투자 판단의 책임은 본인에게.</p>",
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
