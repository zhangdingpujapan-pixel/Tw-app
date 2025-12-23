import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面設定
st.set_page_config(page_title="五維量化交易終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# 定義標的清單
ASSET_LIST = {
    "市值前十大": {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電", "2881.TW": "富邦金", "2882.TW": "國泰金", "2382.TW": "廣達", "2891.TW": "中信金", "3711.TW": "日月光", "2412.TW": "中華電"},
    "市值型ETF": {"0050.TW": "元大50", "006208.TW": "富邦50", "00922.TW": "國泰領袖50"},
    "高股息ETF": {"0056.TW": "元大高股息", "00878.TW": "國泰高股息", "00919.TW": "群益高息", "00929.TW": "復華高息"}
}

@st.cache_data(ttl=300)
def get_pro_data(symbol):
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 核心指標
    df['ma200'] = ta.sma(df['Close'], length=200)
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    def adaptive_logic(r):
        if pd.isna(r['adx']) or pd.isna(r['atr']): return 50
        base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1) if r['adx'] > 25 else (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        return (base + 50) / 2 if (abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0) > 2.5 else base

    df['Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    df['L_Bound'] = df['Score'].rolling(252).quantile(0.15)
    df['U_Bound'] = df['Score'].rolling(252).quantile(0.85)
    df['Buy_Sig'] = (df['Score'] <= df['L_Bound'])
    df['Sell_Sig'] = (df['Score'] >= df['U_Bound'])
    return df

def run_backtest(df, initial_cap=1000000):
    bt_df = df[df.index >= "2025-01-01"].copy()
    if bt_df.empty: return 0, 0, 0, 0
    
    cash = initial_cap
    shares = 0
    equity_curve = []
    buy_days = bt_df[bt_df['Buy_Sig']]
    
    # 簡化回測：黃點出現當天投入可用現金的 20%，紅點出現全部賣出
    for date, row in bt_df.iterrows():
        if row['Buy_Sig'] and cash > 0:
            invest = cash * 0.2
            shares += invest / row['Close']
            cash -= invest
        elif row['Sell_Sig'] and shares > 0:
            cash += shares * row['Close']
            shares = 0
        current_val = cash + (shares * row['Close'])
        equity_curve.append(current_val)
    
    final_val = equity_curve[-1]
    roi = (final_val - initial_cap) / initial_cap * 100
    
    # MDD 計算
    peak = pd.Series(equity_curve).expanding().max()
    dd = (pd.Series(equity_curve) - peak) / peak
    mdd = dd.min() * 100
    
    # 定期定額對照 (每月初)
    dca_df = bt_df.resample('MS').first()
    dca_shares = (initial_cap / len(dca_df) / dca_df['Close']).sum()
    dca_roi = ((dca_shares * bt_df['Close'].iloc[-1] - initial_cap) / initial_cap) * 100
    
    return final_val, roi, mdd, dca_roi

# --- UI 分頁 ---
tab1, tab2 = st.tabs(["📡 績效雷達總表", "🔍 深度量化分析"])

all_symbols = {}
for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])

with tab1:
    st.subheader("📊 2025 策略績效全掃描 (100萬本金)")
    summary = []
    with st.spinner("量化引擎運算中..."):
        for sym, name in all_symbols.items():
            d = get_pro_data(sym)
            f_val, roi, mdd, dca_roi = run_backtest(d)
            curr = d.iloc[-1]
            trend = "📈 多頭" if curr['Close'] > curr['ma200'] else "📉 空頭"
            status = "🟡 買點" if curr['Buy_Sig'] else ("🔴 賣點" if curr['Sell_Sig'] else "⚪ 觀望")
            summary.append({"標的": name, "目前價格": f"{curr['Close']:.1f}", "趨勢": trend, "狀態": status, "策略回報": f"{roi:.2f}%", "定期定額": f"{dca_roi:.2f}%", "最大回撤": f"{mdd:.1f}%"})
    
    st.dataframe(pd.DataFrame(summary).sort_values("策略回報", ascending=False), use_container_width=True)

with tab2:
    st.sidebar.header("🔍 深度量化選擇")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    s_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    s_id = [k for k, v in ASSET_LIST[cat].items() if v == s_name][0]
    
    df = get_pro_data(s_id)
    if not df.empty:
        curr = df.iloc[-1]
        color = "#00d26a" if curr['Close'] > df['Close'].iloc[-2] else "#FF4B4B"
        st.markdown(f"### {s_name} ({s_id}) <span style='color:{color}'>{curr['Close']:.2f}</span>", unsafe_allow_html=True)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # 股價與 200MA
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="white", width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['ma200'], name="200MA", line=dict(color="gray", width=1, dash='dash')), secondary_y=False)
        # 五維分數與邊界
        fig.add_trace(go.Scatter(x=df.index, y=df['Score'], name="檔", line=dict(color="#00BFFF", width=2)), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['U_Bound'], name="壓", line=dict(color="rgba(255, 75, 75, 0.3)", width=1, dash='dot')), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['L_Bound'], name="撐", line=dict(color="rgba(255, 215, 0, 0.3)", width=1, dash='dot')), secondary_y=True)
        
        # 標記訊號
        buy_dots = df[df['Buy_Sig']]
        sell_dots = df[df['Sell_Sig']]
        fig.add_trace(go.Scatter(x=buy_dots.index, y=buy_dots['Score'], mode='markers', name='買', marker=dict(color="#FFD700", size=7)), secondary_y=True)
        fig.add_trace(go.Scatter(x=sell_dots.index, y=sell_dots['Score'], mode='markers', name='賣', marker=dict(color="#FF4B4B", size=7)), secondary_y=True)
        
        fig.update_layout(height=500, template="plotly_dark", dragmode="pan", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_yaxes(secondary_y=True, range=[-5, 105], fixedrange=True)
        fig.update_xaxes(range=[df.index[-30], df.index[-1]])
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示回測詳細指標
        f_val, roi, mdd, dca_roi = run_backtest(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("策略預期價值", f"${f_val:,.0f}", f"{roi:.2f}%")
        c2.metric("最大壓力 (MDD)", f"{mdd:.2f}%")
        c3.metric("趨勢過濾 (200MA)", "偏多" if curr['Close'] > curr['ma200'] else "偏空")
