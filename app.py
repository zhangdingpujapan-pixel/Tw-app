import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：檔位深度優化版", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

ASSET_LIST = {
    "市值前十大公司": {
        "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
        "2881.TW": "富邦金", "2882.TW": "國泰金", "2382.TW": "廣達", "2891.TW": "中信金",
        "3711.TW": "日月光投控", "2412.TW": "中華電"
    },
    "熱門 ETF": {
        "0050.TW": "元大台灣50", "0056.TW": "元大高股息", "00878.TW": "國泰永續高股息", "00919.TW": "群益精選高息"
    }
}

@st.cache_data(ttl=300)
def get_optimized_data(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max", auto_adjust=True)
    if df.empty: return df, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 優化版指標計算 ---
    # 1. 核心五維因子 (保持原始高報酬權重)
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    
    # 2. 檔位線平滑優化：使用 HMA 取代簡單移動平均
    raw_scores = (df['rsi_r'] * 0.4 + df['bias_r'] * 0.4 + df['macd_r'] * 0.2)
    df['Final_Score'] = ta.hma(raw_scores, length=10)
    
    # 3. 自適應邊界：結合百分位與標準差，捕捉真正的極端值
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    
    df['is_support'] = df['Final_Score'] <= df['Lower_Bound']
    
    return df, ticker.info

# --- UI 介面 ---
tab1, tab2 = st.tabs(["📡 實時訊號排行榜", "🔍 檔位深度診斷"])

with tab1:
    st.subheader("📊 2025 優化引擎績效監測")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    for sym, name in all_symbols.items():
        scan_df, _ = get_optimized_data(sym)
        if not scan_df.empty:
            curr = scan_df.iloc[-1]
            # 報酬率回測邏輯
            bt_df = scan_df[scan_df.index >= "2025-01-01"]
            y_days = bt_df[bt_df['is_support']]
            roi = (((1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
            
            status = "🟡 抄底區" if curr['is_support'] else "⚪ 正常"
            radar_results.append({
                "標的": name, "價格": round(curr['Close'], 1), 
                "2025回報": f"{roi:.2f}%", "狀態": status, "檔位分數": round(curr['Final_Score'], 1)
            })
    st.table(pd.DataFrame(radar_results).sort_values("2025回報", ascending=False))

with tab2:
    st.sidebar.header("🔍 分析設定")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    sid = [k for k, v in ASSET_LIST[cat].items() if v == asset_name][0]
    
    df, info = get_optimized_data(sid)
    if not df.empty:
        st.subheader(f"📈 {asset_name}：深度檔位圖表")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 繪製 K 線簡化版 (收盤價)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=2)), secondary_y=False)
        
        # 繪製優化後的檔位線 (HMA)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔位線(HMA)", line=dict(color="#00BFFF", width=3)), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], line=dict(color="rgba(255, 215, 0, 0.4)", dash='dot')), secondary_y=True)
        
        # 標記抄底點
        support_df = df[df['is_support']]
        fig.add_trace(go.Scatter(x=support_df.index, y=support_df['Final_Score'], mode='markers', marker=dict(color="#FFD700", size=10, symbol="star")), secondary_y=True)
        
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=60), df.index[-1]])
        fig.update_layout(height=450, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 歷史分頁紀錄 ---
        st.markdown("---")
        st.subheader("🏛️ 歷史分頁紀錄 (10筆/頁)")
        full_history = df.tail(252).copy()
        recs = []
        for i in range(len(full_history)-1, -1, -1):
            row = full_history.iloc[i]
            recs.append({
                "日期": full_history.index[i].strftime('%Y/%m/%d'),
                "訊號": "🟡 抄底" if row['is_support'] else "",
                "收盤價": f"{row['Close']:.2f}",
                "檔位分數": f"{row['Final_Score']:.1f}"
            })
        
        if 'p_idx' not in st.session_state: st.session_state.p_idx = 0
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1: 
            if st.button("⬅️ 上一頁"): st.session_state.p_idx = max(0, st.session_state.p_idx - 1)
        with c3: 
            if st.button("下一頁 ➡️"): st.session_state.p_idx += 1
        
        st.table(pd.DataFrame(recs[st.session_state.p_idx * 10 : st.session_state.p_idx * 10 + 10]))
