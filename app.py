import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：波段強化版", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

ASSET_LIST = {
    "市值前十大公司": {
        "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
        "2881.TW": "富邦金", "2882.TW": "國泰金", "2382.TW": "廣達", "2891.TW": "中信金",
        "3711.TW": "日月光投控", "2412.TW": "中華電"
    },
    "優秀市值型 ETF": {
        "0050.TW": "元大台灣50", "006208.TW": "富邦台50", "00922.TW": "國泰台灣領袖50"
    }
}

@st.cache_data(ttl=300)
def get_full_data(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max", auto_adjust=True)
    if df.empty: return df, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 1. 基礎指標 ---
    df['ma20'] = ta.sma(df['Close'], length=20)
    df['ma20_slope'] = df['ma20'].diff(3) # 月線斜率 (看3天趨勢)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # --- 2. 五維分數核心 (回歸原始邏輯) ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['ma20']) / df['ma20']).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    
    df['Final_Score'] = (df['rsi_r'] * 0.4 + df['bias_r'] * 0.3 + df['macd_r'] * 0.3)
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['is_support'] = df['Final_Score'] <= df['Lower_Bound']
    
    # --- 3. 波段保護線 (ATR Stop) ---
    df['long_stop'] = df['Close'] - (df['atr'] * 2.5) # 下跌超過2.5倍ATR視為波段結束
    
    return df, ticker.info

# --- UI ---
tab1, tab2 = st.tabs(["📡 2025 波段掃描", "🔍 趨勢深度分析"])

with tab1:
    st.subheader("📊 2025 全資產趨勢排行榜")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    for sym, name in all_symbols.items():
        scan_df, _ = get_full_data(sym)
        if not scan_df.empty:
            curr = scan_df.iloc[-1]
            trend = "📈 多頭" if curr['ma20_slope'] > 0 else "📉 空頭"
            signal = "🟡 買點" if curr['is_support'] and curr['ma20_slope'] > 0 else "⚪ 觀望"
            
            # 回測績效
            bt_df = scan_df[scan_df.index >= "2025-01-01"]
            y_days = bt_df[bt_df['is_support']]
            roi = (((1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
            
            radar_results.append({"標的": name, "趨勢": trend, "狀態": signal, "2025回報": f"{roi:.2f}%", "score": curr['Final_Score']})
    
    st.table(pd.DataFrame(radar_results).sort_values("score"))

with tab2:
    st.sidebar.header("🔍 標的選擇")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    sid = [k for k, v in ASSET_LIST[cat].items() if v == asset_name][0]
    
    df, info = get_full_data(sid)
    if not df.empty:
        st.subheader(f"📈 {asset_name} 波段監控圖")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 價格與趨勢線
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], name="月線", line=dict(color="#FF00FF", width=1, dash='dot')), secondary_y=False)
        
        # 五維分數
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2)), secondary_y=True)
        
        # 買點記號
        support_df = df[df['is_support']]
        fig.add_trace(go.Scatter(x=support_df.index, y=support_df['Close'], mode='markers', marker=dict(color="#FFD700", size=8), name="抄底"), secondary_y=False)
        
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=60), df.index[-1]])
        fig.update_layout(height=400, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 分頁式歷史紀錄 ---
        st.markdown("---")
        st.subheader("🏛️ 波段紀錄查詢 (每頁 10 筆)")
        full_h = df.tail(252).copy()
        all_recs = []
        for i in range(len(full_h)-1, -1, -1):
            r = full_h.iloc[i]
            all_recs.append({
                "日期": full_h.index[i].strftime('%Y/%m/%d'),
                "訊號": "🟡 買入" if r['is_support'] else "",
                "趨勢": "向上" if r['ma20_slope'] > 0 else "向下",
                "收盤價": f"{r['Close']:.2f}",
                "波段停損價": f"{r['long_stop']:.1f}"
            })
        
        if 'p_num' not in st.session_state: st.session_state.p_num = 0
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1: 
            if st.button("⬅️ 上一頁") and st.session_state.p_num > 0: st.session_state.p_num -= 1
        with c3: 
            if st.button("下一頁 ➡️") and st.session_state.p_num < (len(all_recs)//10): st.session_state.p_num += 1
            
        start = st.session_state.p_num * 10
        st.table(pd.DataFrame(all_recs[start : start+10]))

        st.info("💡 **波段小撇步**：當「趨勢」顯示為 **向上** 且出現 **🟡 買入** 時，通常是回測月線的絕佳波段進場點。")
