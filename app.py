import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：頂底極端交易系統", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

ASSET_LIST = {
    "市值前十大公司": {
        "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
        "2881.TW": "富邦金", "2882.TW": "國泰金", "2382.TW": "廣達", "2891.TW": "中信金",
        "3711.TW": "日月光投控", "2412.TW": "中華電"
    },
    "優秀市值型 ETF": {
        "0050.TW": "元大台灣50", "006208.TW": "富邦台50", "00922.TW": "國泰台灣領袖50"
    },
    "熱門高股息 ETF": {
        "0056.TW": "元大高股息", "00878.TW": "國泰永續高股息", "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息"
    }
}

@st.cache_data(ttl=300)
def get_full_data(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max", auto_adjust=True)
    if df.empty: return df, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 極端值演算法 ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    df['mfi_r'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14).rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    
    # 綜合分數計算
    def extreme_logic(r):
        if pd.isna(r['adx']): return 50
        # 混合權重
        base = (r['rsi_r'] * 0.3 + r['bias_r'] * 0.3 + r['macd_r'] * 0.2 + r['mfi_r'] * 0.2)
        return base

    df['Final_Score'] = ta.hma(df.apply(extreme_logic, axis=1), length=8)
    
    # 動態極端邊界 (縮窄區間至 10/90 以捕捉更極端的頂底)
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.10)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.90)
    
    # 訊號定義
    df['is_bottom'] = df['Final_Score'] <= df['Lower_Bound'] # 跌無可跌
    df['is_top'] = df['Final_Score'] >= df['Upper_Bound']    # 漲無可漲
    
    return df, ticker.info

# --- UI ---
tab1, tab2 = st.tabs(["📡 2025 頂底排行榜", "🔍 極端區間分析"])

with tab1:
    st.subheader("📊 2025 全資產極端訊號掃描")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    for sym, name in all_symbols.items():
        scan_df, _ = get_full_data(sym)
        if not scan_df.empty:
            curr = scan_df.iloc[-1]
            status = "⚪ 觀望"
            if curr['is_bottom']: status = "🟡 跌無可跌(買)"
            elif curr['is_top']: status = "🔴 漲無可漲(賣)"
            
            # 回測 2025 績效 (僅算抄底買入)
            bt_df = scan_df[scan_df.index >= "2025-01-01"]
            y_days = bt_df[bt_df['is_bottom']]
            roi = (((1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
            
            radar_results.append({"標的": name, "目前價格": round(curr['Close'], 1), "狀態": status, "2025回報": f"{roi:.2f}%", "sort_val": curr['Final_Score']})
    
    st.table(pd.DataFrame(radar_results).sort_values("sort_val"))

with tab2:
    st.sidebar.header("🔍 標的選擇")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    sid = [k for k, v in ASSET_LIST[cat].items() if v == asset_name][0]
    
    df, info = get_full_data(sid)
    if not df.empty:
        st.subheader(f"📈 {asset_name} 頂底轉折圖")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
        
        # 繪製邊界
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], name="頂", line=dict(color="rgba(255, 75, 75, 0.3)", dash='dot')), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], name="底", line=dict(color="rgba(255, 215, 0, 0.3)", dash='dot')), secondary_y=True)
        
        # 標記極端點
        bottoms = df[df['is_bottom']]
        tops = df[df['is_top']]
        fig.add_trace(go.Scatter(x=bottoms.index, y=bottoms['Final_Score'], mode='markers', marker=dict(color="#FFD700", size=10, symbol="triangle-up"), name="跌無可跌"), secondary_y=True)
        fig.add_trace(go.Scatter(x=tops.index, y=tops['Final_Score'], mode='markers', marker=dict(color="#FF4B4B", size=10, symbol="triangle-down"), name="漲無可漲"), secondary_y=True)
        
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=60), df.index[-1]])
        fig.update_layout(height=450, template="plotly_dark", showlegend=False, margin=dict(l=50, r=50, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # 分頁查詢
        st.markdown("---")
        st.subheader("🗓️ 歷史訊號紀錄 (每頁 10 筆)")
        full_h = df.tail(252).copy()
        recs = []
        for i in range(len(full_h)-1, -1, -1):
            r = full_h.iloc[i]
            sig = ""
            if r['is_bottom']: sig = "🟡 買入(底)"
            elif r['is_top']: sig = "🔴 賣出(頂)"
            recs.append({"日期": full_h.index[i].strftime('%Y/%m/%d'), "訊號": sig, "價格": f"{r['Close']:.2f}", "分數": f"{r['Final_Score']:.1f}"})
        
        if 'p3' not in st.session_state: st.session_state.p3 = 0
        c1, c2, c3 = st.columns([1,2,1])
        with c1: 
            if st.button("⬅️ 上一頁"): st.session_state.p3 = max(0, st.session_state.p3-1)
        with c3: 
            if st.button("下一頁 ➡️"): st.session_state.p3 += 1
        
        st.table(pd.DataFrame(recs[st.session_state.p3*10 : st.session_state.p3*10+10]))

        st.info("💡 **操作指南**：當出現 **🟡 黃色向上三角** 時代表跌勢衰竭，適合分批買入；當出現 **🔴 紅色向下三角** 時代表漲勢衰竭，應考慮獲利了結。")
