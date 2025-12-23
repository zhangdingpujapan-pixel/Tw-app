import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維全能終端：基本+籌碼版", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# 定義資產清單
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
    if df.empty: return df, {}
    
    # 提取基本面數據
    info = ticker.info
    fundamental_data = {
        "PE": info.get("trailingPE", "N/A"),
        "Yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else "N/A",
        "MarketCap": info.get("marketCap", 0) / 10**12, # 兆
        "52W_High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52W_Low": info.get("fiftyTwoWeekLow", "N/A")
    }

    # 五維指標計算
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

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    df['Support_Dots'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
    df['Resistance_Dots'] = np.where(df['Final_Score'] >= df['Upper_Bound'], df['Final_Score'], np.nan)
    
    return df, fundamental_data

# --- 分頁系統 ---
tab1, tab2 = st.tabs(["📡 績效與基本面雷達", "🔍 深度分析與成交籌碼"])

# --- Tab 1 ---
with tab1:
    st.subheader("📊 2025 全能監測榜 (本益比 + 策略績效)")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    with st.spinner("掃描市場數據中..."):
        for sym, name in all_symbols.items():
            scan_df, f_data = get_full_data(sym)
            if not scan_df.empty:
                curr = scan_df.iloc[-1]
                bt_df = scan_df[scan_df.index >= "2025-01-01"]
                y_days = bt_df[bt_df['Final_Score'] <= bt_df['Lower_Bound']]
                roi = (( (1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
                
                radar_results.append({
                    "標的": name, "目前價格": round(curr['Close'], 1),
                    "本益比(PE)": f_data['PE'] if isinstance(f_data['PE'], str) else round(f_data['PE'], 1),
                    "殖利率(%)": f_data['Yield'] if isinstance(f_data['Yield'], str) else round(f_data['Yield'], 2),
                    "2025績效": f"{roi:.2f}%", "狀態": "🟡 抄底" if curr['Final_Score'] <= curr['Lower_Bound'] else ("🔴 過熱" if curr['Final_Score'] >= curr['Upper_Bound'] else "⚪ 穩定"),
                    "sort_roi": roi
                })
    
    st.table(pd.DataFrame(radar_results).sort_values("sort_roi", ascending=False).drop(columns="sort_roi"))

# --- Tab 2 ---
with tab2:
    st.sidebar.header("🔍 分析設定")
    category = st.sidebar.selectbox("資產類別", list(ASSET_LIST.keys()))
    selected_asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[category].values()))
    stock_id = [k for k, v in ASSET_LIST[category].items() if v == selected_asset_name][0]
    
    df, info = get_full_data(stock_id)
    if not df.empty:
        # 基本面 Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("本益比 (PE)", f"{info['PE']}" if isinstance(info['PE'], str) else f"{info['PE']:.1f}")
        col2.metric("殖利率", f"{info['Yield']}" if isinstance(info['Yield'], str) else f"{info['Yield']:.2f}%")
        col3.metric("市值 (兆)", f"{info['MarketCap']:.2f}T" if info['MarketCap'] > 0 else "N/A")
        col4.metric("52週範圍", f"{info['52W_Low']:.1f} - {info['52W_High']:.1f}")

        # 繪圖 (加入成交量)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                           row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        
        # 價格與指標
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Support_Dots'], mode='markers', marker=dict(color="#FFD700", size=6)), row=1, col=1, secondary_y=True)
        
        # 成交量 (籌碼參考)
        colors = ['red' if df['Open'].iloc[i] > df['Close'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量", marker_color=colors, opacity=0.5), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=60), df.index[-1]])
        st.plotly_chart(fig, use_container_width=True)
