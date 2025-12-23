import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# 定義完整股票清單
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
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 指標計算
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    def adaptive_logic(r):
        if pd.isna(r['adx']) or pd.isna(r['atr']): return 50
        vol_ratio = abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0
        base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1) if r['adx'] > 25 else (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        return (base + 50) / 2 if vol_ratio > 2.5 else base

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    df['Support_Dots'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
    df['Resistance_Dots'] = np.where(df['Final_Score'] >= df['Upper_Bound'], df['Final_Score'], np.nan)
    return df

# --- 主程式分頁 ---
tab1, tab2 = st.tabs(["📡 全資產訊號雷達", "🔍 單一標深度分析"])

# --- Tab 1: 全資產監測 ---
with tab1:
    st.subheader("📡 市場掃描儀：即時訊號監測")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    with st.spinner("掃描所有標的中..."):
        for sym, name in all_symbols.items():
            scan_df = get_full_data(sym)
            if not scan_df.empty:
                curr = scan_df.iloc[-1]
                prev = scan_df.iloc[-2]
                change = curr['Close'] - prev['Close']
                change_pct = (change / prev['Close']) * 100
                status = "⚪ 區間穩定"
                if curr['Final_Score'] <= curr['Lower_Bound']: status = "🟡 抄底訊號"
                elif curr['Final_Score'] >= curr['Upper_Bound']: status = "🔴 過熱警告"
                
                radar_results.append({
                    "代碼": sym, "名稱": name, "目前價格": f"{curr['Close']:.2f}",
                    "今日漲跌": f"{'+' if change > 0 else ''}{change:.2f} ({change_pct:.2f}%)",
                    "五維分數": f"{curr['Final_Score']:.1f}", "狀態": status
                })
    
    radar_df = pd.DataFrame(radar_results)
    radar_df['sort_val'] = radar_df['狀態'].apply(lambda x: 0 if "🟡" in x else (2 if "🔴" in x else 1))
    st.table(radar_df.sort_values("sort_val").drop(columns="sort_val"))

# --- Tab 2: 深度分析 ---
with tab2:
    st.sidebar.header("🔍 標的選擇")
    category = st.sidebar.selectbox("資產類別", list(ASSET_LIST.keys()))
    selected_asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[category].values()))
    stock_id = [k for k, v in ASSET_LIST[category].items() if v == selected_asset_name][0]
    
    df = get_full_data(stock_id)
    if not df.empty:
        # 標題與報價
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        diff = curr_price - prev_price
        diff_pct = (diff / prev_price) * 100
        color = "#FF4B4B" if diff < 0 else "#00d26a"
        st.markdown(f"### 🛡️ {selected_asset_name} ({stock_id}) <span style='color:{color};'>{curr_price:.2f} ({'+' if diff > 0 else ''}{diff:.2f}, {diff_pct:.2f}%)</span>", unsafe_allow_html=True)
        
        # 繪圖
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], name="壓", line=dict(color="rgba(255, 75, 75, 0.3)", width=1, dash='dot')), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], name="撐", line=dict(color="rgba(255, 215, 0, 0.3)", width=1, dash='dot')), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Support_Dots'], mode='markers', marker=dict(color="#FFD700", size=6)), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_Dots'], mode='markers', marker=dict(color="#FF4B4B", size=6)), secondary_y=True)
        
        fig.update_yaxes(secondary_y=False, autorange=True, fixedrange=True, showgrid=False)
        fig.update_yaxes(secondary_y=True, range=[-5, 105], fixedrange=True)
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=30), df.index[-1]], fixedrange=False)
        fig.update_layout(height=480, template="plotly_dark", dragmode="pan", uirevision='constant', margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 回測
        st.subheader("📊 2025年 100萬回測數據")
        bt_df = df[df.index >= "2025-01-01"].copy()
        if not bt_df.empty:
            curr_p = bt_df['Close'].iloc[-1]
            y_days = bt_df[bt_df['Final_Score'] <= bt_df['Lower_Bound']]
            num_y = len(y_days)
            sys_val = (1000000 / num_y / y_days['Close']).sum() * curr_p if num_y > 0 else 1000000
            m_buys = bt_df.resample('MS').first()
            dca_val = (1000000 / len(m_buys) / m_buys['Close']).sum() * curr_p if len(m_buys) > 0 else 1000000
            
            res = pd.DataFrame({
                "策略項目": ["五維系統 (黃點布局)", "定期定額 (每月1號)"],
                "期末總市值": [f"${sys_val:,.0f}", f"${dca_val:,.0f}"],
                "累計報酬率": [f"{((sys_val-1000000)/1000000*100):.2f}%", f"{((dca_val-1000000)/1000000*100):.2f}%"]
            })
            st.table(res)
