import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：核心資產終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# 定義股票清單
ASSET_LIST = {
    "市值前十大公司": {
        "2330.TW": "台積電",
        "2317.TW": "鴻海",
        "2454.TW": "聯發科",
        "2308.TW": "台達電",
        "2881.TW": "富邦金",
        "2882.TW": "國泰金",
        "2382.TW": "廣達",
        "2891.TW": "中信金",
        "3711.TW": "日月光投控",
        "2412.TW": "中華電"
    },
    "優秀市值型 ETF": {
        "0050.TW": "元大台灣50",
        "006208.TW": "富邦台50",
        "00922.TW": "國泰台灣領袖50"
    },
    "熱門高股息 ETF": {
        "0056.TW": "元大高股息",
        "00878.TW": "國泰永續高股息",
        "00919.TW": "群益台灣精選高息",
        "00929.TW": "復華台灣科技優息"
    }
}

@st.cache_data(ttl=3600)
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

# --- 側邊欄選單 ---
st.sidebar.header("📁 資產篩選器")
category = st.sidebar.selectbox("選擇資產類別", list(ASSET_LIST.keys()))
asset_options = ASSET_LIST[category]
selected_asset_name = st.sidebar.selectbox("選擇標的", list(asset_options.values()))
# 根據名稱反查代碼
stock_id = [k for k, v in asset_options.items() if v == selected_asset_name][0]

# 手動輸入功能保留
manual_id = st.sidebar.text_input("或手動輸入代碼 (EX: 2330.TW)", value="")
if manual_id: stock_id = manual_id

st.title(f"🛡️ {selected_asset_name} ({stock_id})")

df = get_full_data(stock_id)

if not df.empty:
    # --- 圖表區 ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], name="壓", line=dict(color="rgba(255, 75, 75, 0.4)", width=1, dash='dot')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], name="撐", line=dict(color="rgba(255, 215, 0, 0.4)", width=1, dash='dot')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Support_Dots'], mode='markers', marker=dict(color="#FFD700", size=6)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_Dots'], mode='markers', marker=dict(color="#FF4B4B", size=6)), secondary_y=True)

    fig.update_yaxes(secondary_y=False, autorange=True, fixedrange=True, showgrid=False, zeroline=False, rangemode="normal")
    fig.update_yaxes(secondary_y=True, range=[-5, 105], fixedrange=True, gridcolor="rgba(255, 255, 255, 0.05)", zeroline=False)
    
    last_date = df.index[-1]
    fig.update_xaxes(range=[last_date - pd.Timedelta(days=30), last_date], fixedrange=False, tickformat="%Y-%m-%d")
    fig.update_layout(height=480, template="plotly_dark", dragmode="pan", uirevision='constant', margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- 100 萬回測 ---
    st.subheader(f"📊 100萬回測：{selected_asset_name}")
    backtest_df = df[df.index >= "2025-01-01"].copy()
    
    if not backtest_df.empty:
        curr_p = backtest_df['Close'].iloc[-1]
        
        # 系統策略
        y_days = backtest_df[backtest_df['Final_Score'] <= backtest_df['Lower_Bound']]
        num_y = len(y_days)
        sys_val, sys_roi = 1000000, 0
        if num_y > 0:
            sys_shares = ((1000000 / num_y) / y_days['Close']).sum()
            sys_val = sys_shares * curr_p
            sys_roi = ((sys_val - 1000000) / 1000000) * 100

        # 定期定額
        m_buys = backtest_df.resample('MS').first()
        num_m = len(m_buys)
        dca_val, dca_roi = 1000000, 0
        if num_m > 0:
            dca_shares = ((1000000 / num_m) / m_buys['Close']).sum()
            dca_val = dca_shares * curr_p
            dca_roi = ((dca_val - 1000000) / 1000000) * 100

        res = pd.DataFrame({
            "策略項目": ["五維系統 (黃點布局)", "定期定額 (每月1號)"],
            "買入頻率": [f"{num_y} 次交易日", f"{num_m} 個月"],
            "期末總價值": [f"${sys_val:,.0f}", f"${dca_val:,.0f}"],
            "累計報酬率": [f"{sys_roi:.2f}%", f"{dca_roi:.2f}%"]
        })
        st.table(res)
    else:
        st.info("2025 年數據尚在累積中...")

else:
    st.error("代碼有誤或查無數據。")
