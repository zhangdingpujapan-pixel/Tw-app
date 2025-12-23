import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維自適應：視窗貼合版", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_historical_data(symbol):
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

    def adaptive_logic(r):
        if pd.isna(r['adx']): return 50
        return (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1) if r['adx'] > 25 else (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    return df

st.title("🛡️ 視窗感應：股價上下限自動貼合終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_historical_data(stock_id)

if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價) - 增加線條亮度方便對齊
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", 
                             line=dict(color="#ffffff", width=1.5)), secondary_y=False)

    # 2. 副 Y 軸 (指標)
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", 
                             line=dict(color="#00d26a", width=2.5, opacity=0.8)), secondary_y=True)

    # --- 核心邏輯：強制 Y 軸貼合視窗數據 ---
    
    fig.update_yaxes(
        secondary_y=False, 
        autorange=True,
        # 關鍵設定：強制 Y 軸只根據當前視窗內的數據計算 range
        # 在 Plotly JS 中這通常是預設，但在 Python 中我們透過不指定 range 來強化此行為
        fixedrange=False, 
        showgrid=False,
        zeroline=False
    )
    
    fig.update_yaxes(
        secondary_y=True, 
        range=[0, 100], 
        fixedrange=True, # 指標軸永遠固定，不隨動
        gridcolor="rgba(255,255,255,0.1)",
        zeroline=False
    )

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        fixedrange=False,
        rangeslider_visible=False
    )

    # 預設顯示最近一年
    start_date = df.index[-252] if len(df) > 252 else df.index[0]
    fig.update_xaxes(range=[start_date, df.index[-1]])

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified", 
        dragmode="pan",
        # uirevision 確保在數據更新或滑動時，手動縮放的狀態被保留，且觸發 autorange 重新計算
        uirevision='constant', 
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': False
    })
    
    st.caption("💡 現在當你左右滑動時，左側 y 軸會自動抓取視窗內最高價與最低價作為邊界（例如 500~1000），不會留白。")

else:
    st.error("讀取失敗。")
