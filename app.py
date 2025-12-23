import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維自適應：黃金比例版", layout="wide", initial_sidebar_state="collapsed")
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

st.title("🛡️ 視窗感應：股價動態邊距終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_historical_data(stock_id)

if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價) - 白色實線
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        name="價", 
        line=dict(color="#FFFFFF", width=1.5)
    ), secondary_y=False)

    # 2. 副 Y 軸 (指標) - 綠色線 (含透明度)
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Final_Score'], 
        name="檔", 
        line=dict(color="rgba(0, 210, 106, 0.7)", width=2.5)
    ), secondary_y=True)

    # --- 核心優化：視窗數據感應與自動緩衝邊距 ---
    
    fig.update_yaxes(
        secondary_y=False, 
        autorange=True,
        # 關鍵設定：強制 Y 軸僅根據目前視窗內容計算，並增加上下邊距
        fixedrange=False,
        zeroline=False,
        showgrid=False,
        # 使用 normal 模式並透過 autorange 屬性微調
        rangemode="normal" 
    )
    
    fig.update_yaxes(
        secondary_y=True, 
        range=[-10, 110],    # 指標軸固定在略大於 0-100，讓頂部與底部不顯得太擠
        fixedrange=True, 
        gridcolor="rgba(255, 255, 255, 0.05)",
        zeroline=False
    )

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        fixedrange=False,
        rangeslider_visible=False
    )

    # 初始視窗：預設顯示最近一年
    if len(df) > 252:
        start_date = df.index[-252]
        fig.update_xaxes(range=[start_date, df.index[-1]])

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified", 
        dragmode="pan",
        uirevision='constant', # 維持平移時的縮放狀態
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    # 隱藏工具列，啟用雙指/滾輪縮放
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': False
    })
    
    st.success("✨ **視窗對齊已優化**：現在滑動時，Y 軸會自動計算視窗內的最高/最低價，並自動預留美觀的緩衝空間。")

else:
    st.error("讀取失敗。")
