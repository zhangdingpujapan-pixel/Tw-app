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
    # 下載完整歷史數據
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 指標計算
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    
    # 快速 MACD
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']

    def adaptive_logic(r):
        if pd.isna(r['adx']): return 50
        # 根據趨勢強度調整權重
        if r['adx'] > 25:
            return (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1)
        else:
            return (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    return df

st.title("🛡️ 視窗感應：股價上下限自動貼合終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_historical_data(stock_id)

if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價) - 白色實線
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        name="股價", 
        line=dict(color="#FFFFFF", width=1.5)
    ), secondary_y=False)

    # 2. 副 Y 軸 (指標) - 綠色線 (修正 opacity 錯誤，改用 rgba)
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Final_Score'], 
        name="綜合檔位", 
        line=dict(color="rgba(0, 210, 106, 0.8)", width=2.5)
    ), secondary_y=True)

    # --- 核心邏輯：強制 Y 軸貼合視窗數據 ---
    
    fig.update_yaxes(
        secondary_y=False, 
        autorange=True,      # 讓 Y 軸隨視窗數據自動計算範圍
        fixedrange=False,    # 允許 Y 軸變動
        showgrid=False,
        zeroline=False       # 關閉 0 基準線，防止強制拉低 Y 軸
    )
    
    fig.update_yaxes(
        secondary_y=True, 
        range=[0, 100],      # 指標軸始終固定在 0-100
        fixedrange=True, 
        gridcolor="rgba(255, 255, 255, 0.1)",
        zeroline=False
    )

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        fixedrange=False,
        rangeslider_visible=False
    )

    # 預設顯示最近一年視角
    if len(df) > 252:
        start_date = df.index[-252]
        fig.update_xaxes(range=[start_date, df.index[-1]])

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified", 
        dragmode="pan",      # 預設為平移模式
        uirevision='constant', # 核心設定：平移時保持狀態並重新觸發自動縮放
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': False
    })
    
    st.info("💡 **修正完成**：已移除錯誤的透明度參數。現在當你左右滑動時，左側股價軸會根據當前畫面自動對齊最高/最低價。")

else:
    st.error("讀取失敗，請確認代碼。")
