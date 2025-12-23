import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維自適應：精準月視角", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_advanced_dynamic_data(symbol):
    # 下載歷史數據
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 基礎指標計算 ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # --- 自適應權重邏輯 + 波動率過濾 ---
    def adaptive_logic(r):
        if pd.isna(r['adx']) or pd.isna(r['atr']): return 50
        vol_ratio = abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0
        if r['adx'] > 25:
            base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1)
        else:
            base = (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        return (base + 50) / 2 if vol_ratio > 2.5 else base

    df['Final_Score_Raw'] = df.apply(adaptive_logic, axis=1)
    df['Final_Score'] = df['Final_Score_Raw'].rolling(10).mean()

    # --- 動態邊界計算 ---
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    
    return df

st.title("🛡️ 五維共振：1個月精準視角")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_advanced_dynamic_data(stock_id)

if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name="價", 
        line=dict(color="#FFFFFF", width=1.5)
    ), secondary_y=False)

    # 2. 副 Y 軸 (綜合檔位線) - 已改成藍色
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Final_Score'], name="檔", 
        line=dict(color="#00BFFF", width=2.5) # 深天藍色
    ), secondary_y=True)

    # 3. 動態邊界線 (維持半透明感)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Upper_Bound'], name="壓", 
        line=dict(color="rgba(255, 75, 75, 0.4)", width=1, dash='dot')
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Lower_Bound'], name="撐", 
        line=dict(color="rgba(255, 215, 0, 0.4)", width=1, dash='dot')
    ), secondary_y=True)

    # --- 視覺優化設定 ---
    
    # 左 Y 軸：自動貼合數據，但鎖定手動上下拉動
    fig.update_yaxes(
        secondary_y=False, 
        autorange=True, 
        fixedrange=True, # 固定 Y 軸，禁止拉動
        showgrid=False, 
        zeroline=False, 
        rangemode="normal"
    )
    
    # 右 Y 軸：指標範圍
    fig.update_yaxes(
        secondary_y=True, 
        range=[-5, 105], 
        fixedrange=True, # 固定指標軸
        gridcolor="rgba(255, 255, 255, 0.05)", 
        zeroline=False
    )

    # X 軸：預設顯示「1 個月」，並允許左右滑動
    fig.update_xaxes(
        tickformat="%Y-%m-%d", 
        fixedrange=False,    # 允許左右滑動
        rangeslider_visible=False
    )

    # --- 關鍵修正：初始視角設為 1 個月 ---
    if len(df) > 30:
        last_date = df.index[-1]
        start_date = last_date - pd.Timedelta(days=30)
        fig.update_xaxes(range=[start_date, last_date])

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified",
        dragmode="pan",      # 預設平移模式
        uirevision='constant', 
        margin=dict(l=10, r=10, t=10, b=10), 
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,       # 支援雙指縮放 X 軸
        'displayModeBar': False
    })
    
    st.info("📅 **視角說明**：目前預設顯示近 1 個月數據。您可以透過**左右滑動**來查看歷史細節，股價軸會自動隨日期調整高度。")

else:
    st.error("數據讀取失敗。")
