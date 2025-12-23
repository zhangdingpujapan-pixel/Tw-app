import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維自適應：動態邊界終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_advanced_dynamic_data(symbol):
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
        # 波動率檢查：單日波動超過平均 2.5 倍視為異常
        vol_ratio = abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0
        
        # 基礎加權 (35/35/30)
        if r['adx'] > 25:
            base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1)
        else:
            base = (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        
        # 如果波動太劇烈，分數強行向中性(50)靠攏，避免接刀
        return (base + 50) / 2 if vol_ratio > 2.5 else base

    df['Final_Score_Raw'] = df.apply(adaptive_logic, axis=1)
    df['Final_Score'] = df['Final_Score_Raw'].rolling(10).mean()

    # --- 動態邊界計算 (取過去一年的 15% 與 85% 分位) ---
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    
    return df

st.title("🛡️ 五維共振：動態邊界對齊終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_advanced_dynamic_data(stock_id)

if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name="價", 
        line=dict(color="#FFFFFF", width=1.2)
    ), secondary_y=False)

    # 2. 副 Y 軸 (綜合檔位線)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Final_Score'], name="檔", 
        line=dict(color="#00d26a", width=2.5)
    ), secondary_y=True)

    # 3. 副 Y 軸 (動態上邊界 - 壓力)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Upper_Bound'], name="壓", 
        line=dict(color="rgba(255, 75, 75, 0.4)", width=1, dash='dot')
    ), secondary_y=True)

    # 4. 副 Y 軸 (動態下邊界 - 支撐)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Lower_Bound'], name="撐", 
        line=dict(color="rgba(255, 215, 0, 0.4)", width=1, dash='dot')
    ), secondary_y=True)

    # --- 視覺優化：視窗自動貼合與縮放 ---
    fig.update_yaxes(
        secondary_y=False, autorange=True, fixedrange=False,
        showgrid=False, zeroline=False, rangemode="normal"
    )
    
    fig.update_yaxes(
        secondary_y=True, range=[-5, 105], fixedrange=True,
        gridcolor="rgba(255, 255, 255, 0.05)", zeroline=False
    )

    fig.update_xaxes(
        tickformat="%Y-%m-%d", fixedrange=False, rangeslider_visible=False
    )

    # 預設看最近一年
    if len(df) > 252:
        fig.update_xaxes(range=[df.index[-252], df.index[-1]])

    fig.update_layout(
        height=650, template="plotly_dark", hovermode="x unified",
        dragmode="pan", uirevision='constant',
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
    
    # 底部狀態提示
    curr = df.iloc[-1]
    status = "⚠️ 接近超賣區" if curr['Final_Score'] <= curr['Lower_Bound'] else \
             "🔥 接近超買區" if curr['Final_Score'] >= curr['Upper_Bound'] else "⚖️ 區間震盪"
    
    c1, c2, c3 = st.columns(3)
    c1.metric("當前分數", f"{curr['Final_Score']:.1f}")
    c2.metric("當前狀態", status)
    c3.metric("建議邊界", f"{curr['Lower_Bound']:.1f} - {curr['Upper_Bound']:.1f}")

else:
    st.error("數據加載失敗。")
