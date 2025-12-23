import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維自適應：狀態標記版", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_advanced_dynamic_data(symbol):
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 指標計算 ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # --- 自適應權重邏輯 ---
    def adaptive_logic(r):
        if pd.isna(r['adx']) or pd.isna(r['atr']): return 50
        vol_ratio = abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0
        base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1) if r['adx'] > 25 else (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        return (base + 50) / 2 if vol_ratio > 2.5 else base

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    
    # --- 標記狀態點 ---
    # 低於支撐線的點
    df['Support_Dots'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
    # 高於壓力線的點
    df['Resistance_Dots'] = np.where(df['Final_Score'] >= df['Upper_Bound'], df['Final_Score'], np.nan)
    
    return df

st.title("🛡️ 五維共振：區間狀態監測終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_advanced_dynamic_data(stock_id)

if not df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 股價線
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)

    # 2. 綜合檔位線 (藍色)
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)

    # 3. 狀態圓點標記 (重點優化部分)
    # 超跌圓點 (黃色)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Support_Dots'], name="超跌區",
        mode='markers', marker=dict(color="#FFD700", size=6, opacity=0.8)
    ), secondary_y=True)
    
    # 過熱圓點 (紅色)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Resistance_Dots'], name="過熱區",
        mode='markers', marker=dict(color="#FF4B4B", size=6, opacity=0.8)
    ), secondary_y=True)

    # 4. 動態邊界虛線
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], name="壓", line=dict(color="rgba(255, 75, 75, 0.3)", width=1, dash='dot')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], name="撐", line=dict(color="rgba(255, 215, 0, 0.3)", width=1, dash='dot')), secondary_y=True)

    # --- 視覺軸設定 ---
    fig.update_yaxes(secondary_y=False, autorange=True, fixedrange=True, showgrid=False, zeroline=False, rangemode="normal")
    fig.update_yaxes(secondary_y=True, range=[-5, 105], fixedrange=True, gridcolor="rgba(255, 255, 255, 0.05)", zeroline=False)
    fig.update_xaxes(tickformat="%Y-%m-%d", fixedrange=False, rangeslider_visible=False)

    # 初始範圍 1 個月
    if len(df) > 30:
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=30), df.index[-1]])

    fig.update_layout(
        height=600, template="plotly_dark", hovermode="x unified", dragmode="pan",
        uirevision='constant', margin=dict(l=10, r=10, t=10, b=10), showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
    
    st.info("💡 **圓點標記說明**：🟡 黃色點代表處於支撐線下方的「超跌區間」；🔴 紅色點代表處於壓力線上方的「過熱區間」。")

else:
    st.error("讀取失敗。")
