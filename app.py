import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略回測終端", layout="wide", initial_sidebar_state="collapsed")
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

    def adaptive_logic(r):
        if pd.isna(r['adx']) or pd.isna(r['atr']): return 50
        vol_ratio = abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0
        base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1) if r['adx'] > 25 else (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        return (base + 50) / 2 if vol_ratio > 2.5 else base

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    
    # 標記狀態
    df['Support_Dots'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
    df['Resistance_Dots'] = np.where(df['Final_Score'] >= df['Upper_Bound'], df['Final_Score'], np.nan)
    
    return df

st.title("🛡️ 五維共振：2025 績效回測終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_advanced_dynamic_data(stock_id)

if not df.empty:
    # --- 繪圖部分 ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Support_Dots'], name="超跌區", mode='markers', marker=dict(color="#FFD700", size=6)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_Dots'], name="過熱區", mode='markers', marker=dict(color="#FF4B4B", size=6)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], name="壓", line=dict(color="rgba(255, 75, 75, 0.3)", width=1, dash='dot')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], name="撐", line=dict(color="rgba(255, 215, 0, 0.3)", width=1, dash='dot')), secondary_y=True)

    fig.update_yaxes(secondary_y=False, autorange=True, fixedrange=True, showgrid=False, zeroline=False)
    fig.update_yaxes(secondary_y=True, range=[-5, 105], fixedrange=True, gridcolor="rgba(255, 255, 255, 0.05)")
    
    # 預設 1 個月視角
    last_date = df.index[-1]
    fig.update_xaxes(range=[last_date - pd.Timedelta(days=30), last_date], fixedrange=False)

    fig.update_layout(height=500, template="plotly_dark", dragmode="pan", uirevision='constant', margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

    # --- 2025 回測邏輯 ---
    st.subheader("📊 2025年度策略回測 (黃點布局 vs 定期定額)")
    
    backtest_df = df[df.index >= "2025-01-01"].copy()
    
    if not backtest_df.empty:
        # 1. 系統策略計算
        unit_investment = 10000
        backtest_df['Buy_System'] = np.where(backtest_df['Final_Score'] <= backtest_df['Lower_Bound'], 1, 0)
        total_system_invested = backtest_df['Buy_System'].sum() * unit_investment
        total_system_shares = (backtest_df['Buy_System'] * unit_investment / backtest_df['Close']).sum()
        current_price = backtest_df['Close'].iloc[-1]
        system_value = total_system_shares * current_price
        system_roi = ((system_value - total_system_invested) / total_system_invested * 100) if total_system_invested > 0 else 0

        # 2. 定期定額計算 (每月1號)
        backtest_df['Day'] = backtest_df.index.day
        # 抓取每個月的第一個交易日
        monthly_buys = backtest_df.resample('MS').first() 
        # 為了公平，讓定期定額的總投入本金與系統策略接近
        total_months = len(monthly_buys)
        monthly_investment = total_system_invested / total_months if total_months > 0 else 0
        total_dca_shares = (monthly_investment / monthly_buys['Close']).sum()
        dca_value = total_dca_shares * current_price
        dca_roi = ((dca_value - total_system_invested) / total_system_invested * 100) if total_system_invested > 0 else 0

        # 表格顯示
        res_data = {
            "策略項目": ["五維系統 (黃點布局)", "定期定額 (每月1號)"],
            "總投入本金": [f"${total_system_invested:,.0f}", f"${total_system_invested:,.0f}"],
            "當前總市值": [f"${system_value:,.0f}", f"${dca_value:,.0f}"],
            "累計報酬率": [f"{system_roi:.2f}%", f"{dca_roi:.2f}%"]
        }
        st.table(pd.DataFrame(res_data))
        
        st.caption(f"註：回測從 2025/01/01 至 {last_date.strftime('%Y/%m/%d')}。系統策略於黃點出現當日收盤買入 $10,000。")
    else:
        st.warning("尚無 2025 年之數據。")

else:
    st.error("讀取失敗。")
