import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維百萬資金終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

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
    
    # 標記狀態點
    df['Support_Dots'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
    df['Resistance_Dots'] = np.where(df['Final_Score'] >= df['Upper_Bound'], df['Final_Score'], np.nan)
    
    return df

st.title("💰 百萬實測：動態邊界全功能終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_full_data(stock_id)

if not df.empty:
    # --- 繪圖區 (修復動態邊界顯示) ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 主Y軸：股價
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)
    
    # 副Y軸：藍色檔位線
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
    
    # 副Y軸：動態邊界 (撐壓虛線)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], name="壓", line=dict(color="rgba(255, 75, 75, 0.4)", width=1, dash='dot')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], name="撐", line=dict(color="rgba(255, 215, 0, 0.4)", width=1, dash='dot')), secondary_y=True)
    
    # 副Y軸：黃/紅圓點
    fig.add_trace(go.Scatter(x=df.index, y=df['Support_Dots'], name="超跌區", mode='markers', marker=dict(color="#FFD700", size=6)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_Dots'], name="過熱區", mode='markers', marker=dict(color="#FF4B4B", size=6)), secondary_y=True)

    # 軸設定
    fig.update_yaxes(secondary_y=False, autorange=True, fixedrange=True, showgrid=False, zeroline=False, rangemode="normal")
    fig.update_yaxes(secondary_y=True, range=[-5, 105], fixedrange=True, gridcolor="rgba(255, 255, 255, 0.05)", zeroline=False)
    
    # 預設視角：1 個月
    last_date = df.index[-1]
    fig.update_xaxes(range=[last_date - pd.Timedelta(days=30), last_date], fixedrange=False, tickformat="%Y-%m-%d")

    fig.update_layout(height=500, template="plotly_dark", dragmode="pan", uirevision='constant', margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

    # --- 100 萬回測邏輯 ---
    st.subheader("📊 100萬資金回測對比 (2025/1/1 起算)")
    
    total_capital = 1000000
    backtest_df = df[df.index >= "2025-01-01"].copy()
    
    if not backtest_df.empty:
        current_price = backtest_df['Close'].iloc[-1]
        
        # 1. 系統策略 (黃點買入)
        yellow_days = backtest_df[backtest_df['Final_Score'] <= backtest_df['Lower_Bound']]
        num_yellow_days = len(yellow_days)
        if num_yellow_days > 0:
            per_point = total_capital / num_yellow_days
            system_shares = (per_point / yellow_days['Close']).sum()
            system_val = system_shares * current_price
            system_roi = ((system_val - total_capital) / total_capital) * 100
        else:
            system_val, system_roi = total_capital, 0

        # 2. 定期定額 (每月1號)
        monthly_buys = backtest_df.resample('MS').first()
        num_months = len(monthly_buys)
        if num_months > 0:
            per_month = total_capital / num_months
            dca_shares = (per_month / monthly_buys['Close']).sum()
            dca_val = dca_shares * current_price
            dca_roi = ((dca_val - total_capital) / total_capital) * 100
        else:
            dca_val, dca_roi = total_capital, 0

        res_table = pd.DataFrame({
            "項目": ["五維系統策略 (🟡)", "每月定期定額 (📅)"],
            "買入次數": [f"{num_yellow_days} 天", f"{num_months} 個月"],
            "期末總市值": [f"${system_val:,.0f}", f"${dca_val:,.0f}"],
            "累計報酬率": [f"{system_roi:.2f}%", f"{dca_roi:.2f}%"]
        })
        st.table(res_table)
        
        diff = system_val - dca_val
        if diff > 0:
            st.success(f"📈 系統策略目前領先定期定額 ${diff:,.0f}")
        else:
            st.warning(f"💡 定期定額目前領先系統策略 ${abs(diff):,.0f}")

else:
    st.error("讀取失敗。")
