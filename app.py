import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維百萬資金回測", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_advanced_dynamic_data(symbol):
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
    
    return df

st.title("💰 百萬實測：五維策略 vs 定期定額")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_advanced_dynamic_data(stock_id)

if not df.empty:
    # --- 繪圖 (維持月視角與藍色線) ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
    
    # 標記撐壓
    df['Support_Dots'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
    fig.add_trace(go.Scatter(x=df.index, y=df['Support_Dots'], mode='markers', marker=dict(color="#FFD700", size=6)), secondary_y=True)
    
    fig.update_yaxes(secondary_y=False, autorange=True, fixedrange=True, showgrid=False)
    fig.update_yaxes(secondary_y=True, range=[-5, 105], fixedrange=True)
    
    last_date = df.index[-1]
    fig.update_xaxes(range=[last_date - pd.Timedelta(days=30), last_date], fixedrange=False)
    fig.update_layout(height=450, template="plotly_dark", dragmode="pan", uirevision='constant', margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- 100 萬回測邏輯 ---
    st.subheader("📊 100萬資金模擬 (2025/1/1 起算)")
    
    total_capital = 1000000
    backtest_df = df[df.index >= "2025-01-01"].copy()
    
    if not backtest_df.empty:
        current_price = backtest_df['Close'].iloc[-1]
        
        # 1. 系統策略：總資金平分給所有黃點
        yellow_days = backtest_df[backtest_df['Final_Score'] <= backtest_df['Lower_Bound']]
        num_yellow_days = len(yellow_days)
        
        if num_yellow_days > 0:
            per_point_invest = total_capital / num_yellow_days
            system_shares = (per_point_invest / yellow_days['Close']).sum()
            system_final_value = system_shares * current_price
            system_roi = ((system_final_value - total_capital) / total_capital) * 100
        else:
            system_final_value = total_capital
            system_roi = 0

        # 2. 定期定額：每月1號平均投
        monthly_buys = backtest_df.resample('MS').first()
        num_months = len(monthly_buys)
        per_month_invest = total_capital / num_months
        dca_shares = (per_month_invest / monthly_buys['Close']).sum()
        dca_final_value = dca_shares * current_price
        dca_roi = ((dca_final_value - total_capital) / total_capital) * 100

        # 顯示結果表格
        res_table = pd.DataFrame({
            "項目": ["五維系統策略", "每月定期定額"],
            "總投入資金": [f"${total_capital:,.0f}", f"${total_capital:,.0f}"],
            "買入次數/月數": [f"{num_yellow_days} 天", f"{num_months} 個月"],
            "期末總市值": [f"${system_final_value:,.0f}", f"${dca_final_value:,.0f}"],
            "累計報酬率": [f"{system_roi:.2f}%", f"{dca_roi:.2f}%"]
        })
        
        st.table(res_table)
        
        # 獲利差距提醒
        diff = system_final_value - dca_final_value
        if diff > 0:
            st.success(f"✅ 系統策略表現較優，多賺了 ${diff:,.0f}")
        else:
            st.warning(f"💡 定期定額表現較優，多賺了 ${abs(diff):,.0f}")

    else:
        st.warning("目前尚無 2025 年之完整數據。")

else:
    st.error("讀取失敗。")
