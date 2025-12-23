import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 介面與專業深色風格
st.set_page_config(page_title="五維一體：進階優化終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_advanced_data(symbol):
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 基礎指標計算 ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    
    # 優化版快速 MACD (6, 13, 5)
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['sma_200'] = df['Close'].rolling(200).mean() # 長線濾網

    # --- 進階邏輯：自適應綜權重 ---
    def adaptive_logic(r):
        if r['adx'] > 25: # 趨勢盤
            return (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1)
        else: # 盤整盤
            return (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)

    df['Composite_Raw'] = df.apply(adaptive_logic, axis=1)
    df['Final_Score'] = df['Composite_Raw'].rolling(10).mean() # 10日平滑
    
    # --- 買賣訊號過濾 ---
    # 買入：低檔 + 動能轉強 + 在年線上 (順勢)
    buy_cond = (df['Final_Score'] < 25) & (df['macd_h'] > df['macd_h'].shift(1)) & (df['Close'] > df['sma_200'])
    # 賣出：高檔 + 動能轉弱
    sell_cond = (df['Final_Score'] > 75) & (df['macd_h'] < df['macd_h'].shift(1))
    
    df['Buy_Signal'] = buy_cond & (buy_cond.shift(1) == False)
    df['Sell_Signal'] = sell_cond & (sell_cond.shift(1) == False)
    
    return df

st.title("🛡️ 專業級：五維自適應共振系統")

stock_id = st.sidebar.text_input("輸入台股代碼 (例: 2330.TW)", value="2330.TW")
df = get_advanced_data(stock_id)

if not df.empty:
    plot_df = df.tail(252) # 鎖定一年
    
    # 1. 圖表顯示 (主軸股價 / 副軸綜合線)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價", line=dict(color="rgba(200,200,200,0.3)")), secondary_y=False)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="綜合檔位線", line=dict(color="#00d26a", width=3)), secondary_y=True)
    
    fig.update_xaxes(tickformat="%Y-%m-%d", dtick="M2", fixedrange=True)
    fig.update_yaxes(secondary_y=True, range=[0, 100], fixedrange=True)
    fig.update_layout(height=450, template="plotly_dark", dragmode=False, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 2. 訊號數據表格 (高勝率過濾後)
    st.subheader("📋 進階策略交易明細")
    signals = plot_df[(plot_df['Buy_Signal']) | (plot_df['Sell_Signal'])].copy()
    
    if not signals.empty:
        table_list = []
        for idx, row in signals.iterrows():
            table_list.append({
                "日期": idx.strftime('%Y-%m-%d'),
                "操作": "🟢 低買" if row['Buy_Signal'] else "🔴 高賣",
                "執行價位": f"{row['Close']:.2f}",
                "綜合分": f"{row['Final_Score']:.1f}",
                "趨勢過濾": "✅ 順勢交易" if row['Close'] > row['sma_200'] else "⚠️ 逆勢風險"
            })
        st.table(pd.DataFrame(table_list))
    else:
        st.info("當前篩選條件下無高勝率訊號 (或標的處於長期空頭)。")

    # 底部狀態區
    curr = df.iloc[-1]
    col1, col2 = st.columns(2)
    col1.metric("當前綜合檔位", f"{curr['Final_Score']:.1f}")
    col2.metric("趨勢環境 (ADX)", f"{curr['adx']:.1f}")

else:
    st.error("請確認代碼輸入是否正確。")
