import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維一體：等比鎖定終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_advanced_data(symbol):
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 指標計算 (進階優化配方) ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['sma_200'] = df['Close'].rolling(200).mean()

    # 自適應權重邏輯
    def adaptive_logic(r):
        if r['adx'] > 25: return (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1)
        else: return (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    
    # 訊號過濾
    buy_cond = (df['Final_Score'] < 25) & (df['macd_h'] > df['macd_h'].shift(1)) & (df['Close'] > df['sma_200'])
    sell_cond = (df['Final_Score'] > 75) & (df['macd_h'] < df['macd_h'].shift(1))
    df['Buy_Signal'] = buy_cond & (buy_cond.shift(1) == False)
    df['Sell_Signal'] = sell_cond & (sell_cond.shift(1) == False)
    
    return df

st.title("🛡️ 五維自適應：等比鎖定終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_advanced_data(stock_id)

if not df.empty:
    plot_df = df.tail(252) # 鎖定顯示一年份
    
    # 建立雙 Y 軸
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價)：設定自動縮放並禁止手動調整
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="價", 
                             line=dict(color="rgba(200,200,200,0.3)", width=1.5)), secondary_y=False)

    # 2. 副 Y 軸 (指標)：固定 0-100 並禁止手動調整
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="檔", 
                             line=dict(color="#00d26a", width=3)), secondary_y=True)

    # --- 關鍵修正：自動等比縮放與禁止手動位移 ---
    # 設定左軸 (股價)
    fig.update_yaxes(
        secondary_y=False, 
        autorange=True,      # 根據畫面數據自動調整範圍
        fixedrange=True,     # 禁止手動上下拉動/縮放
        showgrid=False
    )
    
    # 設定右軸 (指標)
    fig.update_yaxes(
        secondary_y=True, 
        range=[0, 100],      # 固定指標高度
        fixedrange=True,     # 禁止手動上下拉動/縮放
        gridcolor="rgba(255,255,255,0.05)"
    )

    # 設定 X 軸 (日期)
    fig.update_xaxes(
        tickformat="%Y-%m-%d", 
        dtick="M2", 
        fixedrange=True      # 禁止左右縮放，維持一年視角
    )

    fig.update_layout(
        height=500, 
        template="plotly_dark", 
        hovermode="x unified", 
        dragmode=False,      # 徹底關閉拖拽功能
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # --- 數據表格 ---
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
        st.info("一年內無高勝率訊號。")

else:
    st.error("代碼錯誤。")
