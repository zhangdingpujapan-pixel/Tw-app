import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維一體：純數字日期版", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_clean_data(symbol):
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 指標排名計算
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'])
    df['macd_h'] = macd['MACDh_12_26_9']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']

    # 10日平滑，讓線條變圓滑
    df['Final_Score'] = (df['rsi_r'] * 0.3 + df['bias_r'] * 0.3 + df['macd_r'] * 0.4).rolling(10).mean()
    
    # 判斷趨勢穩定度
    df['m_up'] = df['macd_h'] > df['macd_h'].shift(1)
    df['m_down'] = df['macd_h'] < df['macd_h'].shift(1)

    # 基礎買賣邏輯
    raw_buy = (df['Final_Score'] < 25) & (df['m_up'])
    raw_sell = (df['Final_Score'] > 75) & (df['m_down'])
    
    # 只保留「第一個」訊號點
    df['Buy_Point'] = (raw_buy) & (raw_buy.shift(1) == False)
    df['Sell_Point'] = (raw_sell) & (raw_sell.shift(1) == False)
    
    return df

st.title("🛡️ 五維一體：低買高賣終端")

top_stocks = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "0050.TW": "元大台灣50", "006208.TW": "富邦台50"}
stock_id = st.sidebar.selectbox("標的選擇", options=list(top_stocks.keys()), format_func=lambda x: top_stocks[x])

df = get_clean_data(stock_id)

if not df.empty:
    plot_df = df.tail(252) # 顯示一整年數據
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 股價線
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="價", 
                             line=dict(color="rgba(150, 150, 150, 0.3)", width=1.5)), secondary_y=False)

    # 2. 綜合檔位線
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="檔", 
                             line=dict(color="#00d26a", width=2.5)), secondary_y=True)

    # 3. 標記買點
    buys = plot_df[plot_df['Buy_Point']]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Final_Score'], mode='markers', 
                             marker=dict(symbol='star', size=14, color='gold', line=dict(width=1, color='white')),
                             name='買'), secondary_y=True)

    # 4. 標記賣點
    sells = plot_df[plot_df['Sell_Point']]
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Final_Score'], mode='markers', 
                             marker=dict(symbol='x', size=12, color='#ff4b4b'),
                             name='賣'), secondary_y=True)

    # --- 關鍵優化：強制日期顯示為數字格式 ---
    fig.update_xaxes(
        tickformat="%Y-%m-%d", # 格式化為 2024-12-24
        dtick="M2",            # 每 2 個月顯示一個刻度，避免太擠
        fixedrange=True,
        gridcolor="rgba(255,255,255,0.05)"
    )

    fig.update_yaxes(secondary_y=False, fixedrange=True, showgrid=False)
    fig.update_yaxes(secondary_y=True, range=[0, 100], fixedrange=True, gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified", 
        dragmode=False,
        hoverlabel=dict(bgcolor="#161b22", font_size=12),
        margin=dict(l=10, r=10, t=20, b=10), 
        showlegend=False
    )
    
    # 統一提示框的日期格式
    fig.update_traces(xhoverformat="%Y-%m-%d")

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # 底部數據看板
    curr = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("當前日期", curr.name.strftime('%Y-%m-%d'))
    c2.metric("當前檔位", f"{curr['Final_Score']:.1f}")
    c3.metric("趨勢環境", "強" if curr['adx'] > 25 else "平穩")

else:
    st.error("讀取失敗。")
