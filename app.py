import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維共振終端 (固定1年版)", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_ultimate_data(symbol):
    # 下載至少一年半的數據以計算一年份的百分位
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 維度計算
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'])
    df['macd_h'] = macd['MACDh_12_26_9']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['vol_r'] = df['Volume'].rolling(252).rank(pct=True) * 100

    # 綜合檔位融合
    df['Final_Score'] = (df['rsi_r'] * 0.3 + df['bias_r'] * 0.3 + df['macd_r'] * 0.4).rolling(5).mean()
    
    # 買賣點訊號
    df['Buy_Point'] = (df['Final_Score'] < 25) & (df['macd_h'] > df['macd_h'].shift(1))
    df['Sell_Point'] = (df['Final_Score'] > 75) & (df['macd_h'] < df['macd_h'].shift(1))
    
    return df

st.title("🛡️ 五維一體：固定一年期決策終端")

top_stocks = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "0050.TW": "元大台灣50", "006208.TW": "富邦台50"}
stock_id = st.sidebar.selectbox("標的選擇", options=list(top_stocks.keys()), format_func=lambda x: top_stocks[x])

df = get_ultimate_data(stock_id)

if not df.empty:
    # --- 關鍵：選取最近 252 筆交易日 (約 1 年) ---
    plot_df = df.tail(252)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 股價線 (主軸)
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價 (左軸)", 
                   line=dict(color="rgba(180, 180, 180, 0.4)", width=1.5)),
        secondary_y=False,
    )

    # 2. 綜合檔位線 (副軸 0-100)
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="檔位線 (右軸)", 
                   line=dict(color="#00d26a", width=3)),
        secondary_y=True,
    )

    # 3. 黃金星訊號
    buys = plot_df[plot_df['Buy_Point']]
    fig.add_trace(
        go.Scatter(x=buys.index, y=buys['Final_Score'], mode='markers', 
                   marker=dict(symbol='star', size=14, color='gold', line=dict(width=1, color='white')),
                   name='低買'),
        secondary_y=True,
    )

    # 4. 高賣訊號
    sells = plot_df[plot_df['Sell_Point']]
    fig.add_trace(
        go.Scatter(x=sells.index, y=sells['Final_Score'], mode='markers', 
                   marker=dict(symbol='x', size=12, color='#ff4b4b'),
                   name='高賣'),
        secondary_y=True,
    )

    # 固定 Y 軸設定
    fig.update_yaxes(title_text="股價 (NTD)", secondary_y=False, fixedrange=True)
    fig.update_yaxes(title_text="綜合檔位 (0-100)", secondary_y=True, range=[0, 100], fixedrange=True)
    fig.update_xaxes(fixedrange=True) # 禁止 X 軸縮放

    # 警戒線
    fig.add_hline(y=75, line_dash="dash", line_color="#ff4b4b", secondary_y=True)
    fig.add_hline(y=25, line_dash="dash", line_color="#00d26a", secondary_y=True)

    # --- 關鍵：關閉所有交互縮放功能 ---
    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified",
        dragmode=False, # 禁止拖拽選取
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 顯示圖表並隱藏工具欄
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # 狀態面板
    curr = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("當前檔位", f"{curr['Final_Score']:.1f}")
    c2.metric("MACD 趨勢", "向上" if curr['macd_h'] > df['macd_h'].iloc[-2] else "向下")
    c3.metric("資料範圍", "過去 252 交易日")

else:
    st.error("讀取失敗。")
