import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面與風格設定
st.set_page_config(page_title="五維低買高賣終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_optimized_data(symbol):
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 核心計算 (與之前五維一體邏輯相同)
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'])
    df['macd_h'] = macd['MACDh_12_26_9']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['vol_r'] = df['Volume'].rolling(252).rank(pct=True) * 100

    # 綜合檔位線 (直接融合 MACD)
    df['Final_Score'] = (df['rsi_r'] * 0.3 + df['bias_r'] * 0.3 + df['macd_r'] * 0.4).rolling(5).mean()
    
    # --- 優化：低買高賣邏輯 ---
    # 低買 (黃金星)：超跌區 + 動能止跌
    df['Buy_Point'] = (df['Final_Score'] < 25) & (df['macd_h'] > df['macd_h'].shift(1))
    
    # 高賣 (紅警告)：超漲區 + 動能轉弱
    df['Sell_Point'] = (df['Final_Score'] > 75) & (df['macd_h'] < df['macd_h'].shift(1))
    
    return df

st.title("🛡️ 五維一體：低買高賣決策系統")

# 選單與輸入
top_list = {"2330.TW": "台積電", "2317.TW": "鴻海", "0050.TW": "元大台灣50", "006208.TW": "富邦台50"}
stock_id = st.sidebar.selectbox("標的", options=list(top_list.keys()), format_func=lambda x: top_list[x])

df = get_optimized_data(stock_id)

if not df.empty:
    curr = df.iloc[-1]
    
    # 頂部狀態顯示
    c1, c2, c3 = st.columns(3)
    c1.metric("當前綜合檔位", f"{curr['Final_Score']:.1f}")
    
    # 策略建議文字
    advice = "🟢 建議低位佈局" if curr['Final_Score'] < 25 else "🔴 建議逢高減碼" if curr['Final_Score'] > 75 else "⚪ 區間震盪觀望"
    c2.subheader(f"戰略建議：{advice}")
    c3.metric("趨勢強度 (ADX)", f"{curr['adx']:.1f}")

    # 圖表視覺化
    fig = make_subplots(rows=1, cols=1)
    plot_df = df.tail(150)
    
    # 背景價格
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價", line=dict(color="rgba(100,100,100,0.3)")))
    
    # 核心檔位線
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="綜合檔位", line=dict(color="#00d26a", width=3)))
    
    # 標記低買點 (黃金星)
    buys = plot_df[plot_df['Buy_Point']]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Final_Score'], mode='markers', marker=dict(symbol='star', size=15, color='gold'), name='低買訊號'))
    
    # 標記高賣點 (紅色警告)
    sells = plot_df[plot_df['Sell_Point']]
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Final_Score'], mode='markers', marker=dict(symbol='x', size=12, color='#ff4b4b'), name='高賣訊號'))
    
    fig.update_layout(height=600, template="plotly_dark", hovermode="x unified")
    fig.add_hline(y=75, line_dash="dash", line_color="#ff4b4b")
    fig.add_hline(y=25, line_dash="dash", line_color="#00d26a")
    
    st.plotly_chart(fig, use_container_width=True)

    st.success("**操作手冊**：看黃金星買入，看紅叉叉賣出。中間區域不隨便操作，這就是最穩定的低買高賣。")
