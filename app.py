import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面優化
st.set_page_config(page_title="五維共振終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_ultimate_data(symbol):
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 指標計算
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'])
    df['macd_h'] = macd['MACDh_12_26_9']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['vol_r'] = df['Volume'].rolling(252).rank(pct=True) * 100

    # 綜合檔位融合邏輯
    df['Final_Score'] = (df['rsi_r'] * 0.3 + df['bias_r'] * 0.3 + df['macd_r'] * 0.4).rolling(5).mean()
    
    # 買賣點訊號
    df['Buy_Point'] = (df['Final_Score'] < 25) & (df['macd_h'] > df['macd_h'].shift(1))
    df['Sell_Point'] = (df['Final_Score'] > 75) & (df['macd_h'] < df['macd_h'].shift(1))
    
    return df

st.title("🛡️ 五維一體：低買高賣決策系統")

top_stocks = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "0050.TW": "元大台灣50", "006208.TW": "富邦台50"}
stock_id = st.sidebar.selectbox("標的", options=list(top_stocks.keys()), format_func=lambda x: top_stocks[x])

df = get_ultimate_data(stock_id)

if not df.empty:
    plot_df = df.tail(150)
    
    # --- 建立雙 Y 軸圖表 ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (左側)：股價
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價 (主軸)", 
                   line=dict(color="rgba(200, 200, 200, 0.4)", width=1.5)),
        secondary_y=False,
    )

    # 2. 副 Y 軸 (右側)：綜合檔位線 (0-100)
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="綜合檔位 (副軸)", 
                   line=dict(color="#00d26a", width=3)),
        secondary_y=True,
    )

    # 3. 標記低買點 (黃金星) - 必須掛在副軸 (0-100)
    buys = plot_df[plot_df['Buy_Point']]
    fig.add_trace(
        go.Scatter(x=buys.index, y=buys['Final_Score'], mode='markers', 
                   marker=dict(symbol='star', size=15, color='gold', line=dict(width=1, color='white')),
                   name='低買訊號'),
        secondary_y=True,
    )

    # 4. 標記高賣點 (紅叉) - 必須掛在副軸 (0-100)
    sells = plot_df[plot_df['Sell_Point']]
    fig.add_trace(
        go.Scatter(x=sells.index, y=sells['Final_Score'], mode='markers', 
                   marker=dict(symbol='x', size=12, color='#ff4b4b'),
                   name='高賣訊號'),
        secondary_y=True,
    )

    # 設定軸標籤與範圍
    fig.update_yaxes(title_text="股價 (NTD)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="綜合檔位 (0-100)", secondary_y=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)")
    
    # 加入 25/75 警戒線 (掛在副軸)
    fig.add_hline(y=75, line_dash="dash", line_color="#ff4b4b", secondary_y=True)
    fig.add_hline(y=25, line_dash="dash", line_color="#00d26a", secondary_y=True)

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 底部狀態簡報
    curr = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("當前檔位", f"{curr['Final_Score']:.1f}")
    c2.metric("MACD動能", "🟢 轉強" if curr['macd_h'] > df['macd_h'].iloc[-2] else "🔴 轉弱")
    c3.metric("趨勢強度", "強" if curr['adx'] > 25 else "平穩")

else:
    st.error("讀取失敗，請確認代碼。")
