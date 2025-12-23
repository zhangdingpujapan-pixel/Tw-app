import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維一體：數據分析終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_processed_data(symbol):
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 指標排名計算
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'])
    df['macd_h'] = macd['MACDh_12_26_9']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    
    # 10日平滑綜合檔位
    df['Final_Score'] = (df['rsi_r'] * 0.3 + df['bias_r'] * 0.3 + df['macd_r'] * 0.4).rolling(10).mean()
    
    # 判斷動能方向
    df['m_up'] = df['macd_h'] > df['macd_h'].shift(1)
    df['m_down'] = df['macd_h'] < df['macd_h'].shift(1)

    # 買賣邏輯 (首發訊號)
    raw_buy = (df['Final_Score'] < 25) & (df['m_up'])
    raw_sell = (df['Final_Score'] > 75) & (df['m_down'])
    df['Buy_Signal'] = (raw_buy) & (raw_buy.shift(1) == False)
    df['Sell_Signal'] = (raw_sell) & (raw_sell.shift(1) == False)
    
    return df

st.title("🛡️ 五維一體：專業數據分析終端")

top_stocks = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "0050.TW": "元大台灣50", "006208.TW": "富邦台50"}
stock_id = st.sidebar.selectbox("標的選擇", options=list(top_stocks.keys()), format_func=lambda x: top_stocks[x])

df = get_processed_data(stock_id)

if not df.empty:
    plot_df = df.tail(252) # 顯示最近一年
    
    # --- 1. 繪製純淨圖表 (移除星星叉叉) ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 股價線 (主軸)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價", 
                             line=dict(color="rgba(150, 150, 150, 0.4)", width=1.5)), secondary_y=False)

    # 綜合檔位線 (副軸)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="綜合檔位", 
                             line=dict(color="#00d26a", width=2.5)), secondary_y=True)

    # X 軸格式化
    fig.update_xaxes(tickformat="%Y-%m-%d", dtick="M2", fixedrange=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(secondary_y=False, fixedrange=True, showgrid=False)
    fig.update_yaxes(secondary_y=True, range=[0, 100], fixedrange=True, gridcolor="rgba(255,255,255,0.05)")

    fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", dragmode=False,
                      margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # --- 2. 建立訊號數據表格 ---
    st.subheader("📋 交易訊號明細清單")
    
    # 篩選出有買入或賣出訊號的日期
    signals = plot_df[(plot_df['Buy_Signal']) | (plot_df['Sell_Signal'])].copy()
    
    if not signals.empty:
        # 整理表格數據
        table_data = []
        for index, row in signals.iterrows():
            signal_type = "🟢 低買" if row['Buy_Signal'] else "🔴 高賣"
            table_data.append({
                "日期": index.strftime('%Y-%m-%d'),
                "訊號類型": signal_type,
                "當日價位": f"{row['Close']:.2f}",
                "綜合檔位數值": f"{row['Final_Score']:.1f}"
            })
        
        # 轉換為 DataFrame 並顯示
        st.table(pd.DataFrame(table_data))
    else:
        st.info("過去一年內尚無觸發買賣訊號。")

    # 底部快速資訊
    curr = df.iloc[-1]
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("最新日期", curr.name.strftime('%Y-%m-%d'))
    c2.metric("當前綜合檔位", f"{curr['Final_Score']:.1f}")

else:
    st.error("數據讀取失敗。")
