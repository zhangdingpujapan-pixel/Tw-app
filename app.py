import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面終端感優化
st.set_page_config(page_title="頂尖交易者：五維共振終端", layout="wide", initial_sidebar_state="collapsed")

# 專業深色主題 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心數據引擎 (融入 MACD 百分位) ---
@st.cache_data(ttl=3600)
def get_ultimate_data(symbol):
    # 下載兩年數據以確保百分位計算穩定
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 【維度一：空間】RSI 與 乖離率
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['bias'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    
    # 【維度二：動能】MACD 柱狀體歸一化
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd_h'] = macd['MACDh_12_26_9']
    
    # 【維度三：趨勢環境】ADX
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    
    # 【維度四：燃料】成交量
    df['vol_r'] = df['Volume'].rolling(252, min_periods=1).rank(pct=True) * 100

    # --- 關鍵：將所有指標轉換為 0-100 的歷史排名 ---
    for col in ['rsi', 'bias', 'macd_h']:
        df[f'{col}_r'] = df[col].rolling(252, min_periods=1).rank(pct=True) * 100

    # --- 核心公式：MACD 直接融合 ---
    # 邏輯：綜合檔位 = (RSI排名*0.3) + (乖離率排名*0.3) + (MACD動能排名*0.4)
    df['Integrated_Score'] = (
        df['rsi_r'] * 0.3 + 
        df['bias_r'] * 0.3 + 
        df['macd_h_r'] * 0.4
    )
    
    # 【環境修正】根據 ADX 調整：趨勢強時加重乖離率佔比，盤整時維持原樣
    df['Final_Score'] = np.where(df['adx'] > 25,
                                 (df['Integrated_Score'] * 0.7 + df['bias_r'] * 0.3),
                                 df['Integrated_Score'])
    
    # 五日平滑，去除訊號雜訊
    df['Final_Score'] = df['Final_Score'].rolling(5, min_periods=1).mean()
    
    # 【共振買點訊號】綜合檔位低於 25 且 MACD 動能開始翻揚(柱狀體收縮) 且 成交量非縮量
    df['Buy_Signal'] = (df['Final_Score'] < 25) & (df['macd_h'] > df['macd_h'].shift(1)) & (df['vol_r'] > 30)
    
    return df

# --- 3. 介面呈現 ---
st.title("🛡️ 五維一體：台股共振分析系統")

# 市值前十大與熱門 ETF 清單
top_stocks = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", 
    "0050.TW": "元大台灣50", "006208.TW": "富邦台50", "2881.TW": "富邦金"
}

with st.sidebar:
    st.header("⚙️ 設定")
    choice = st.selectbox("選擇監控標的", options=list(top_stocks.keys()), format_func=lambda x: top_stocks[x])
    custom_id = st.text_input("或手動輸入代碼", value="")
    stock_id = custom_id if custom_id else choice
    st.divider()
    st.caption("五維一體核心：RSI、BIAS、MACD、ADX、Volume")

df = get_pro_data_optimized = get_ultimate_data(stock_id)

if not df.empty:
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 頂部儀表板
    col1, col2, col3, col4 = st.columns(4)
    
    score_delta = curr['Final_Score'] - prev['Final_Score']
    col1.metric("綜合檔位 (含MACD)", f"{curr['Final_Score']:.1f}", f"{score_delta:.1f}")
    
    # 動能文字
    m_color = "🟢 增強" if curr['macd_h'] > prev['macd_h'] else "🔴 衰退"
    col2.metric("MACD 動能狀態", m_color)
    
    # 環境判定
    env_status = "📉 趨勢盤" if curr['adx'] > 25 else "↔️ 盤整盤"
    col3.metric("目前市場性質", env_status)
    
    # 訊號狀態
    sig_status = "🔥 共振買點出現" if curr['Buy_Signal'] else "🛡️ 觀察中"
    col4.metric("交易共振訊號", sig_status)

    # --- 4. 專業圖表視覺化 ---
    # 繪製主圖與子圖 (MACD 輔助對照)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.08, row_heights=[0.7, 0.3])

    # 顯示最近半年的數據，讓手機看圖更清楚
    plot_df = df.tail(120)

    # 軌跡：股價線 (灰色背景)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價", 
                             line=dict(color="rgba(150, 150, 150, 0.5)", width=1.5)), row=1, col=1)
    
    # 軌跡：融合後的綜合檔位線 (核心主角)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="融合檔位線", 
                             line=dict(color="#00d26a", width=3)), row=1, col=1)

    # 標記：共振買點 (黃金星)
    buys = plot_df[plot_df['Buy_Signal']]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Final_Score'], mode='markers',
                             marker=dict(symbol='star', size=14, color='#FFD700', line=dict(width=1, color='white')),
                             name='共振買點'), row=1, col=1)

    # 子圖：MACD 柱狀體 (動能視覺化)
    bar_colors = ['#00ff00' if v > 0 else '#ff4b4b' for v in plot_df['macd_h']]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['macd_h'], name="MACD柱狀體", 
                         marker_color=bar_colors, opacity=0.7), row=2, col=1)

    # 介面細節調整
    fig.update_layout(height=650, template="plotly_dark", hovermode="x unified",
                      margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    
    # 警戒線
    fig.add_hline(y=75, line_dash="dash", line_color="#ff4b4b", row=1, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="#00d26a", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**學者解析**：目前的「融合檔位線」已直接嵌入了 MACD 動能因子。當線條跌破 25 且 MACD 柱狀體停止擴張時，黃金星訊號會自動觸發。")
else:
    st.error("代碼讀取失敗，請確認代號（如 2330.TW）。")
