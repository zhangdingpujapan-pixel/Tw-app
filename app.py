import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 讓網頁寬度適配手機
st.set_page_config(page_title="台股精準檔位 App", layout="wide")

st.title("🎯 精準綜合檔位回測系統")

# --- 側邊欄：互動參數 ---
with st.sidebar:
    st.header("參數設定")
    stock_id = st.text_input("股票代碼", value="2330.TW")
    lookback_date = st.date_input("開始日期", value=pd.to_datetime("2024-01-01"))
    st.divider()
    st.info("指標包含：RSI、WILLR、BIAS、ADX、成交量權重")

# --- 數據抓取 (使用緩存優化速度) ---
@st.cache_data
def get_data(symbol, start_date):
    df = yf.download(symbol, start=start_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = get_data(stock_id, lookback_date)

if not df.empty:
    # --- 計算邏輯 (維持你原本的精準算法) ---
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['willr'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    df['bias'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['vol_rank'] = df['Volume'].rolling(252, min_periods=1).rank(pct=True) * 100

    # 歷史百分位
    for col in ['rsi', 'willr', 'bias']:
        df[f'{col}_r'] = df[col].rolling(252, min_periods=1).rank(pct=True) * 100

    # 動態權重邏輯
    df['Composite'] = np.where(df['adx'] > 25,
                               (df['bias_r'] * 0.6 + df['rsi_r'] * 0.2 + df['willr_r'] * 0.2),
                               (df['bias_r'] * 0.2 + df['rsi_r'] * 0.4 + df['willr_r'] * 0.4))

    # 量價修正
    df['Final_Score'] = np.where((df['Composite'] > 70) & (df['vol_rank'] < 30), 
                                 df['Composite'] - 15, df['Composite'])
    
    # 平滑處理
    df['Final_Score'] = df['Final_Score'].rolling(5, min_periods=1).mean()

    # --- 繪圖區 ---
    # 預設顯示 2025 年後的數據
    plot_df = df.loc["2025-01-01":].copy()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 軌跡 1: 收盤價 (灰色背景)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="收盤價",
                             line=dict(color="rgba(200, 200, 200, 0.4)", width=1.5)), secondary_y=False)
    
    # 軌跡 2: 綜合檔位
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="精準檔位",
                             line=dict(color="#1f77b4", width=3)), secondary_y=True)

    # 軌跡 3: 買點標記
    signals = plot_df[(plot_df['Final_Score'] > 15) & (plot_df['Final_Score'].shift(1) <= 15)]
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Final_Score'], mode='markers',
                             marker=dict(symbol='diamond', size=12, color='#FFD700', line=dict(width=1, color="black")),
                             name='買點訊號'), secondary_y=True)

    # 警戒線
    fig.add_hline(y=75, line_dash="dash", line_color="#FF4B4B", secondary_y=True, annotation_text="過熱")
    fig.add_hline(y=25, line_dash="dash", line_color="#00D26A", secondary_y=True, annotation_text="低迷")

    # 佈局優化
    fig.update_layout(
        height=600, 
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]) # 移除週末
    )
    fig.update_yaxes(range=[0, 100], secondary_y=True, title="檔位評分")
    
    # 在 Streamlit 顯示
    st.plotly_chart(fig, use_container_width=True)
    
    # 顯示數據摘要
    col1, col2, col3 = st.columns(3)
    col1.metric("當前檔位", f"{df['Final_Score'].iloc[-1]:.1f}")
    col2.metric("趨勢強度 (ADX)", f"{df['adx'].iloc[-1]:.1f}")
    col3.metric("成交量百分位", f"{df['vol_rank'].iloc[-1]:.1f}%")

else:
    st.warning("請確認代號是否正確或是否有交易數據。")
