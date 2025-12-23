import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面基本設定
st.set_page_config(page_title="台股精準檔位 App", layout="wide", initial_sidebar_state="expanded")

# 自定義 CSS 優化手機視覺
st.markdown("""
    <style>
    .stMetric { background-color: #1e212b; padding: 15px; border-radius: 10px; border: 1px solid #3e424b; }
    [data-testid="stMetricValue"] { color: #00d26a; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 台股核心標的精準回測")

# --- 1. 定義推薦清單 ---
# 包含市值前十大與熱門市值型 ETF
top_stocks = {
    "2330.TW": "台積電 (2330)",
    "2317.TW": "鴻海 (2317)",
    "2454.TW": "聯發科 (2454)",
    "2881.TW": "富邦金 (2881)",
    "2882.TW": "國泰金 (2882)",
    "2308.TW": "台達電 (2308)",
    "2412.TW": "中華電 (2412)",
    "2891.TW": "中信金 (2891)",
    "2382.TW": "廣達 (2382)",
    "1303.TW": "南亞 (1303)",
    "0050.TW": "元大台灣50 (ETF)",
    "006208.TW": "富邦台50 (ETF)"
}

# --- 側邊欄參數 ---
with st.sidebar:
    st.header("🔍 選股與設定")
    
    # 推薦清單下拉選單
    selected_name = st.selectbox("熱門權值股 / ETF", options=list(top_stocks.values()))
    
    # 從名稱反推代碼
    default_stock = [k for k, v in top_stocks.items() if v == selected_name][0]
    
    # 也可以手動輸入其他代碼
    stock_id = st.text_input("手動輸入代碼 (例: 2603.TW)", value=default_stock)
    
    lookback_date = st.date_input("數據回溯起點", value=pd.to_datetime("2024-01-01"))
    
    st.divider()
    st.caption("註：台股請記得加 .TW，如 2330.TW")

# --- 數據處理函數 ---
@st.cache_data(ttl=3600) # 快取一小時，增加反應速度
def get_processed_data(symbol, start_date):
    df = yf.download(symbol, start=start_date, auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 計算邏輯 (保留您原始的精準算法)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['willr'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    df['bias'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['vol_rank'] = df['Volume'].rolling(252, min_periods=1).rank(pct=True) * 100

    # 歷史百分位轉換
    for col in ['rsi', 'willr', 'bias']:
        df[f'{col}_r'] = df[col].rolling(252, min_periods=1).rank(pct=True) * 100

    # 動態權重與量價修正
    df['Composite'] = np.where(df['adx'] > 25,
                               (df['bias_r'] * 0.6 + df['rsi_r'] * 0.2 + df['willr_r'] * 0.2),
                               (df['bias_r'] * 0.2 + df['rsi_r'] * 0.4 + df['willr_r'] * 0.4))
    
    df['Final_Score'] = np.where((df['Composite'] > 70) & (df['vol_rank'] < 30), 
                                 df['Composite'] - 15, df['Composite'])
    
    df['Final_Score'] = df['Final_Score'].rolling(5, min_periods=1).mean()
    return df

# --- 顯示結果 ---
df = get_processed_data(stock_id, lookback_date)

if not df.empty:
    # 頂部數據卡片
    current_val = df['Final_Score'].iloc[-1]
    prev_val = df['Final_Score'].iloc[-2]
    diff = current_val - prev_val
    
    col1, col2, col3 = st.columns(3)
    col1.metric("當前綜合檔位", f"{current_val:.1f}", delta=f"{diff:.1f}")
    
    # 狀態判斷
    if current_val < 20: status = "極度低迷 (買點關注)"
    elif current_val > 80: status = "高度過熱 (注意風險)"
    else: status = "中性區間"
    col2.metric("當前市場狀態", status)
    
    # 成交量狀態
    vol_p = df['vol_rank'].iloc[-1]
    col3.metric("成交量百分位", f"{vol_p:.1f}%")

    # 繪圖
    plot_df = df.tail(120).copy() # 預設顯示最近 120 根 K 線
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 價格線
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價",
                             line=dict(color="rgba(100,100,100,0.3)", width=1)), secondary_y=False)
    
    # 檔位線
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="綜合檔位",
                             line=dict(color="#007bff", width=3)), secondary_y=True)

    # 買點觸發點
    signals = plot_df[(plot_df['Final_Score'] > 15) & (plot_df['Final_Score'].shift(1) <= 15)]
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Final_Score'], mode='markers',
                             marker=dict(symbol='star', size=12, color='gold'), name='買點觸發'), secondary_y=True)

    fig.add_hline(y=75, line_dash="dash", line_color="red", secondary_y=True)
    fig.add_hline(y=25, line_dash="dash", line_color="green", secondary_y=True)

    fig.update_layout(height=550, template="plotly_white", hovermode="x unified",
                      margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.error("無法抓取數據，請檢查代號是否正確。")
