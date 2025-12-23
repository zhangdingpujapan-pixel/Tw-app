import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維歷史全覽終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_historical_data(symbol):
    # 下載該標的所有歷史數據
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 進階指標計算 (維持 252 日滾動排名以保持標準統一) ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['sma_200'] = df['Close'].rolling(200).mean()

    # 自適應權重邏輯
    def adaptive_logic(r):
        if pd.isna(r['adx']): return 50 # 初始數據不足時給中值
        if r['adx'] > 25: return (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1)
        else: return (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    
    return df

st.title("🛡️ 五維歷史全覽：動態縮放終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_historical_data(stock_id)

if not df.empty:
    # 建立雙 Y 軸
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", 
                             line=dict(color="rgba(200,200,200,0.3)", width=1.2)), secondary_y=False)

    # 2. 副 Y 軸 (指標)
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔位", 
                             line=dict(color="#00d26a", width=2)), secondary_y=True)

    # --- 關鍵修正：開啟左右滑動與縮放 ---
    # 設定左軸 (股價)：自動等比縮放
    fig.update_yaxes(
        secondary_y=False, 
        autorange=True, 
        fixedrange=True, # 垂直方向固定，避免上下跳動
        showgrid=False
    )
    
    # 設定右軸 (指標)：固定 0-100
    fig.update_yaxes(
        secondary_y=True, 
        range=[0, 100], 
        fixedrange=True, # 垂直方向固定
        gridcolor="rgba(255,255,255,0.05)"
    )

    # 設定 X 軸 (日期)：允許縮放與滑動
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        rangeslider_visible=False, # 隱藏下方的小滑桿以節省手機空間
        fixedrange=False           # 允許左右滑動與縮放
    )

    # 設定初始顯示範圍 (預設看最近一年，但可以往左滑)
    start_date = df.index[-252] if len(df) > 252 else df.index[0]
    end_date = df.index[-1]
    fig.update_xaxes(range=[start_date, end_date])

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified", 
        dragmode="pan",           # 設定預設模式為「平移」，方便手指滑動
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    # 針對手機操作優化
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,       # 允許滾輪/雙指縮放
        'displayModeBar': False   # 隱藏工具列
    })
    
    # 底部數據
    curr = df.iloc[-1]
    st.info(f"💡 **操作指南**：現在可以手動**左右滑動**查看歷史紀錄。雙指撥弄可放大縮小。目前顯示：{stock_id} 從上市至今的所有數據。")
    st.metric("當前綜合檔位", f"{curr['Final_Score']:.1f}")

else:
    st.error("代碼錯誤或無歷史數據。")
