import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維歷史全覽：動態對齊版", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_historical_data(symbol):
    # 下載完整歷史數據
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 指標計算 (252日滾動排名) ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_h'] = macd['MACDh_6_13_5']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']

    # 自適應權重邏輯
    def adaptive_logic(r):
        if pd.isna(r['adx']): return 50
        if r['adx'] > 25: return (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1)
        else: return (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    
    return df

st.title("🛡️ 五維自適應：視窗動態對齊終端")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_historical_data(stock_id)

if not df.empty:
    # 建立雙 Y 軸
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 主 Y 軸 (股價)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", 
                             line=dict(color="rgba(255, 255, 255, 0.5)", width=1.5)), secondary_y=False)

    # 2. 副 Y 軸 (指標)
    fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", 
                             line=dict(color="#00d26a", width=2.5)), secondary_y=True)

    # --- 核心優化：視窗動態對應 ---
    
    # 設定左軸 (股價)：關鍵在於不設死 range，讓它隨視窗 autorange
    fig.update_yaxes(
        secondary_y=False, 
        autorange=True,      # 關鍵：根據目前顯示的 X 軸範圍自動調整 Y 軸高低
        fixedrange=False,    # 允許為了對齊進行動態跳動
        showgrid=False,
        title_text="目前視窗股價"
    )
    
    # 設定右軸 (指標)：始終固定 0-100
    fig.update_yaxes(
        secondary_y=True, 
        range=[0, 100], 
        fixedrange=True,     # 指標軸不隨動，保持 0-100 標準化
        gridcolor="rgba(255,255,255,0.05)",
        title_text="綜合檔位"
    )

    # 設定 X 軸 (日期)：允許自由滑動
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        fixedrange=False,
        rangeslider_visible=False
    )

    # 初始視窗：預設看最近一年
    start_date = df.index[-252] if len(df) > 252 else df.index[0]
    fig.update_xaxes(range=[start_date, df.index[-1]])

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        hovermode="x unified", 
        dragmode="pan", # 預設平移模式，滑動最順手
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    # 隱藏工具欄，啟用滾輪/雙指縮放
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': False
    })
    
    # 底部最新數據提示
    curr = df.iloc[-1]
    st.success(f"📈 **動態對齊已開啟**：現在當你左右滑動時，左側股價軸會自動根據該時段的最高/最低價調整高度，確保綠色檔位線與股價永遠完美重疊。")
    st.metric("最新綜合檔位", f"{curr['Final_Score']:.1f}")

else:
    st.error("讀取失敗，請確認代碼。")
