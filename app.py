import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：強勢領航終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# 定義資產清單
ASSET_LIST = {
    "市值前十大公司": {
        "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
        "2881.TW": "富邦金", "2882.TW": "國泰金", "2382.TW": "廣達", "2891.TW": "中信金",
        "3711.TW": "日月光投控", "2412.TW": "中華電"
    },
    "優秀市值型 ETF": {
        "0050.TW": "元大台灣50", "006208.TW": "富邦台50", "00922.TW": "國泰台灣領袖50"
    },
    "熱門高股息 ETF": {
        "0056.TW": "元大高股息", "00878.TW": "國泰永續高股息", "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息"
    }
}

@st.cache_data(ttl=300)
def get_full_data(symbol):
    df = yf.download(symbol, period="max", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 五維指標計算
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    def adaptive_logic(r):
        if pd.isna(r['adx']) or pd.isna(r['atr']): return 50
        base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1) if r['adx'] > 25 else (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        return (base + 50) / 2 if (abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0) > 2.5 else base

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    
    # 計算相對強弱 (對比 0050)
    benchmark = yf.download("0050.TW", period="max", auto_adjust=True)
    if not benchmark.empty:
        if isinstance(benchmark.columns, pd.MultiIndex): benchmark.columns = benchmark.columns.get_level_values(0)
        df['RS'] = df['Close'] / benchmark['Close']
        df['RS_Line'] = df['RS'].rolling(20).mean() # 平滑化
    
    return df

# --- 分頁系統 ---
tab1, tab2 = st.tabs(["📡 績效與訊號排行", "🔍 RS強弱深度分析"])

# --- Tab 1: 績效排行雷達 ---
with tab1:
    st.subheader("📊 2025 全資產策略績效排行榜")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    with st.spinner("掃描市場並回測 100 萬本金中..."):
        for sym, name in all_symbols.items():
            scan_df = get_full_data(sym)
            if not scan_df.empty:
                curr = scan_df.iloc[-1]
                # 2025 回測
                bt_df = scan_df[scan_df.index >= "2025-01-01"]
                y_days = bt_df[bt_df['Final_Score'] <= bt_df['Lower_Bound']]
                roi = (( (1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
                
                status = "⚪ 穩定"
                if curr['Final_Score'] <= curr['Lower_Bound']: status = "🟡 抄底"
                elif curr['Final_Score'] >= curr['Upper_Bound']: status = "🔴 過熱"
                
                radar_results.append({
                    "標的": name, "目前價格": round(curr['Close'], 2),
                    "五維分數": round(curr['Final_Score'], 1),
                    "2025策略回報": f"{roi:.2f}%", "狀態": status, "sort_roi": roi
                })
    
    rank_df = pd.DataFrame(radar_results).sort_values("sort_roi", ascending=False).drop(columns="sort_roi")
    st.table(rank_df)

# --- Tab 2: 深度分析與 RS 線 ---
with tab2:
    st.sidebar.header("🔍 分析設定")
    category = st.sidebar.selectbox("資產類別", list(ASSET_LIST.keys()))
    selected_asset_name = st.sidebar.selectbox("詳細標的", list(ASSET_LIST[category].values()))
    stock_id = [k for k, v in ASSET_LIST[category].items() if v == selected_asset_name][0]
    
    df = get_full_data(stock_id)
    if not df.empty:
        # 標題區
        st.markdown(f"### {selected_asset_name} ({stock_id}) - RS 強弱度與動態邊界")
        
        # 建立三軸圖：主價、指標、RS線
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, row_heights=[0.7, 0.3],
                           specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

        # 1. 股價與五維分數 (Row 1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], name="壓", line=dict(color="rgba(255, 75, 75, 0.3)", width=1, dash='dot')), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], name="撐", line=dict(color="rgba(255, 215, 0, 0.3)", width=1, dash='dot')), row=1, col=1, secondary_y=True)
        
        # 標記點
        df['SD'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
        fig.add_trace(go.Scatter(x=df.index, y=df['SD'], mode='markers', marker=dict(color="#FFD700", size=6), name="抄底區"), row=1, col=1, secondary_y=True)

        # 2. 相對強弱 RS 線 (Row 2) - 對比 0050
        fig.add_trace(go.Scatter(x=df.index, y=df['RS_Line'], name="RS相對強弱", line=dict(color="#E066FF", width=2)), row=2, col=1)

        # 設定美化
        fig.update_yaxes(title_text="股價", row=1, col=1, secondary_y=False, autorange=True, fixedrange=True, showgrid=False)
        fig.update_yaxes(title_text="分數", row=1, col=1, secondary_y=True, range=[-5, 105], fixedrange=True)
        fig.update_yaxes(title_text="對比0050強度", row=2, col=1, showgrid=False)
        
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=30), df.index[-1]], fixedrange=False)
        fig.update_layout(height=700, template="plotly_dark", dragmode="pan", uirevision='constant', showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.info("💡 **RS 線觀察法**：紫線向上代表該股「比大盤強」，若此時出現黃點，代表強勢股回檔，是極佳買點。")
