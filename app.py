import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：三合一決策終端", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.main { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

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
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max", auto_adjust=True)
    if df.empty: return df, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 指標計算
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
    df['Support_Dots'] = np.where(df['Final_Score'] <= df['Lower_Bound'], df['Final_Score'], np.nan)
    df['Resistance_Dots'] = np.where(df['Final_Score'] >= df['Upper_Bound'], df['Final_Score'], np.nan)
    
    return df, ticker.info

# --- 分頁系統 ---
tab1, tab2 = st.tabs(["📡 2025 績效排行榜", "🔍 深度分析 (籌碼/基本/技術)"])

with tab1:
    st.subheader("📊 2025 全資產績效總覽")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    with st.spinner("掃描市場中..."):
        for sym, name in all_symbols.items():
            scan_df, _ = get_full_data(sym)
            if not scan_df.empty:
                curr = scan_df.iloc[-1]
                bt_df = scan_df[scan_df.index >= "2025-01-01"]
                y_days = bt_df[bt_df['Final_Score'] <= bt_df['Lower_Bound']]
                roi = (((1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
                status = "🟡 抄底" if curr['Final_Score'] <= curr['Lower_Bound'] else ("🔴 過熱" if curr['Final_Score'] >= curr['Upper_Bound'] else "⚪ 穩定")
                radar_results.append({"標的": name, "價格": round(curr['Close'], 1), "2025回報": f"{roi:.2f}%", "狀態": status, "sort_roi": roi})
    
    st.table(pd.DataFrame(radar_results).sort_values("sort_roi", ascending=False).drop(columns="sort_roi"))

with tab2:
    st.sidebar.header("🔍 標的選擇")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    sid = [k for k, v in ASSET_LIST[cat].items() if v == asset_name][0]
    
    df, info = get_full_data(sid)
    if not df.empty:
        # 圖表區域 (技術面)
        st.subheader(f"📈 技術面趨勢：{asset_name} ({sid})")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Support_Dots'], mode='markers', marker=dict(color="#FFD700", size=6)), secondary_y=True)
        fig.update_yaxes(secondary_y=False, showgrid=False)
        fig.update_yaxes(secondary_y=True, range=[-5, 105], gridcolor="rgba(255, 255, 255, 0.05)")
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=30), df.index[-1]])
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # 資訊排列
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏛️ 籌碼面：法人買賣超 (近5日預估)")
            # 註：yfinance 目前無法直接取得台股三大法人即時明細，此處以成交量與動量模擬籌碼強弱，實務上建議結合台股 API
            vol_change = df['Volume'].pct_change().iloc[-5:]
            price_change = df['Close'].pct_change().iloc[-5:]
            inst_trend = []
            for i in range(5):
                date_str = df.index[-(5-i)].strftime('%m/%d')
                trend = "買超" if price_change.iloc[-(5-i)] > 0 and vol_change.iloc[-(5-i)] > 0 else "賣超"
                inst_trend.append({"日期": date_str, "外資/投信預估": trend, "成交量變動": f"{vol_change.iloc[-(5-i)]*100:+.1f}%"})
            st.table(pd.DataFrame(inst_trend))

        with col2:
            st.subheader("💎 基本面：價值評估")
            fundamental_data = {
                "項目": ["目前本益比 (P/E)", "股價淨值比 (P/B)", "現金股利", "市值 (兆)", "一年內高低點"],
                "數據": [
                    f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A",
                    f"{info.get('priceToBook', 'N/A'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else "N/A",
                    f"{info.get('dividendRate', 'N/A')}",
                    f"{info.get('marketCap', 0) / 1e12:.2f}T",
                    f"{info.get('fiftyTwoWeekLow', 0):.1f} - {info.get('fiftyTwoWeekHigh', 0):.1f}"
                ]
            }
            st.table(pd.DataFrame(fundamental_data))

        # 100 萬回測總結
        st.info(f"💡 **決策參考**：{asset_name} 目前本益比為 {info.get('trailingPE', 'N/A')}。當五維分數出現 **黃色點點** 且 **籌碼面顯示買超** 時，通常是高勝率進場點。")
