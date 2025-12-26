import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：純淨分頁版", layout="wide", initial_sidebar_state="collapsed")
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
    
    # 原始五維指標計算 (不含 2.0 的複雜過濾)
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    def adaptive_logic(r):
        if pd.isna(r['adx']) or pd.isna(r['atr']): return 50
        base = (r['bias_r'] * 0.6 + r['macd_r'] * 0.3 + r['rsi_r'] * 0.1) if r['adx'] > 25 else (r['rsi_r'] * 0.5 + r['macd_r'] * 0.3 + r['bias_r'] * 0.2)
        # 簡單的極端波動處理
        return (base + 50) / 2 if (abs(r['Close'] - r['Open']) / r['atr'] if r['atr'] != 0 else 0) > 2.5 else base

    df['Final_Score'] = df.apply(adaptive_logic, axis=1).rolling(10).mean()
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    df['is_support'] = df['Final_Score'] <= df['Lower_Bound']
    
    return df, ticker.info

# --- UI 介面 ---
tab1, tab2 = st.tabs(["📡 績效排行榜", "🔍 深度分析"])

with tab1:
    st.subheader("📊 2025 全資產績效總覽")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    for sym, name in all_symbols.items():
        scan_df, _ = get_full_data(sym)
        if not scan_df.empty:
            curr = scan_df.iloc[-1]
            bt_df = scan_df[scan_df.index >= "2025-01-01"]
            y_days = bt_df[bt_df['is_support']]
            roi = (((1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
            status = "🟡 抄底" if curr['is_support'] else ("🔴 過熱" if curr['Final_Score'] >= curr['Upper_Bound'] else "⚪ 穩定")
            radar_results.append({"標的": name, "價格": round(curr['Close'], 1), "2025回報": f"{roi:.2f}%", "狀態": status, "sort_roi": roi})
    
    st.table(pd.DataFrame(radar_results).sort_values("sort_roi", ascending=False).drop(columns="sort_roi"))

with tab2:
    st.sidebar.header("🔍 標的選擇")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    sid = [k for k, v in ASSET_LIST[cat].items() if v == asset_name][0]
    
    df, info = get_full_data(sid)
    if not df.empty:
        # 技術圖表
        st.subheader(f"📈 技術面趨勢：{asset_name} ({sid})")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], line=dict(color="rgba(255, 75, 75, 0.4)", width=1, dash='dot')), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], line=dict(color="rgba(255, 215, 0, 0.4)", width=1, dash='dot')), secondary_y=True)
        
        support_df = df[df['is_support']]
        fig.add_trace(go.Scatter(x=support_df.index, y=support_df['Final_Score'], mode='markers', marker=dict(color="#FFD700", size=8)), secondary_y=True)
        
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=30), df.index[-1]])
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=50, r=50, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 分頁式籌碼與訊號紀錄 ---
        st.markdown("---")
        st.subheader("🏛️ 歷史紀錄查詢 (每頁 10 筆)")
        
        full_history = df.tail(252).copy() # 取一年紀錄
        vol_change = full_history['Volume'].pct_change()
        price_change = full_history['Close'].pct_change()
        
        all_records = []
        for i in range(len(full_history)-1, -1, -1):
            row = full_history.iloc[i]
            all_records.append({
                "日期": full_history.index[i].strftime('%Y/%m/%d'),
                "訊號": "🟡 抄底" if row['is_support'] else "", # 這裡會在抄底區做記號
                "收盤價": f"{row['Close']:.2f}",
                "法人預估": "買超" if (price_change.iloc[i] > 0 and vol_change.iloc[i] > 0) else "賣超",
                "量能增減": f"{vol_change.iloc[i]*100:+.1f}%" if not pd.isna(vol_change.iloc[i]) else "--"
            })
        
        # 分頁邏輯
        if 'page_idx' not in st.session_state: st.session_state.page_idx = 0
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ 上一頁") and st.session_state.page_idx > 0: st.session_state.page_idx -= 1
        with c3:
            if st.button("下一頁 ➡️") and st.session_state.page_idx < (len(all_records)//10): st.session_state.page_idx += 1
        
        start = st.session_state.page_idx * 10
        st.table(pd.DataFrame(all_records[start : start+10]))

        # 基本面輔助
        st.markdown("---")
        st.write(f"目前 P/E: {info.get('trailingPE', 'N/A')} | P/B: {info.get('priceToBook', 'N/A')} | 市值: {info.get('marketCap', 0)/1e12:.2f}T")
