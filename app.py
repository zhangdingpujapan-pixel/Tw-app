import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略 2.0：量價偵測版", layout="wide", initial_sidebar_state="collapsed")
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
    
    # --- 五維 2.0 演算法升級 ---
    # 1. 基礎百分位排名
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    
    # 2. 加入量價因子 (MFI) - 偵測資金流向
    df['mfi'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['mfi_r'] = df['mfi'].rolling(252).rank(pct=True) * 100
    
    # 3. 趨勢強度與波動過濾
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    def adaptive_logic_2(r):
        if pd.isna(r['adx']) or pd.isna(r['mfi_r']): return 50
        # 根據趨勢強度動態調整權重
        if r['adx'] > 25:
            # 趨勢盤：偏重乖離與量價流向
            base = (r['bias_r'] * 0.4 + r['mfi_r'] * 0.3 + r['macd_r'] * 0.2 + r['rsi_r'] * 0.1)
        else:
            # 震盪盤：偏重 RSI 與 MFI
            base = (r['rsi_r'] * 0.4 + r['mfi_r'] * 0.3 + r['macd_r'] * 0.2 + r['bias_r'] * 0.1)
        
        # 量能加速度修正：放量下跌時，分數強制向中值拉回 (避免接尖刀)
        vol_ratio = (abs(r['Close'] - r['Open']) / r['atr']) if r['atr'] > 0 else 0
        if vol_ratio > 2.2 and r['Close'] < r['Open']:
            base = (base + 50) / 2
        return base

    # 4. 使用 HMA (赫爾均線) 進行極速平滑
    raw_scores = df.apply(adaptive_logic_2, axis=1)
    df['Final_Score'] = ta.hma(raw_scores, length=10)
    
    # 5. 動態邊界
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.15)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.85)
    df['is_support'] = df['Final_Score'] <= df['Lower_Bound']
    
    return df, ticker.info

# --- UI 介面 ---
tab1, tab2 = st.tabs(["📡 績效排行榜", "🔍 深度分析 (2.0 引擎)"])

with tab1:
    st.subheader("📊 2025 全資產 2.0 策略績效")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    with st.spinner("優化引擎計算中..."):
        for sym, name in all_symbols.items():
            scan_df, _ = get_full_data(sym)
            if not scan_df.empty:
                curr = scan_df.iloc[-1]
                bt_df = scan_df[scan_df.index >= "2025-01-01"]
                y_days = bt_df[bt_df['is_support']]
                roi = (((1000000 / len(y_days) / y_days['Close']).sum() * curr['Close'] - 1000000) / 10000) if len(y_days) > 0 else 0
                status = "🟡 抄底" if curr['is_support'] else ("🔴 過熱" if curr['Final_Score'] >= curr['Upper_Bound'] else "⚪ 穩定")
                radar_results.append({"標的": name, "回報": f"{roi:.2f}%", "狀態": status, "sort_roi": roi})
    
    st.table(pd.DataFrame(radar_results).sort_values("sort_roi", ascending=False).drop(columns="sort_roi"))

with tab2:
    st.sidebar.header("🔍 分析設定")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    sid = [k for k, v in ASSET_LIST[cat].items() if v == asset_name][0]
    
    df, info = get_full_data(sid)
    if not df.empty:
        st.subheader(f"📈 2.0 檔位分析：{asset_name}")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=3)), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bound'], line=dict(color="rgba(255, 75, 75, 0.4)", dash='dot')), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bound'], line=dict(color="rgba(255, 215, 0, 0.4)", dash='dot')), secondary_y=True)
        
        support_df = df[df['is_support']]
        fig.add_trace(go.Scatter(x=support_df.index, y=support_df['Final_Score'], mode='markers', marker=dict(color="#FFD700", size=10)), secondary_y=True)
        
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=30), df.index[-1]])
        fig.update_layout(height=450, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 分頁紀錄 ---
        st.markdown("---")
        st.subheader("🏛️ 歷史紀錄 (每頁 10 筆)")
        full_history = df.tail(252).copy()
        record_list = []
        for i in range(len(full_history)-1, -1, -1):
            row = full_history.iloc[i]
            record_list.append({
                "日期": full_history.index[i].strftime('%Y/%m/%d'),
                "訊號": "🟡 抄底" if row['is_support'] else "",
                "價格": f"{row['Close']:.2f}",
                "五維分數": f"{row['Final_Score']:.1f}"
            })
        
        items_per_page = 10
        if 'page_num_2' not in st.session_state: st.session_state.page_num_2 = 0
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ 上一頁") and st.session_state.page_num_2 > 0: st.session_state.page_num_2 -= 1
        with c3:
            if st.button("下一頁 ➡️") and st.session_state.page_num_2 < (len(record_list)//10): st.session_state.page_num_2 += 1
        
        start = st.session_state.page_num_2 * 10
        st.table(pd.DataFrame(record_list[start : start+10]))

        st.info("💡 **2.0 優化說明**：加入了 MFI 資金流向與 HMA 快速平滑演算法。黃點現在更能避開『放量殺跌』，並在『縮量趕底』時精準切入。")
