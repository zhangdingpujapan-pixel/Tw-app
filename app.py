import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面基礎設定
st.set_page_config(page_title="五維策略：移動止盈監控版", layout="wide", initial_sidebar_state="collapsed")
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
    
    # --- 1. 指標計算 ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
    df['macd_r'] = macd['MACDh_6_13_5'].rolling(252).rank(pct=True) * 100
    df['mfi_r'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14).rolling(252).rank(pct=True) * 100
    
    raw_scores = (df['rsi_r'] * 0.3 + df['bias_r'] * 0.3 + df['macd_r'] * 0.2 + df['mfi_r'] * 0.2)
    df['Final_Score'] = ta.hma(raw_scores, length=8)
    
    df['Lower_Bound'] = df['Final_Score'].rolling(252).quantile(0.10)
    df['Upper_Bound'] = df['Final_Score'].rolling(252).quantile(0.90)
    
    # --- 2. 移動止盈邏輯 (回檔 5%) ---
    trailing_percent = 0.05
    df['is_bottom'] = df['Final_Score'] <= df['Lower_Bound']
    df['is_exit_score'] = (df['Final_Score'].shift(1) >= df['Upper_Bound']) & (df['Final_Score'] < df['Upper_Bound'])
    
    # 計算買入後的最高點
    df['trailing_stop'] = np.nan
    df['is_trailing_exit'] = False
    
    last_buy_idx = -1
    highest_price = 0
    
    for i in range(len(df)):
        if df['is_bottom'].iloc[i]:
            last_buy_idx = i
            highest_price = df['Close'].iloc[i]
        
        if last_buy_idx != -1:
            if df['Close'].iloc[i] > highest_price:
                highest_price = df['Close'].iloc[i]
            
            # 如果價格低於最高點的 95%，觸發移動止盈
            stop_price = highest_price * (1 - trailing_percent)
            df.iloc[i, df.columns.get_loc('trailing_stop')] = stop_price
            
            if df['Close'].iloc[i] < stop_price:
                df.iloc[i, df.columns.get_loc('is_trailing_exit')] = True
                last_buy_idx = -1 # 重置買入狀態，直到下一個黃點
                highest_price = 0
                
    return df, ticker.info

# --- UI ---
tab1, tab2 = st.tabs(["📡 實時移動止盈監測", "🔍 深度轉折分析"])

with tab1:
    st.subheader("📊 2025 全資產移動止盈狀態")
    all_symbols = {}
    for cat in ASSET_LIST: all_symbols.update(ASSET_LIST[cat])
    
    radar_results = []
    for sym, name in all_symbols.items():
        scan_df, _ = get_full_data(sym)
        if not scan_df.empty:
            curr = scan_df.iloc[-1]
            status = "⚪ 持有/觀望"
            if curr['is_bottom']: status = "🟡 買入(底)"
            elif curr['is_trailing_exit']: status = "🟣 移動止盈(回檔5%)"
            elif curr['is_exit_score']: status = "🔵 分數轉弱停利"
            
            radar_results.append({
                "標的": name, 
                "目前價格": round(curr['Close'], 1), 
                "狀態": status, 
                "離最高點回檔": f"{((curr['Close']/scan_df['Close'].tail(20).max()-1)*100):+.1f}%"
            })
    st.table(pd.DataFrame(radar_results))

with tab2:
    st.sidebar.header("🔍 分析設定")
    cat = st.sidebar.selectbox("類別", list(ASSET_LIST.keys()))
    asset_name = st.sidebar.selectbox("標的", list(ASSET_LIST[cat].values()))
    sid = [k for k, v in ASSET_LIST[cat].items() if v == asset_name][0]
    
    df, info = get_full_data(sid)
    if not df.empty:
        st.subheader(f"📈 {asset_name}：移動止盈與五維監控")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 股價與移動止盈線
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="價", line=dict(color="#FFFFFF", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['trailing_stop'], name="移動止盈線", line=dict(color="rgba(160, 32, 240, 0.4)", dash='dash')), secondary_y=False)
        
        # 五維分數
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Score'], name="檔", line=dict(color="#00BFFF", width=2.5)), secondary_y=True)
        
        # 標記訊號
        bottoms = df[df['is_bottom']]
        score_exits = df[df['is_exit_score']]
        trail_exits = df[df['is_trailing_exit']]
        
        fig.add_trace(go.Scatter(x=bottoms.index, y=bottoms['Close'], mode='markers', marker=dict(color="#FFD700", size=10, symbol="triangle-up"), name="買"), secondary_y=False)
        fig.add_trace(go.Scatter(x=score_exits.index, y=score_exits['Close'], mode='markers', marker=dict(color="#00FFFF", size=10, symbol="triangle-down"), name="分數賣"), secondary_y=False)
        fig.add_trace(go.Scatter(x=trail_exits.index, y=trail_exits['Close'], mode='markers', marker=dict(color="#A020F0", size=12, symbol="x"), name="移動止盈賣"), secondary_y=False)
        
        fig.update_xaxes(range=[df.index[-1] - pd.Timedelta(days=90), df.index[-1]])
        fig.update_layout(height=450, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # 紀錄表
        st.markdown("---")
        st.subheader("🗓️ 歷史操作紀錄")
        full_h = df.tail(252).copy()
        recs = []
        for i in range(len(full_h)-1, -1, -1):
            r = full_h.iloc[i]
            sig = ""
            if r['is_bottom']: sig = "🟡 買入"
            elif r['is_trailing_exit']: sig = "🟣 移動止盈"
            elif r['is_exit_score']: sig = "🔵 分數轉弱"
            recs.append({"日期": full_h.index[i].strftime('%Y/%m/%d'), "訊號": sig, "價格": f"{r['Close']:.2f}", "移動止盈點": f"{r['trailing_stop']:.1f}" if not pd.isna(r['trailing_stop']) else "--"})
        
        if 'p5' not in st.session_state: st.session_state.p5 = 0
        c1, c2, c3 = st.columns([1,2,1])
        with c1: 
            if st.button("⬅️ 上一頁"): st.session_state.p5 = max(0, st.session_state.p5-1)
        with c3: 
            if st.button("下一頁 ➡️"): st.session_state.p5 += 1
        
        st.table(pd.DataFrame(recs[st.session_state.p5*10 : st.session_state.p5*10+10]))

        st.info("💡 **移動止盈說明**：系統會在買入後自動跟蹤最高價，一旦股價從波段高點回檔 5% (🟣 紫色 X)，即判定趨勢反轉並離場，這能幫你鎖定大部分利潤。")
