import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="五維一體：勝率優化終端", layout="wide")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_pro_strategy_data(symbol):
    df = yf.download(symbol, period="2y", auto_adjust=True)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 1. 基礎指標 ---
    df['rsi_r'] = ta.rsi(df['Close'], length=14).rolling(252).rank(pct=True) * 100
    df['bias_r'] = ((df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()).rolling(252).rank(pct=True) * 100
    macd = ta.macd(df['Close'])
    df['macd_h'] = macd['MACDh_12_26_9']
    df['macd_r'] = df['macd_h'].rolling(252).rank(pct=True) * 100
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['sma_200'] = df['Close'].rolling(200).mean() # 年線：大趨勢濾網
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) # 波動率：計算止損

    # --- 2. 核心權重優化 (ADX 自適應) ---
    # 趨勢盤與盤整盤採用不同的權重比例
    df['Final_Score'] = np.where(df['adx'] > 25,
                                 (df['bias_r'] * 0.5 + df['macd_r'] * 0.4 + df['rsi_r'] * 0.1), # 趨勢市：看乖離與動能
                                 (df['rsi_r'] * 0.5 + df['macd_r'] * 0.3 + df['bias_r'] * 0.2)) # 盤整市：看超買超賣
    
    df['Final_Score'] = df['Final_Score'].rolling(10).mean() # 10日平滑降噪

    # --- 3. 買賣訊號 (加入年線濾網) ---
    # 買入：檔位低 + 動能轉向 + 股價在年線上 (順勢而為)
    buy_cond = (df['Final_Score'] < 25) & (df['macd_h'] > df['macd_h'].shift(1)) & (df['Close'] > df['sma_200'])
    # 賣出：檔位高 + 動能轉弱
    sell_cond = (df['Final_Score'] > 75) & (df['macd_h'] < df['macd_h'].shift(1))
    
    df['Buy_Signal'] = buy_cond & (buy_cond.shift(1) == False)
    df['Sell_Signal'] = sell_cond & (sell_cond.shift(1) == False)
    
    return df

st.title("🛡️ 綜合成交量/趨勢/動能/空間：勝率優化版")

stock_id = st.sidebar.text_input("輸入台股代碼", value="2330.TW")
df = get_pro_strategy_data(stock_id)

if not df.empty:
    plot_df = df.tail(252)
    
    # 圖表部分維持乾淨線條
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name="股價", line=dict(color="rgba(150,150,150,0.4)")), secondary_y=False)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Final_Score'], name="綜合檔位線", line=dict(color="#00d26a", width=2.5)), secondary_y=True)
    fig.update_xaxes(tickformat="%Y-%m-%d", dtick="M2", fixedrange=True)
    fig.update_yaxes(secondary_y=True, range=[0, 100], fixedrange=True)
    fig.update_layout(height=450, template="plotly_dark", dragmode=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # --- 表格優化：加入獲利與止損參考 ---
    st.subheader("📋 交易策略明細 (含止損參考)")
    signals = plot_df[(plot_df['Buy_Signal']) | (plot_df['Sell_Signal'])].copy()
    
    if not signals.empty:
        table_data = []
        for index, row in signals.iterrows():
            is_buy = row['Buy_Signal']
            sl_price = row['Close'] - (2 * row['atr']) if is_buy else None # 買入時建議止損設在 2倍ATR 處
            
            table_data.append({
                "日期": index.strftime('%Y-%m-%d'),
                "類型": "🟢 買入" if is_buy else "🔴 賣出",
                "執行價位": f"{row['Close']:.2f}",
                "建議止損價": f"{sl_price:.2f}" if sl_price else "---",
                "當時檔位": f"{row['Final_Score']:.1f}",
                "趨勢狀態": "多頭順向" if row['Close'] > row['sma_200'] else "弱勢反彈"
            })
        st.table(pd.DataFrame(table_data))
    else:
        st.info("目前無符合高勝率條件之訊號。")

    # 策略小教室
    st.warning("⚠️ **為何這能提高勝率？** 我們加入了 **200MA 年線濾網**，系統會自動無視掉「空頭趨勢中的反彈」。雖然訊號變少了，但每一次觸發的品質都會顯著提升。")
