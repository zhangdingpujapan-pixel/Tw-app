import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

# 設定頁面
st.set_page_config(page_title="VCP 形態偵測器", layout="wide")

st.title("📈 VCP 波動收縮型態視覺化")
st.sidebar.header("設定參數")

# 1. 使用者輸入
ticker = st.sidebar.text_input("輸入股票代號", value="NVDA").upper()
period = st.sidebar.selectbox("觀測區間", ["6mo", "1y", "2y"], index=1)
distance = st.sidebar.slider("收縮點偵測靈敏度", 5, 30, 15)

@st.cache_data
def load_data(symbol, p):
    df = yf.download(symbol, period=p)
    return df

try:
    data = load_data(ticker, period)
    
    if data.empty:
        st.error("找不到該股票數據，請重新檢查代號。")
    else:
        # 2. VCP 演算法：找出局部高點 (Pivot points)
        # 使用收盤價來尋找峰值
        prices = data['Close'].values
        peaks, _ = find_peaks(prices, distance=distance)
        
        # 3. 繪製 K 線圖
        fig = go.Figure()

        # 加入 K 線
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="K線"
        ))

        # 4. 標註收縮波浪與計算幅度
        vcp_details = []
        last_peak_idx = None
        
        for i in range(len(peaks)):
            current_idx = peaks[i]
            # 尋找前一個高點到目前高點之間的最低點 (回撤深度)
            if i > 0:
                prev_idx = peaks[i-1]
                trough_price = data['Low'].iloc[prev_idx:current_idx].min()
                prev_peak_price = data['High'].iloc[prev_idx]
                drawdown = (trough_price - prev_peak_price) / prev_peak_price * 100
                
                vcp_details.append({
                    "日期": data.index[current_idx].strftime('%Y-%m-%d'),
                    "幅度": f"{drawdown:.2f}%"
                })

                # 在圖上畫出收縮連線
                fig.add_annotation(
                    x=data.index[current_idx],
                    y=data['High'].iloc[current_idx],
                    text=f"收縮 {drawdown:.1f}%",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )

        fig.update_layout(
            title=f"{ticker} VCP 潛在收縮點偵測",
            yaxis_title="價格",
            xaxis_rangeslider_visible=False,
            height=700
        )

        # 顯示圖表
        st.plotly_chart(fig, use_container_width=True)

        # 5. 顯示分析結果
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("收縮紀錄 (V's)")
            st.table(pd.DataFrame(vcp_details))
        
        with col2:
            st.info("""
            **VCP 觀察要點：**
            1. **幅度收窄**：每次回撤的百分比應該逐漸變小（例如：-25% -> -12% -> -5%）。
            2. **波浪次數**：通常出現 2 到 4 次收縮最為理想。
            3. **底部墊高**：低點不應跌破前一波大幅回撤的低點。
            """)

except Exception as e:
    st.write("請輸入正確的代號以開始分析。")
