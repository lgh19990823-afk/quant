import streamlit as st
import akshare as ak
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="A股真实量化推演终端", layout="wide")

st.title("📈 A股真实量化推演终端")
st.caption("数据源：AkShare | 算法：20日真实ATR + 1万次蒙特卡洛模拟")

stock_code = st.text_input("请输入 6 位 A 股代码", value="002160")

def fetch_data(code):
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}")
        if df is not None and not df.empty:
            df = df.rename(columns={
                'date': '日期', 'open': '开盘', 'high': '最高', 
                'low': '最低', 'close': '收盘', 'volume': '成交量'
            })
            df['日期'] = pd.to_datetime(df['日期'])
            return df
    except Exception:
        pass
    
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    if df is not None and not df.empty:
        df['日期'] = pd.to_datetime(df['日期'])
    return df

if st.button("🚀 开始量化计算"):
    with st.spinner("正在拉取行情并计算推演数据..."):
        try:
            hist_df = fetch_data(stock_code)
            
            if hist_df is None or hist_df.empty:
                st.error("未能获取到该股票的行情数据，请检查代码是否正确。")
            else:
                st.success("数据获取成功！")
                
                # 1. 计算 20日 ATR
                high_low = hist_df['最高'] - hist_df['最低']
                high_close = np.abs(hist_df['最高'] - hist_df['收盘'].shift(1))
                low_close = np.abs(hist_df['最低'] - hist_df['收盘'].shift(1))
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr20 = tr.rolling(20).mean().iloc[-1]
                
                last_price = hist_df['收盘'].iloc[-1]
                stop_loss = last_price - (2 * atr20)
                
                st.subheader("📊 波动率与止损位 (ATR)")
                col1, col2, col3 = st.columns(3)
                col1.metric("当前收盘价", f"{last_price:.2f} 元")
                col2.metric("20日 ATR 波动值", f"{atr20:.2f} 元")
                col3.metric("推荐移动止损位 (2xATR)", f"{stop_loss:.2f} 元")
                
                # 2. 10000 次蒙特卡洛模拟 (5天)
                returns = np.log(hist_df['收盘'] / hist_df['收盘'].shift(1)).dropna()
                mu = returns.mean()
                sigma = returns.std()
                
                n_simulations = 10000
                n_days = 5
                
                simulated_end_prices = []
                for _ in range(n_simulations):
                    price_path = [last_price]
                    for _ in range(n_days):
                        drift = mu - (0.5 * sigma**2)
                        shock = sigma * np.random.normal()
                        price_next = price_path[-1] * np.exp(drift + shock)
                        price_path.append(price_next)
                    simulated_end_prices.append(price_path[-1])
                
                simulated_end_prices = np.array(simulated_end_prices)
                
                prob_up = (simulated_end_prices > last_price).mean() * 100
                p50 = np.percentile(simulated_end_prices, 50)
                p95 = np.percentile(simulated_end_prices, 95)
                p5 = np.percentile(simulated_end_prices, 5)
                
                st.subheader("🎲 5日蒙特卡洛价格概率推演 (1万次模拟)")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("上涨概率", f"{prob_up:.1f}%")
                col_b.metric("中位数预测价 (P50)", f"{p50:.2f} 元")
                col_c.metric("乐观目标价 (P95)", f"{p95:.2f} 元")
                col_d.metric("悲观底线价 (P5)", f"{p5:.2f} 元")
                
                st.line_chart(hist_df.set_index('日期')['收盘'].tail(60))
                st.caption("注：以上推演仅供量化模型演示，不构成任何投资建议。")
                
        except Exception as e:
            st.error(f"计算出错: {e}")
