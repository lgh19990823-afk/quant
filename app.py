import streamlit as st
import akshare as ak
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="量化推演终端", page_icon="📈", layout="centered")

st.title("📈 A股真实量化推演终端")
st.caption("数据源：AkShare | 算法：20日真实ATR + 1万次蒙特卡洛模拟")

stock_code = st.text_input("请输入 6 位 A 股代码", value="600519", max_chars=6)

if st.button("🚀 开始量化计算", type="primary", use_container_width=True):
    with st.spinner("正在抓取真实历史 K 线进行矩阵推演..."):
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20230101", end_date=end_date, adjust="qfq")

            if hist_df.empty:
                st.error("未查到该股票，请检查代码！")
            else:
                stock_name = hist_df['股票名称'].iloc[-1]
                current_price = float(hist_df['收盘'].iloc[-1])

                df_recent = hist_df.tail(20).copy()
                df_recent['tr'] = np.maximum(
                    df_recent['最高'] - df_recent['最低'],
                    np.abs(df_recent['最高'] - df_recent['收盘'].shift(1))
                )
                atr_20 = df_recent['tr'].mean()

                returns = np.log(hist_df['收盘'] / hist_df['收盘'].shift(1)).dropna().tail(60)
                annual_vol = float(np.std(returns) * np.sqrt(252))

                simulations, days, dt = 10000, 5, 1/252
                Z = np.random.normal(0, 1, (simulations, days))
                delta_returns = np.exp((-0.5 * (annual_vol ** 2) * dt) + (annual_vol * np.sqrt(dt) * Z))
                final_prices = current_price * np.prod(delta_returns, axis=1)

                gbm_win_rate = int((np.sum(final_prices > current_price) / simulations) * 100)
                p5, p95 = np.percentile(final_prices, 5), np.percentile(final_prices, 95)

                st.success(f"股票名称：{stock_name} ({stock_code})")

                col1, col2 = st.columns(2)
                col1.metric("最新价格", f"￥{current_price:.2f}")
                col2.metric("20日真实波幅(ATR)", f"￥{atr_20:.2f}")

                st.subheader("📊 未来 5 天概率推演 (10,000次模拟)")
                col3, col4 = st.columns(2)
                col3.metric("5日正期望胜率", f"{gbm_win_rate}%")
                col4.metric("95%置信上轨", f"￥{p95:.2f}")

                st.info(f"💡 动态防守位（止损点参考）：￥{current_price - 1.5 * atr_20:.2f}")

        except Exception as e:
            st.error(f"计算出错: {e}")
