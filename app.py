import streamlit as st
import akshare as ak
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- 页面基本配置 ---
st.set_page_config(page_title="A股真实量化推演终端", layout="wide")

st.title("📈 A股真实量化推演终端")
st.caption("数据源: AkShare | 算法: 20日真实ATR + 10,000次蒙特卡洛模拟")

# --- 股票代码输入 ---
stock_code = st.text_input("请输入 6 位 A 股代码 (例如: 600519 或 000001):", value="600519").strip()

# --- 数据获取函数 (含多数据源降级备用) ---
@st.cache_data(ttl=3600)
def fetch_data(code):
    prefix = "sh" if code.startswith("6") else "sz"
    
    # 尝试接口 1: stock_zh_a_daily
    try:
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", start_date="20230101")
        if df is not None and not df.empty:
            df = df.rename(columns={
                'date': '日期', 'open': '开盘', 
                'low': '最低', 'close': '收盘', 'high': '最高'
            })
            df['日期'] = pd.to_datetime(df['日期'])
            return df[['日期', '开盘', '最高', '最低', '收盘']]
    except Exception:
        pass

    # 尝试接口 2: stock_zh_a_hist
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20230101", adjust="qfq")
        if df is not None and not df.empty:
            df = df.rename(columns={
                '日期': '日期', '开盘': '开盘', 
                '最高': '最高', '最低': '最低', '收盘': '收盘'
            })
            df['日期'] = pd.to_datetime(df['日期'])
            return df[['日期', '开盘', '最高', '最低', '收盘']]
    except Exception:
        pass

    return None

if stock_code:
    with st.spinner("正在获取行情数据并计算中..."):
        df = fetch_data(stock_code)

    if df is None or len(df) < 30:
        st.error("无法获取该股票数据，请检查股票代码输入是否正确，或稍后重试。")
    else:
        # --- 1. 计算 20 日 ATR (真实波动幅值) ---
        df['prev_close'] = df['收盘'].shift(1)
        df['tr1'] = df['最高'] - df['最低']
        df['tr2'] = (df['最高'] - df['prev_close']).abs()
        df['tr3'] = (df['最低'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=20).mean()

        last_price = df['收盘'].iloc[-1]
        atr_val = df['atr'].iloc[-1]

        # 计算日对数收益率与波动率
        df['log_ret'] = np.log(df['收盘'] / df['收盘'].shift(1))
        daily_vol = df['log_ret'].std()
        daily_drift = df['log_ret'].mean() - 0.5 * (daily_vol ** 2)

        # --- 基础数据卡片显示 ---
        st.subheader("📌 基础行情与波动指标")
        c1, c2, c3 = st.columns(3)
        c1.metric("最新收盘价", f"¥{last_price:.2f}")
        c2.metric("20日 ATR (真实波动)", f"±¥{atr_val:.2f}")
        c3.metric("日波动率", f"{daily_vol*100:.2f}%")

        # --- 2. 蒙特卡洛模拟 (10,000 次，推演 30 天) ---
        sim_days = 30
        num_simulations = 10000

        # 生成随机矩阵 (使用标准正态分布)
        Z = np.random.normal(0, 1, (num_simulations, sim_days))
        daily_returns = np.exp(daily_drift + daily_vol * Z)

        # 模拟价格路径 (广播计算)
        simulation_results = np.zeros((num_simulations, sim_days))
        simulation_results[:, 0] = last_price * daily_returns[:, 0]
        for t in range(1, sim_days):
            simulation_results[:, t] = simulation_results[:, t-1] * daily_returns[:, t]

        # --- 3. 🔮 次日（第1个交易日）涨跌概率预测 ---
        st.markdown("---")
        st.subheader("🔮 次日（第1个交易日）涨跌概率预测")

        day1_prices = simulation_results[:, 0]
        up_count = np.sum(day1_prices > last_price)
        down_count = np.sum(day1_prices < last_price)
        total_sims = len(day1_prices)

        prob_up = (up_count / total_sims) * 100
        prob_down = (down_count / total_sims) * 100
        avg_day1_price = np.mean(day1_prices)
        expected_change = ((avg_day1_price - last_price) / last_price) * 100

        col_up, col_down, col_exp = st.columns(3)
        col_up.metric("次日上涨概率", f"{prob_up:.2f}%", delta=f"{prob_up:.1f}%")
        col_down.metric("次日下跌概率", f"{prob_down:.2f}%", delta=f"-{prob_down:.1f}%", delta_color="inverse")
        col_exp.metric("次日预期平均涨跌", f"{expected_change:+.2f}%")

        if prob_up > 55:
            st.success(f"💡 模拟结果显示：次日偏向**上涨**（胜率 {prob_up:.1f}%）。")
        elif prob_down > 55:
            st.warning(f"💡 模拟结果显示：次日偏向**下跌**（概率 {prob_down:.1f}%）。")
        else:
            st.info("💡 模拟结果显示：次日多空势力较为**均衡**，无明显单边倾向。")

        # --- 4. 🎯 当日量化操作指示 ---
        st.markdown("---")
        st.subheader("🎯 当日量化操作指示")

        take_profit = last_price + (1.5 * atr_val)  # 止盈位：+1.5倍 ATR
        stop_loss = last_price - (1.0 * atr_val)    # 止损位：-1.0倍 ATR

        if prob_up >= 55.0:
            signal = "🚀 建议买入 / 持股"
            reason = f"次日推演胜率达 {prob_up:.1f}%，具备较好短期上冲动能。"
            st.success(f"### 当前信号：{signal}\n\n**决策依据**：{reason}")
        elif prob_down >= 55.0:
            signal = "⚠️ 建议卖出 / 规避"
            reason = f"次日推演下跌概率达 {prob_down:.1f}%，短期回调风险较高。"
            st.error(f"### 当前信号：{signal}\n\n**决策依据**：{reason}")
        else:
            signal = "⏸️ 建议观望 / 保持当前仓位"
            reason = "次日多空力量较为均衡，方向不明确，建议等待突破信号。"
            st.info(f"### 当前信号：{signal}\n\n**决策依据**：{reason}")

        c_tp, c_sl, c_atr = st.columns(3)
        c_tp.metric("🎯 建议止盈参考价", f"¥{take_profit:.2f}", delta=f"+{(1.5*atr_val):.2f}")
        c_sl.metric("🛡️ 建议止损参考价", f"¥{stop_loss:.2f}", delta=f"-{(1.0*atr_val):.2f}")
        c_atr.metric("📊 20日单日波动(ATR)", f"±¥{atr_val:.2f}")

        # --- 5. 📈 未来 3 至 5 天走向预测 ---
        st.markdown("---")
        st.subheader("📈 未来 3 至 5 天趋势与区间推演")

        # 提取第 3 天 (index=2) 和 第 5 天 (index=4) 的模拟价格数据
        day3_prices = simulation_results[:, 2]
        day5_prices = simulation_results[:, 4]

        # 第 3 天统计
        day3_up_prob = (np.sum(day3_prices > last_price) / total_sims) * 100
        day3_mean = np.mean(day3_prices)
        day3_p10, day3_p90 = np.percentile(day3_prices, [10, 90])  # 80%置信区间
        day3_pct = ((day3_mean - last_price) / last_price) * 100

        # 第 5 天统计
        day5_up_prob = (np.sum(day5_prices > last_price) / total_sims) * 100
        day5_mean = np.mean(day5_prices)
        day5_p10, day5_p90 = np.percentile(day5_prices, [10, 90])  # 80%置信区间
        day5_pct = ((day5_mean - last_price) / last_price) * 100

        col_d3, col_d5 = st.columns(2)

        with col_d3:
            st.markdown("#### 📅 未来第 3 个交易日预测")
            st.metric("3日后上涨胜率", f"{day3_up_prob:.1f}%", delta=f"{day3_pct:+.2f}% (平均期望)")
            st.write(f"- **期望均价**：`¥{day3_mean:.2f}`")
            st.write(f"- **80% 置信运行区间**：`¥{day3_p10:.2f}` ~ `¥{day3_p90:.2f}`")

        with col_d5:
            st.markdown("#### 📅 未来第 5 个交易日预测")
            st.metric("5日后上涨胜率", f"{day5_up_prob:.1f}%", delta=f"{day5_pct:+.2f}% (平均期望)")
            st.write(f"- **期望均价**：`¥{day5_mean:.2f}`")
            st.write(f"- **80% 置信运行区间**：`¥{day5_p10:.2f}` ~ `¥{day5_p90:.2f}`")

        # 趋势形态解读
        if day5_pct > 1.5 and day5_up_prob > 55:
            st.success("📈 **中短期趋势预测**：未来 3~5 天呈现**震荡上行**态势，波段多头力量占优。")
        elif day5_pct < -1.5 and day5_up_prob < 45:
            st.error("📉 **中短期趋势预测**：未来 3~5 天呈现**回调向下**态势，防范阴跌风险。")
        else:
            st.info("↔️ **中短期趋势预测**：未来 3~5 天呈现**横盘震荡**态势，建议控制仓位高抛低吸。")

        # --- 6. 30 天蒙特卡洛模拟路径可视化 ---
        st.markdown("---")
        st.subheader("📊 未来 30 天蒙特卡洛模拟分布（10,000次抽样）")
        
        # 抽样 200 条路径进行可视化展示
        sample_paths = simulation_results[::50, :]
        st.line_chart(sample_paths.T)
