import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import numpy as np
import datetime
import re

st.set_page_config(page_title="Pro 級質押戰略戰情室", layout="wide")
RISK_FREE_RATE = 0.04

# ==========================================
# 1. 自動抓取市場數據函數 (Yahoo Finance)
# ==========================================
def fetch_asset_data(ticker):
    try:
        ticker = ticker.strip().upper()
        
        # 智能防呆：如果是純數字且沒有 .TW/.TWO，自動補上 .TW (台股)
        if re.match(r'^\d+[A-Z]*$', ticker) and '.TW' not in ticker and '.TWO' not in ticker:
            ticker = ticker + '.TW'
            
        # 設定回溯期間：過去 10 年，以涵蓋完整多空循環
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=10*365)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, f"找不到代號 {ticker} 的數據，請確認標的名稱。"
        
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]
            
        daily_returns = close_prices.pct_change().dropna()
        
        # 計算年化報酬率 (CAGR)
        total_return = close_prices.iloc[-1] / close_prices.iloc[0]
        years = (close_prices.index[-1] - close_prices.index[0]).days / 365.25
        ann_return = (total_return ** (1 / years)) - 1
        
        # 計算年化波動率 & 最大回撤
        ann_vol = daily_returns.std() * np.sqrt(252)
        rolling_max = close_prices.cummax()
        mdd = ((close_prices / rolling_max) - 1.0).min()
        
        # 計算歷年年度報酬率 (取最近 5 年畫圖用)
        # resample('YE') 會取每年的最後一個交易日
        annual_data = close_prices.resample('YE').last()
        annual_returns_series = annual_data.pct_change().dropna()
        annual_returns = {str(year.year): float(val) for year, val in annual_returns_series.items()}
        annual_returns = dict(list(annual_returns.items())[-5:]) # 只保留最近5年
        
        return {
            "ret": float(ann_return), 
            "beta": 1.0, # 簡化預設
            "vol": float(ann_vol), 
            "mdd": float(mdd),
            "annuals": annual_returns
        }, f"成功抓取 {ticker}！"
    except Exception as e:
        return None, f"抓取失敗: {str(e)}"

# ==========================================
# 2. 初始化 Session State (包含預設模擬數據)
# ==========================================
if 'asset_library' not in st.session_state:
    st.session_state.asset_library = {
        "無 (不配置)": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0, "annuals": {}},
        "QQQ (美股大盤)": {"ret": 0.15, "beta": 1.0, "vol": 0.18, "mdd": -0.33, 
                       "annuals": {"2019": 0.38, "2020": 0.47, "2021": 0.27, "2022": -0.33, "2023": 0.54}},
        "QLD (美股正2)": {"ret": 0.26, "beta": 2.0, "vol": 0.36, "mdd": -0.60, 
                       "annuals": {"2019": 0.80, "2020": 1.10, "2021": 0.50, "2022": -0.60, "2023": 1.20}},
        "00713 (台股高息)": {"ret": 0.10, "beta": 0.65, "vol": 0.12, "mdd": -0.15, 
                        "annuals": {"2019": 0.20, "2020": 0.10, "2021": 0.30, "2022": -0.07, "2023": 0.46}},
        "SGOV (短債)": {"ret": 0.045, "beta": 0.0, "vol": 0.02, "mdd": -0.01, 
                     "annuals": {"2019": 0.02, "2020": 0.01, "2021": 0.0, "2022": 0.01, "2023": 0.05}}
    }

if 'benchmark_strategies' not in st.session_state:
    st.session_state.benchmark_strategies = {
        "經典 CLEC 433": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SGOV (短債)": 30.0},
        "穩健 623 (防禦質押)": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SGOV (短債)": 30.0}
    }

if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}

# ==========================================
# 3. 核心計算引擎
# ==========================================
def calculate_metrics(weights_dict, margin_rate):
    total_weight = sum(weights_dict.values())
    debt_ratio = max(0, total_weight - 100.0)
    
    asset_ret, sys_beta, est_vol, est_mdd = 0.0, 0.0, 0.0, 0.0
    strategy_annuals = {}
    all_years = set()
    
    # 收集所有存在的年份
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            all_years.update(st.session_state.asset_library[name].get("annuals", {}).keys())
            
    # 計算各項加權指標
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            asset = st.session_state.asset_library[name]
            w_pct = weight / 100.0
            asset_ret += asset["ret"] * w_pct
            sys_beta += asset["beta"] * w_pct
            est_vol += asset["vol"] * w_pct
            est_mdd += asset.get("mdd", 0) * w_pct 
            
    # 計算每年的加權報酬率 (扣除質押利息)
    for year in sorted(all_years):
        yr_ret = 0
        for name, weight in weights_dict.items():
            if name in st.session_state.asset_library and weight > 0:
                yr_ret += st.session_state.asset_library[name].get("annuals", {}).get(year, 0) * (weight / 100.0)
        yr_ret -= (debt_ratio / 100.0) * margin_rate
        strategy_annuals[year] = yr_ret
            
    debt_cost = (debt_ratio / 100.0) * margin_rate
    net_return = asset_ret - debt_cost
    sharpe = (net_return - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    
    return {
        "總權重": total_weight, "實質負債": debt_ratio,
        "淨報酬率": net_return, "最大回撤": est_mdd,
        "波動率": est_vol, "夏普值": sharpe,
        "annuals": strategy_annuals
    }

# ==========================================
# 4. 側邊欄：智能抓取與資產庫
# ==========================================
st.sidebar.title("⚙️ 系統設定與資產庫")
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 智能抓取新增資產")
st.sidebar.caption("純數字代碼 (如 006208) 系統會自動轉為台股格式。")

with st.sidebar.form("auto_fetch_form"):
    ticker_input = st.text_input("輸入股票/ETF代號")
    fetch_btn = st.form_submit_button("自動抓取並新增")
    
    if fetch_btn and ticker_input:
        with st.spinner(f"正在分析資料..."):
            data, msg = fetch_asset_data(ticker_input)
            if data:
                # 為了顯示美觀，將輸入的代號作為顯示名稱
                display_name = f"{ticker_input.upper()} (自訂)"
                st.session_state.asset_library[display_name] = data
                st.success(msg)
            else:
                st.error(msg)

with st.sidebar.expander("查看當前資產庫參數"):
    lib_df = pd.DataFrame(st.session_state.asset_library).T
    if not lib_df.empty and "ret" in lib_df.columns:
        display_lib = lib_df[["ret", "vol", "mdd"]].copy()
        display_lib.columns = ["年化報酬", "波動率", "最大回撤"]
        st.dataframe(display_lib.style.format("{:.2%}"))

# ==========================================
# 5. 主畫面：策略建構器
# ==========================================
st.title("📊 頂級質押戰略戰情室")

st.subheader("🛠️ 建立新的自訂戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱", "我的新戰略")
    st.write("精確輸入資產權重 (%)，加總超過 100% 系統將自動視為質押借款：")
    
    cols = st.columns(5)
    selected_assets = {}
    asset_opts = list(st.session_state.asset_library.keys())
    
    for i in range(5):
        with cols[i]:
            asset = st.selectbox(f"部位 {i+1}", asset_opts, index=0, key=f"sel_{i}")
            weight = st.number_input(f"權重 (%)", 0.0, 300.0, 0.0, 1.0, key=f"w_{i}")
            if asset != "無 (不配置)" and weight > 0:
                selected_assets[asset] = selected_assets.get(asset, 0) + weight

    if st.form_submit_button("📥 儲存策略並加入比較表"):
        if selected_assets:
            st.session_state.custom_strategies[strat_name] = selected_assets
            st.success(f"策略已加入！")

if st.button("🗑️ 清空所有自訂策略"):
    st.session_state.custom_strategies = {}
    st.rerun()

st.markdown("---")

# ==========================================
# 6. 終極比較表與視覺化圖表
# ==========================================
st.subheader("🏆 戰略終極比較表")
comp_data = []
annual_chart_data = []

# 處理經典策略
for name, wts in st.session_state.benchmark_strategies.items():
    res = calculate_metrics(wts, margin_rate)
    res["策略名稱"] = name; res["類型"] = "經典對照"
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": name, "年份": year, "報酬率": ret, "類型": "經典對照"})

# 處理自訂策略
for name, wts in st.session_state.custom_strategies.items():
    res = calculate_metrics(wts, margin_rate)
    res["策略名稱"] = "🎯 " + name; res["類型"] = "自訂戰略"
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": "🎯 " + name, "年份": year, "報酬率": ret, "類型": "自訂戰略"})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    cols_order = ["類型", "策略名稱", "總權重", "實質負債", "淨報酬率", "最大回撤", "夏普值"]
    df_display = df_comp[cols_order].copy()
    
    df_display["總權重"] = df_display["總權重"].apply(lambda x: f"{x:.0f}%")
    df_display["實質負債"] = df_display["實質負債"].apply(lambda x: f"{x:.0f}%")
    df_display["淨報酬率"] = df_display["淨報酬率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ 壓力測試：最大回撤 (MDD)")
        df_chart_mdd = df_comp.sort_values(by="最大回撤", ascending=True)
        fig_mdd = px.bar(df_chart_mdd, x="最大回撤", y="策略名稱", color="類型", orientation='h', text="最大回撤",
            color_discrete_map={"經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_mdd.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_mdd.update_layout(xaxis_tickformat='.0%', xaxis_title="回撤幅度")
        st.plotly_chart(fig_mdd, use_container_width=True)
        
    with col2:
        st.subheader("📈 策略效率：夏普值 (Sharpe)")
        df_chart_sharpe = df_comp.sort_values(by="夏普值", ascending=True)
        fig_sharpe = px.bar(df_chart_sharpe, x="夏普值", y="策略名稱", color="類型", orientation='h', text="夏普值",
            color_discrete_map={"經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_sharpe.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_sharpe.update_layout(xaxis_title="數值越高越好")
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # 繪製年度報酬率對比圖
    st.markdown("---")
    st.subheader("📆 歷年報酬率壓力測試 (近 5 年)")
    if annual_chart_data:
        df_annual = pd.DataFrame(annual_chart_data)
        fig_annual = px.bar(df_annual, x="年份", y="報酬率", color="策略名稱", barmode="group")
        fig_annual.update_layout(yaxis_tickformat='.0%', yaxis_title="年度報酬率", xaxis_title="年份")
        st.plotly_chart(fig_annual, use_container_width=True)
