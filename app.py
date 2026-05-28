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
# 1. 自動抓取市場數據函數 (20年期)
# ==========================================
def fetch_asset_data(ticker):
    try:
        ticker = ticker.strip().upper()
        
        if re.match(r'^\d+[A-Z]*$', ticker) and '.TW' not in ticker and '.TWO' not in ticker:
            ticker = ticker + '.TW'
            
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=20*365)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, f"找不到 {ticker} 的數據。"
        
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]
            
        daily_returns = close_prices.pct_change().dropna()
        
        total_return = close_prices.iloc[-1] / close_prices.iloc[0]
        years = (close_prices.index[-1] - close_prices.index[0]).days / 365.25
        ann_return = (total_return ** (1 / years)) - 1
        
        ann_vol = daily_returns.std() * np.sqrt(252)
        rolling_max = close_prices.cummax()
        mdd = ((close_prices / rolling_max) - 1.0).min()
        
        annual_data = close_prices.resample('YE').last()
        annual_returns_series = annual_data.pct_change().dropna()
        annual_returns = {str(year.year): float(val) for year, val in annual_returns_series.items()}
        
        return {
            "ret": float(ann_return), 
            "beta": 1.0, 
            "vol": float(ann_vol), 
            "mdd": float(mdd),
            "annuals": annual_returns
        }, f"成功抓取 {ticker}！"
    except Exception as e:
        return None, f"抓取失敗: {str(e)}"

# ==========================================
# 2. 背景自動初始化真實數據
# ==========================================
@st.cache_data(ttl=86400)
def load_default_assets():
    lib = {
        "無 (不配置)": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0, "annuals": {}},
        "現金": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0, "annuals": {}}
    }
    defaults = {
        "QQQ": "QQQ (美股大盤)",
        "QLD": "QLD (美股正2)",
        "00713.TW": "00713 (台股高息)",
        "SGOV": "SGOV (短債)"
    }
    
    for ticker, display_name in defaults.items():
        data, _ = fetch_asset_data(ticker)
        if data:
            if "QLD" in ticker: data["beta"] = 2.0
            if "00713" in ticker: data["beta"] = 0.65
            if "SGOV" in ticker: data["beta"] = 0.0
            lib[display_name] = data
    return lib

if 'asset_library' not in st.session_state:
    st.session_state.asset_library = load_default_assets()

if 'benchmark_strategies' not in st.session_state:
    st.session_state.benchmark_strategies = {
        "經典 CLEC 433": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SGOV (短債)": 30.0},
        "穩健 623 (防禦質押)": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SGOV (短債)": 30.0}
    }

if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}

# ==========================================
# 3. 核心計算引擎 (修復欄位與命名)
# ==========================================
def calculate_metrics(weights_dict, margin_rate):
    total_weight = sum(weights_dict.values())
    debt_ratio = max(0, total_weight - 100.0)
    
    asset_ret, sys_beta, est_vol, est_mdd = 0.0, 0.0, 0.0, 0.0
    strategy_annuals = {}
    all_years = set()
    
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            all_years.update(st.session_state.asset_library[name].get("annuals", {}).keys())
            
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            asset = st.session_state.asset_library[name]
            w_pct = weight / 100.0
            asset_ret += asset["ret"] * w_pct
            sys_beta += asset["beta"] * w_pct
            est_vol += asset["vol"] * w_pct
            est_mdd += asset.get("mdd", 0) * w_pct 
            
    for year in sorted(all_years):
        yr_ret = 0
        valid_assets = 0
        for name, weight in weights_dict.items():
            if name in st.session_state.asset_library and weight > 0:
                asset_annuals = st.session_state.asset_library[name].get("annuals", {})
                if year in asset_annuals:
                    yr_ret += asset_annuals[year] * (weight / 100.0)
                    valid_assets += 1
        
        if valid_assets > 0:
            yr_ret -= (debt_ratio / 100.0) * margin_rate
            strategy_annuals[year] = yr_ret
            
    debt_cost = (debt_ratio / 100.0) * margin_rate
    net_return = asset_ret - debt_cost
    sharpe = (net_return - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    
    return {
        "總權重": total_weight, "實質負債": debt_ratio,
        "系統 Beta": sys_beta,
        "年化淨報酬率": net_return, 
        "年化波動率": est_vol,
        "最大回撤": est_mdd,
        "夏普值": sharpe,
        "annuals": strategy_annuals
    }

# ==========================================
# 4. 介面渲染：側邊欄
# ==========================================
st.sidebar.title("⚙️ 系統設定與資產庫")
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 智能抓取新增資產")

with st.sidebar.form("auto_fetch_form"):
    ticker_input = st.text_input("輸入股票/ETF代號 (支援 20 年回測)")
    fetch_btn = st.form_submit_button("自動抓取並新增")
    
    if fetch_btn and ticker_input:
        with st.spinner(f"正在分析 {ticker_input} 歷史數據..."):
            data, msg = fetch_asset_data(ticker_input)
            if data:
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
st.title("📊 頂級質押戰略戰情室 (20年回測版)")

st.subheader("🛠️ 建立新的自訂戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱", "我的新戰略")
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
st.caption("💡 備註：此表為「靜態起始權重」預估。真實的「動態恆定維持率」會隨資產上漲持續擴張債務與複利，實質長期利潤將高於此靜態預估值。")

comp_data = []
annual_chart_data = []

for name, wts in st.session_state.benchmark_strategies.items():
    res = calculate_metrics(wts, margin_rate)
    res["策略名稱"] = name; res["類型"] = "經典對照"
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": name, "年份": year, "報酬率": ret, "類型": "經典對照"})

for name, wts in st.session_state.custom_strategies.items():
    res = calculate_metrics(wts, margin_rate)
    res["策略名稱"] = "🎯 " + name; res["類型"] = "自訂戰略"
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": "🎯 " + name, "年份": year, "報酬率": ret, "類型": "自訂戰略"})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    # 補回系統 Beta 與 年化波動率
    cols_order = ["類型", "策略名稱", "總權重", "實質負債", "系統 Beta", "年化淨報酬率", "年化波動率", "最大回撤", "夏普值"]
    df_display = df_comp[cols_order].copy()
    
    df_display["總權重"] = df_display["總權重"].apply(lambda x: f"{x:.0f}%")
    df_display["實質負債"] = df_display["實質負債"].apply(lambda x: f"{x:.0f}%")
    df_display["系統 Beta"] = df_display["系統 Beta"].apply(lambda x: f"{x:.2f}")
    df_display["年化淨報酬率"] = df_display["年化淨報酬率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["年化波動率"] = df_display["年化波動率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📆 歷年報酬率壓力測試 (最長 20 年)")
    if annual_chart_data:
        df_annual = pd.DataFrame(annual_chart_data)
        df_annual = df_annual.sort_values(by="年份")
        fig_annual = px.bar(df_annual, x="年份", y="報酬率", color="策略名稱", barmode="group")
        fig_annual.update_layout(yaxis_tickformat='.0%', yaxis_title="年度淨報酬率", xaxis_title="年份", height=500)
        st.plotly_chart(fig_annual, use_container_width=True)
