import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import numpy as np
import datetime
import re

st.set_page_config(page_title="Pro 級質押戰略戰情室", layout="wide")
RISK_FREE_RATE = 0.04
WITHDRAWAL_RATE = 0.03 # 預設買借死提領率 3%

# ==========================================
# 1. 自動抓取市場數據函數
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
            "annuals": annual_returns,
            "type": "Leverage" if "L" in ticker or "正2" in ticker else ("Defensive" if "債" in ticker or "SHY" in ticker else "Prototype")
        }, f"成功抓取 {ticker}！"
    except Exception as e:
        return None, f"抓取失敗: {str(e)}"

# ==========================================
# 2. 初始化真實數據
# ==========================================
@st.cache_data(ttl=86400)
def load_default_assets():
    lib = {
        "無 (不配置)": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0, "annuals": {}, "type": "None"},
        "現金": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0, "annuals": {}, "type": "Defensive"}
    }
    defaults = {
        "SPY": "SPY (標普大盤)",
        "QQQ": "QQQ (美股大盤)",
        "QLD": "QLD (美股正2)",
        "00713.TW": "00713 (台股高息)",
        "SHY": "SHY (1-3年短債)"
    }
    for ticker, display_name in defaults.items():
        data, _ = fetch_asset_data(ticker)
        if data:
            if "QLD" in ticker: data["beta"] = 2.0; data["type"] = "Leverage"
            if "00713" in ticker: data["beta"] = 0.65; data["type"] = "Prototype"
            if "SHY" in ticker: data["beta"] = 0.0; data["type"] = "Defensive"
            if "QQQ" in ticker or "SPY" in ticker: data["type"] = "Prototype"
            lib[display_name] = data
    return lib

if 'asset_library' not in st.session_state:
    st.session_state.asset_library = load_default_assets()

# 定義策略時，明確指定其「再平衡」與「負債」行為
if 'benchmark_strategies' not in st.session_state:
    st.session_state.benchmark_strategies = {
        "純抱 SPY": {"wts": {"SPY (標普大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無"},
        "純抱 QQQ": {"wts": {"QQQ (美股大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無"},
        "經典 CLEC 433 (買借死)": {"wts": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SHY (1-3年短債)": 30.0}, "rebal": "CLEC", "debt_mode": "買借死 (提領生活費)"},
        "穩健 623 (恆定增貸)": {"wts": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SHY (1-3年短債)": 30.0}, "rebal": "CLEC", "debt_mode": "恆定維持率 (增貸再投資)"}
    }

if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}

# ==========================================
# 3. 核心計算引擎 (解耦再平衡與負債模式)
# ==========================================
def calculate_metrics(strategy_config, margin_rate, target_margin_ratio=6.0):
    weights_dict = strategy_config["wts"]
    rebalance_type = strategy_config["rebal"]
    debt_mode = strategy_config["debt_mode"]
    
    initial_total_weight = sum(weights_dict.values())
    initial_debt_ratio = max(0, initial_total_weight - 100.0)
    
    sys_beta, est_vol, est_mdd = 0.0, 0.0, 0.0
    all_years = set()
    
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            asset = st.session_state.asset_library[name]
            all_years.update(asset.get("annuals", {}).keys())
            w_pct = weight / 100.0
            sys_beta += asset["beta"] * w_pct
            est_vol += asset["vol"] * w_pct
            est_mdd += asset.get("mdd", 0) * w_pct
            
    strategy_annuals = {}
    portfolio_equity = 1.0 
    current_debt_amount = initial_debt_ratio / 100.0
    current_asset_amounts = {name: (weight/100.0) for name, weight in weights_dict.items()}
    
    is_bankrupt = False

    for year in sorted(all_years):
        if is_bankrupt:
            strategy_annuals[year] = 0
            continue
            
        year_start_assets = sum(current_asset_amounts.values())
        
        # 1. 模擬該年度資產增長
        for name, amount in current_asset_amounts.items():
            if name in st.session_state.asset_library and amount > 0:
                asset_annuals = st.session_state.asset_library[name].get("annuals", {})
                ret = asset_annuals.get(year, 0)
                if ret == 0 and st.session_state.asset_library[name].get("type") == "Defensive":
                    ret = 0.02 # 填補早期債券空白
                current_asset_amounts[name] = amount * (1 + ret)
                
        # 2. 處理負債與利息
        interest_cost = current_debt_amount * margin_rate
        current_debt_amount += interest_cost
        
        # 3. 處理「買借死」的生活費提領 (假設每年提領年初淨值的 3%)
        withdrawal_amount = 0
        if debt_mode == "買借死 (提領生活費)":
            withdrawal_amount = portfolio_equity * WITHDRAWAL_RATE
            current_debt_amount += withdrawal_amount # 借款支付生活費，不買資產
            
        year_end_assets = sum(current_asset_amounts.values())
        portfolio_equity = year_end_assets - current_debt_amount
        
        if portfolio_equity <= 0:
            portfolio_equity = 0
            is_bankrupt = True
            strategy_annuals[year] = -1.0
            continue
            
        if year_start_assets > 0:
            # 淨報酬率計算需扣除提領額，以反映真實資產池成長
            net_year_return = (portfolio_equity - (year_start_assets - (current_debt_amount - interest_cost - withdrawal_amount))) / (year_start_assets - (current_debt_amount - interest_cost - withdrawal_amount))
        else:
            net_year_return = 0
        strategy_annuals[year] = net_year_return
        
        # 4. 執行再平衡與負債調整
        if rebalance_type == "CLEC":
            # 聰明再平衡 (上抽30%，下接2%)
            for name, amount in current_asset_amounts.items():
                asset_type = st.session_state.asset_library[name].get("type", "")
                ret = st.session_state.asset_library[name].get("annuals", {}).get(year, 0)
                if asset_type == "Leverage":
                    if ret > 0:
                        profit = (amount / (1+ret)) * ret
                        extract = profit * 0.3
                        current_asset_amounts[name] -= extract
                        for d_name in current_asset_amounts.keys():
                            if st.session_state.asset_library[d_name].get("type") == "Defensive":
                                current_asset_amounts[d_name] += extract
                                break
                    elif ret < 0:
                        for d_name in current_asset_amounts.keys():
                            if st.session_state.asset_library[d_name].get("type") == "Defensive":
                                rescue = current_asset_amounts[d_name] * 0.02
                                current_asset_amounts[d_name] -= rescue
                                current_asset_amounts[name] += rescue
                                break
                                
        elif rebalance_type == "傳統定時":
            # 恢復初始權重比例
            total_assets = sum(current_asset_amounts.values())
            for name, weight in weights_dict.items():
                current_asset_amounts[name] = total_assets * (weight/sum(weights_dict.values()))

        # 5. 恆定增貸邏輯 (只對 623 等有開啟增貸模式的策略執行)
        if debt_mode == "恆定維持率 (增貸再投資)":
            collateral_value = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library[n].get("type") == "Prototype"])
            if collateral_value > 0:
                current_margin_ratio = collateral_value / current_debt_amount if current_debt_amount > 0 else float('inf')
                if current_margin_ratio > target_margin_ratio:
                    target_debt = collateral_value / target_margin_ratio
                    new_loan = target_debt - current_debt_amount
                    if new_loan > 0:
                        current_debt_amount += new_loan
                        # 依照當前權重比例回灌資產池
                        total_assets_now = sum(current_asset_amounts.values())
                        if total_assets_now > 0:
                            for n in current_asset_amounts.keys():
                                current_asset_amounts[n] += new_loan * (current_asset_amounts[n] / total_assets_now)
                
    num_years = len(strategy_annuals)
    if num_years > 0 and not is_bankrupt:
        cagr = (portfolio_equity ** (1 / num_years)) - 1
        avg_annual_ret = sum(strategy_annuals.values()) / num_years
        sharpe = (avg_annual_ret - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    else:
        cagr = 0; sharpe = 0
        if is_bankrupt: portfolio_equity = 0
    
    is_pure_index = len([w for w in weights_dict.values() if w > 0]) == 1
    type_label = "純大盤對照" if is_pure_index else ("自訂戰略" if "🎯" in strategy_config.get("name", "") else "經典對照")
    
    return {
        "總權重": initial_total_weight, 
        "負債模式": debt_mode,
        "再平衡": rebalance_type,
        "系統 Beta": sys_beta, 
        "年化淨報酬率(CAGR)": cagr, 
        "20年終值倍數": portfolio_equity, 
        "最大回撤": est_mdd, 
        "夏普值": sharpe, 
        "annuals": strategy_annuals,
        "類型": type_label
    }

# ==========================================
# 4. 介面渲染：側邊欄
# ==========================================
st.sidebar.title("⚙️ 系統全局設定")
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 智能抓取新增資產")
with st.sidebar.form("auto_fetch_form"):
    ticker_input = st.text_input("輸入股票/ETF代號")
    fetch_btn = st.form_submit_button("抓取並新增")
    if fetch_btn and ticker_input:
        with st.spinner("分析中..."):
            data, msg = fetch_asset_data(ticker_input)
            if data:
                st.session_state.asset_library[f"{ticker_input.upper()} (自訂)"] = data
                st.success(msg)

# ==========================================
# 5. 主畫面：策略建構器 (加入解耦選項)
# ==========================================
st.title("📊 頂級質押戰略戰情室")

st.subheader("🛠️ 建立新的自訂戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱", "我的新戰略")
    
    # 策略模組選擇
    col_r, col_d = st.columns(2)
    with col_r:
        rebal_mode = st.selectbox("再平衡模組", ["CLEC", "傳統定時", "不執行"])
    with col_d:
        debt_mode = st.selectbox("負債運用模組", ["無", "恆定維持率 (增貸再投資)", "買借死 (提領生活費)"])
    
    st.write("精確輸入資產權重 (%)：")
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
            st.session_state.custom_strategies[strat_name] = {
                "name": "🎯 " + strat_name,
                "wts": selected_assets,
                "rebal": rebal_mode,
                "debt_mode": debt_mode
            }
            st.success("策略已加入！")

if st.button("🗑️ 清空自訂策略"):
    st.session_state.custom_strategies = {}
    st.rerun()

st.markdown("---")

# ==========================================
# 6. 終極比較表與視覺化圖表
# ==========================================
st.subheader("🏆 戰略終極比較表")
st.caption("✅ 已解耦：433 將執行生活費提領，而 623 將執行恆定增貸再投資。")

comp_data = []
annual_chart_data = []

for name, config in st.session_state.benchmark_strategies.items():
    res = calculate_metrics(config, margin_rate)
    res["策略名稱"] = name
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": name, "年份": year, "報酬率": ret, "類型": res["類型"]})

for name, config in st.session_state.custom_strategies.items():
    res = calculate_metrics(config, margin_rate)
    res["策略名稱"] = config["name"]
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": config["name"], "年份": year, "報酬率": ret, "類型": res["類型"]})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    cols_order = ["類型", "策略名稱", "總權重", "負債模式", "再平衡", "系統 Beta", "年化淨報酬率(CAGR)", "20年終值倍數", "最大回撤", "夏普值"]
    df_display = df_comp[cols_order].copy()
    
    df_display["總權重"] = df_display["總權重"].apply(lambda x: f"{x:.0f}%")
    df_display["系統 Beta"] = df_display["系統 Beta"].apply(lambda x: f"{x:.2f}")
    df_display["年化淨報酬率(CAGR)"] = df_display["年化淨報酬率(CAGR)"].apply(lambda x: f"{x*100:.2f}%")
    df_display["20年終值倍數"] = df_display["20年終值倍數"].apply(lambda x: f"{x:.1f}x")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💰 複利終值倍數")
        df_chart_multiple = df_comp.sort_values(by="20年終值倍數", ascending=True)
        fig_mult = px.bar(df_chart_multiple, x="20年終值倍數", y="策略名稱", color="類型", orientation='h', text="20年終值倍數",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_mult.update_traces(texttemplate='%{text:.1f}x', textposition='outside')
        st.plotly_chart(fig_mult, use_container_width=True)
        
    with col2:
        st.subheader("🛡️ 壓力測試：最大回撤")
        df_chart_mdd = df_comp.sort_values(by="最大回撤", ascending=True)
        fig_mdd = px.bar(df_chart_mdd, x="最大回撤", y="策略名稱", color="類型", orientation='h', text="最大回撤",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_mdd.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_mdd.update_layout(xaxis_tickformat='.0%')
        st.plotly_chart(fig_mdd, use_container_width=True)

    st.markdown("---")
    st.subheader("📆 歷年淨報酬率走勢 (20 年)")
    if annual_chart_data:
        df_annual = pd.DataFrame(annual_chart_data)
        df_annual = df_annual.sort_values(by="年份")
        fig_annual = px.bar(df_annual, x="年份", y="報酬率", color="策略名稱", barmode="group",
                            color_discrete_map={
                                "純抱 SPY (標普500)": "#c7c7c7",
                                "純抱 QQQ (納斯達克)": "#7f7f7f",
                                "經典 CLEC 433 (買借死)": "#1f77b4", 
                                "穩健 623 (恆定增貸)": "#ff7f0e",
                                "🎯 我的新戰略": "#E45756"
                            })
        fig_annual.update_layout(yaxis_tickformat='.0%', yaxis_title="年度淨報酬率", xaxis_title="年份", height=500)
        st.plotly_chart(fig_annual, use_container_width=True)
