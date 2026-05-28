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
            "annuals": annual_returns,
            "type": "Leverage" if "L" in ticker or "正2" in ticker else ("Defensive" if "債" in ticker or "SHY" in ticker else "Prototype")
        }, f"成功抓取 {ticker}！"
    except Exception as e:
        return None, f"抓取失敗: {str(e)}"

# ==========================================
# 2. 背景自動初始化真實數據
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

if 'benchmark_strategies' not in st.session_state:
    st.session_state.benchmark_strategies = {
        "純抱 SPY (標普500)": {"SPY (標普大盤)": 100.0},
        "純抱 QQQ (納斯達克)": {"QQQ (美股大盤)": 100.0},
        "經典 CLEC 433 (無借貸)": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SHY (1-3年短債)": 30.0},
        "穩健 623 (恆定600%)": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SHY (1-3年短債)": 30.0}
    }

if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}

# ==========================================
# 3. 核心計算引擎 (加入真實恆定維持率邏輯與防呆)
# ==========================================
def calculate_metrics(weights_dict, margin_rate, rebalance_type, target_margin_ratio=6.0):
    initial_total_weight = sum(weights_dict.values())
    # 初始負債 (超過100%的部分視為借款)
    initial_debt = max(0, initial_total_weight - 100.0)
    
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
    
    # 初始本金 1 單位，轉換為絕對金額進行模擬
    portfolio_equity = 1.0 
    current_debt_amount = initial_debt / 100.0
    # 初始各資產的絕對金額
    current_asset_amounts = {name: (weight/100.0) for name, weight in weights_dict.items()}
    
    is_bankrupt = False

    for year in sorted(all_years):
        if is_bankrupt:
            strategy_annuals[year] = 0
            continue
            
        year_gross_return = 0
        year_start_assets = sum(current_asset_amounts.values())
        
        # 1. 模擬該年度資產增減
        for name, amount in current_asset_amounts.items():
            if name in st.session_state.asset_library and amount > 0:
                asset_annuals = st.session_state.asset_library[name].get("annuals", {})
                ret = asset_annuals.get(year, 0) if year in asset_annuals else 0
                current_asset_amounts[name] = amount * (1 + ret)
                
        year_end_assets = sum(current_asset_amounts.values())
        
        # 2. 扣除借貸利息
        interest_cost = current_debt_amount * margin_rate
        current_debt_amount += interest_cost # 利息滾入負債
        
        # 3. 結算本年度淨值 (Equity)
        portfolio_equity = year_end_assets - current_debt_amount
        
        # 💥 破產保護：如果淨值歸零或為負，宣告破產，避免產生虛數
        if portfolio_equity <= 0:
            portfolio_equity = 0
            is_bankrupt = True
            strategy_annuals[year] = -1.0 # 該年回報 -100%
            continue
            
        # 計算年度淨報酬率
        if year_start_assets > 0:
            net_year_return = (portfolio_equity - (year_start_assets - (current_debt_amount - interest_cost))) / (year_start_assets - (current_debt_amount - interest_cost))
        else:
            net_year_return = 0
        strategy_annuals[year] = net_year_return
        
        # 判斷是否為純大盤策略 (不執行複雜再平衡)
        is_pure_index = len([w for w in weights_dict.values() if w > 0]) == 1
        
        if not is_pure_index:
            if rebalance_type == "傳統定時再平衡":
                # 強制將權重調回初始設定
                for name, weight in weights_dict.items():
                    current_asset_amounts[name] = portfolio_equity * (weight/100.0)
                current_debt_amount = portfolio_equity * (initial_debt/100.0)
                
            elif rebalance_type == "CLEC 聰明再平衡 (含恆定增貸)":
                # --- A. 聰明再平衡邏輯 ---
                for name, amount in current_asset_amounts.items():
                    asset_type = st.session_state.asset_library[name].get("type", "")
                    ret = st.session_state.asset_library[name].get("annuals", {}).get(year, 0)
                    
                    if asset_type == "Leverage":
                        if ret > 0:
                            # 抽出獲利的 30% 給防守部位
                            profit = (amount / (1+ret)) * ret
                            profit_to_extract = profit * 0.3
                            current_asset_amounts[name] -= profit_to_extract
                            
                            for d_name in current_asset_amounts.keys():
                                if st.session_state.asset_library[d_name].get("type") == "Defensive":
                                    current_asset_amounts[d_name] += profit_to_extract
                                    break
                        elif ret < 0:
                            # 用防守部位的 2% 救援
                            for d_name in current_asset_amounts.keys():
                                if st.session_state.asset_library[d_name].get("type") == "Defensive":
                                    rescue_amount = current_asset_amounts[d_name] * 0.02
                                    current_asset_amounts[d_name] -= rescue_amount
                                    current_asset_amounts[name] += rescue_amount
                                    break
                                    
                # --- B. 恆定維持率 (動態增貸) 邏輯 ---
                if initial_debt > 0: # 代表這是一個有質押設定的策略 (如 623)
                    # 尋找原型資產作為擔保品 (這裡簡化，將所有 Prototype 視為可質押)
                    collateral_value = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library[n].get("type") == "Prototype"])
                    
                    if collateral_value > 0:
                        # 計算維持率 (擔保品市值 / 負債)
                        # 如果負債為 0，視為維持率無限大
                        current_margin_ratio = collateral_value / current_debt_amount if current_debt_amount > 0 else float('inf')
                        
                        # 如果當前維持率高於目標 (例如 > 600%)，代表資產膨脹，可以增貸
                        if current_margin_ratio > target_margin_ratio:
                            # 計算目標負債 = 擔保品市值 / 目標維持率
                            target_debt = collateral_value / target_margin_ratio
                            new_loan_amount = target_debt - current_debt_amount
                            
                            # 實際增貸
                            current_debt_amount += new_loan_amount
                            
                            # 將增貸的錢回灌到資產池中 (依照原本 623 的比例分配)
                            # 這裡簡化為按比例均分給所有大於 0 的資產
                            active_assets_count = len([n for n, amt in current_asset_amounts.items() if amt > 0])
                            if active_assets_count > 0:
                                add_per_asset = new_loan_amount / active_assets_count
                                for n in current_asset_amounts.keys():
                                    if current_asset_amounts[n] > 0:
                                        current_asset_amounts[n] += add_per_asset
                
    num_years = len(strategy_annuals)
    
    if num_years > 0 and not is_bankrupt:
        cagr = (portfolio_equity ** (1 / num_years)) - 1
        avg_annual_ret = sum(strategy_annuals.values()) / num_years
        sharpe = (avg_annual_ret - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    else:
        cagr = 0; sharpe = 0
        if is_bankrupt: portfolio_equity = 0
    
    is_pure_index = len([w for w in weights_dict.values() if w > 0]) == 1
    type_label = "純大盤對照" if is_pure_index else "經典對照"
    
    return {
        "總權重": initial_total_weight, "實質負債": initial_debt,
        "系統 Beta": sys_beta, "年化淨報酬率(CAGR)": cagr, 
        "20年終值倍數": portfolio_equity, "年化波動率": est_vol,
        "最大回撤": est_mdd, "夏普值": sharpe, "annuals": strategy_annuals,
        "類型": type_label
    }

# ==========================================
# 4. 介面渲染：側邊欄
# ==========================================
st.sidebar.title("⚙️ 系統設定與資產庫")
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 再平衡機制選擇")
rebalance_choice = st.sidebar.radio(
    "選擇年度再平衡策略", 
    ("傳統定時再平衡", "CLEC 聰明再平衡 (含恆定增貸)", "不執行再平衡"),
    index=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 智能抓取新增資產")

with st.sidebar.form("auto_fetch_form"):
    ticker_input = st.text_input("輸入股票/ETF代號")
    fetch_btn = st.form_submit_button("自動抓取並新增")
    
    if fetch_btn and ticker_input:
        with st.spinner(f"正在分析 {ticker_input}..."):
            data, msg = fetch_asset_data(ticker_input)
            if data:
                display_name = f"{ticker_input.upper()} (自訂)"
                st.session_state.asset_library[display_name] = data
                st.success(msg)
            else:
                st.error(msg)

# ==========================================
# 5. 主畫面：策略建構器
# ==========================================
st.title("📊 頂級質押戰略戰情室")

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
            st.success("策略已加入！")

if st.button("🗑️ 清空所有自訂策略"):
    st.session_state.custom_strategies = {}
    st.rerun()

st.markdown("---")

# ==========================================
# 6. 終極比較表與視覺化圖表
# ==========================================
st.subheader(f"🏆 戰略終極比較表 (目前模式: {rebalance_choice})")

comp_data = []
annual_chart_data = []

for name, wts in st.session_state.benchmark_strategies.items():
    res = calculate_metrics(wts, margin_rate, rebalance_choice)
    res["策略名稱"] = name; res["類型"] = res["類型"]
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": name, "年份": year, "報酬率": ret, "類型": res["類型"]})

for name, wts in st.session_state.custom_strategies.items():
    res = calculate_metrics(wts, margin_rate, rebalance_choice)
    res["策略名稱"] = "🎯 " + name; res["類型"] = "自訂戰略"
    comp_data.append(res)
    for year, ret in res["annuals"].items():
        annual_chart_data.append({"策略名稱": "🎯 " + name, "年份": year, "報酬率": ret, "類型": "自訂戰略"})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    cols_order = ["類型", "策略名稱", "總權重", "實質負債", "系統 Beta", "年化淨報酬率(CAGR)", "20年終值倍數", "年化波動率", "最大回撤", "夏普值"]
    df_display = df_comp[cols_order].copy()
    
    df_display["總權重"] = df_display["總權重"].apply(lambda x: f"{x:.0f}%")
    df_display["實質負債"] = df_display["實質負債"].apply(lambda x: f"{x:.0f}%")
    df_display["系統 Beta"] = df_display["系統 Beta"].apply(lambda x: f"{x:.2f}")
    df_display["年化淨報酬率(CAGR)"] = df_display["年化淨報酬率(CAGR)"].apply(lambda x: f"{x*100:.2f}%")
    df_display["20年終值倍數"] = df_display["20年終值倍數"].apply(lambda x: f"{x:.1f}x")
    df_display["年化波動率"] = df_display["年化波動率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💰 複利終值倍數 (越高越好)")
        df_chart_multiple = df_comp.sort_values(by="20年終值倍數", ascending=True)
        fig_mult = px.bar(df_chart_multiple, x="20年終值倍數", y="策略名稱", color="類型", orientation='h', text="20年終值倍數",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_mult.update_traces(texttemplate='%{text:.1f}x', textposition='outside')
        st.plotly_chart(fig_mult, use_container_width=True)
        
    with col2:
        st.subheader("🛡️ 壓力測試：最大回撤 (MDD)")
        df_chart_mdd = df_comp.sort_values(by="最大回撤", ascending=True)
        fig_mdd = px.bar(df_chart_mdd, x="最大回撤", y="策略名稱", color="類型", orientation='h', text="最大回撤",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_mdd.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_mdd.update_layout(xaxis_tickformat='.0%')
        st.plotly_chart(fig_mdd, use_container_width=True)

    st.markdown("---")
    st.subheader("📆 歷年淨報酬率壓力測試 (最長 20 年)")
    if annual_chart_data:
        df_annual = pd.DataFrame(annual_chart_data)
        df_annual = df_annual.sort_values(by="年份")
        fig_annual = px.bar(df_annual, x="年份", y="報酬率", color="策略名稱", barmode="group",
                            color_discrete_map={
                                "純抱 SPY (標普500)": "#c7c7c7",
                                "純抱 QQQ (納斯達克)": "#7f7f7f",
                                "經典 CLEC 433 (無借貸)": "#1f77b4", 
                                "穩健 623 (恆定600%)": "#ff7f0e",
                                "🎯 我的新戰略": "#E45756"
                            })
        fig_annual.update_layout(yaxis_tickformat='.0%', yaxis_title="年度淨報酬率", xaxis_title="年份", height=500)
        st.plotly_chart(fig_annual, use_container_width=True)
