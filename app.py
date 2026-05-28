import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import datetime
import re

st.set_page_config(page_title="頂級 CLEC 質押策略回測平台", layout="wide")
RISK_FREE_RATE = 0.04
WITHDRAWAL_RATE = 0.03

# ==========================================
# 1. 自動抓取市場數據函數 (保留完整日線資料)
# ==========================================
def fetch_asset_data(ticker, lookback_years=20):
    try:
        ticker = ticker.strip().upper()
        if re.match(r'^\d+[A-Z]*$', ticker) and '.TW' not in ticker and '.TWO' not in ticker:
            ticker = ticker + '.TW'
            
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=lookback_years*365)
        
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
        
        # 計算日線水下回撤
        rolling_max = close_prices.cummax()
        drawdown = (close_prices / rolling_max) - 1.0
        mdd = drawdown.min()
        
        # 轉換為以年為單位的收盤價字典 (供歷史年度對照使用)
        annual_data = close_prices.resample('YE').last()
        annual_returns = {str(year.year): float(val) for year, val in annual_data.pct_change().dropna().items()}
        inception_year = min([int(y) for y in annual_returns.keys()]) if annual_returns else datetime.date.today().year
        
        return {
            "ret": float(ann_return), "beta": 1.0, "vol": float(ann_vol), "mdd": float(mdd),
            "annuals": annual_returns, "inception_year": inception_year, "prices": close_prices,
            "type": "Leverage" if "L" in ticker or "正2" in ticker else ("Defensive" if "債" in ticker or "SHY" in ticker else "Prototype")
        }, f"成功抓取 {ticker}！"
    except Exception as e:
        return None, f"抓取失敗: {str(e)}"

# ==========================================
# 2. 初始化真實數據
# ==========================================
@st.cache_data(ttl=86400)
def load_default_assets(lookback=20):
    lib = {
        "無 (不配置)": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0, "annuals": {}, "inception_year": 0, "type": "None", "prices": pd.Series()},
        "現金": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0, "annuals": {}, "inception_year": 0, "type": "Defensive", "prices": pd.Series()}
    }
    defaults = {"SPY": "SPY (標普大盤)", "QQQ": "QQQ (美股大盤)", "QLD": "QLD (美股正2)", "00713.TW": "00713 (台股高息)", "SHY": "SHY (1-3年短債)"}
    for ticker, display_name in defaults.items():
        data, _ = fetch_asset_data(ticker, lookback)
        if data:
            if "QLD" in ticker: data["beta"] = 2.0; data["type"] = "Leverage"
            if "00713" in ticker: data["beta"] = 0.65; data["type"] = "Prototype"
            if "SHY" in ticker: data["beta"] = 0.0; data["type"] = "Defensive"
            lib[display_name] = data
    return lib

if 'lookback_years' not in st.session_state: st.session_state.lookback_years = 20
if 'asset_library' not in st.session_state: st.session_state.asset_library = load_default_assets(st.session_state.lookback_years)

if 'benchmark_strategies' not in st.session_state:
    st.session_state.benchmark_strategies = {
        "純抱 SPY": {"wts": {"SPY (標普大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無"},
        "純抱 QQQ": {"wts": {"QQQ (美股大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無"},
        "經典 CLEC 433 (買借死)": {"wts": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SHY (1-3年短債)": 30.0}, "rebal": "CLEC", "debt_mode": "買借死 (提領生活費)"},
        "穩健 623 (恆定增貸)": {"wts": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SHY (1-3年短債)": 30.0}, "rebal": "CLEC", "debt_mode": "恆定維持率 (增貸再投資)"}
    }
if 'custom_strategies' not in st.session_state: st.session_state.custom_strategies = {}

# ==========================================
# 3. 核心計算引擎 (修復期與卡瑪比率計算)
# ==========================================
def calculate_metrics(strategy_config, margin_rate, align_inception=True, target_margin_ratio=6.0):
    weights_dict = strategy_config["wts"]
    rebalance_type = strategy_config["rebal"]
    debt_mode = strategy_config["debt_mode"]
    
    initial_total_weight = sum(weights_dict.values())
    initial_debt_ratio = max(0, initial_total_weight - 100.0)
    
    sys_beta, est_vol = 0.0, 0.0
    all_years = set()
    max_inception_year = 0
    
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            asset = st.session_state.asset_library[name]
            all_years.update(asset.get("annuals", {}).keys())
            if asset.get("inception_year", 0) > max_inception_year and name not in ["無 (不配置)", "現金"]:
                max_inception_year = asset.get("inception_year", 0)
            sys_beta += asset["beta"] * (weight / 100.0)
            est_vol += asset["vol"] * (weight / 100.0)

    valid_years = sorted([y for y in all_years if int(y) >= max_inception_year]) if align_inception and max_inception_year > 0 else sorted(all_years)
            
    strategy_annuals = {}
    portfolio_equity = 1.0 
    current_debt_amount = initial_debt_ratio / 100.0
    current_asset_amounts = {name: (weight/100.0) for name, weight in weights_dict.items()}
    
    equity_curve = [] # 儲存資產成長曲線數據
    is_bankrupt = False

    for year in valid_years:
        if is_bankrupt:
            strategy_annuals[year] = 0; equity_curve.append({"年份": year, "淨值": 0.0})
            continue
            
        year_start_assets = sum(current_asset_amounts.values())
        
        # 1. 資產增長
        for name, amount in current_asset_amounts.items():
            if name in st.session_state.asset_library and amount > 0:
                ret = st.session_state.asset_library[name].get("annuals", {}).get(year, 0)
                if ret == 0 and st.session_state.asset_library[name].get("type") == "Defensive": ret = 0.02
                current_asset_amounts[name] = amount * (1 + ret)
                
        # 2. 利息與提領
        interest_cost = current_debt_amount * margin_rate
        current_debt_amount += interest_cost
        withdrawal_amount = portfolio_equity * WITHDRAWAL_RATE if debt_mode == "買借死 (提領生活費)" else 0
        current_debt_amount += withdrawal_amount
            
        year_end_assets = sum(current_asset_amounts.values())
        portfolio_equity = year_end_assets - current_debt_amount
        
        if portfolio_equity <= 0:
            portfolio_equity = 0; is_bankrupt = True; strategy_annuals[year] = -1.0; equity_curve.append({"年份": year, "淨值": 0.0})
            continue
            
        net_year_return = (portfolio_equity - (year_start_assets - (current_debt_amount - interest_cost - withdrawal_amount))) / (year_start_assets - (current_debt_amount - interest_cost - withdrawal_amount)) if year_start_assets > 0 else 0
        strategy_annuals[year] = net_year_return
        equity_curve.append({"年份": year, "淨值": portfolio_equity})
        
        # 4. 再平衡與恆定維持率
        if rebalance_type == "CLEC":
            for name, amount in current_asset_amounts.items():
                if st.session_state.asset_library[name].get("type") == "Leverage":
                    ret = st.session_state.asset_library[name].get("annuals", {}).get(year, 0)
                    if ret > 0:
                        extract = ((amount / (1+ret)) * ret) * 0.3
                        current_asset_amounts[name] -= extract
                        for d_name in current_asset_amounts.keys():
                            if st.session_state.asset_library[d_name].get("type") == "Defensive": current_asset_amounts[d_name] += extract; break
                    elif ret < 0:
                        for d_name in current_asset_amounts.keys():
                            if st.session_state.asset_library[d_name].get("type") == "Defensive":
                                rescue = current_asset_amounts[d_name] * 0.02
                                current_asset_amounts[d_name] -= rescue; current_asset_amounts[name] += rescue; break
        elif rebalance_type == "傳統定時":
            total_assets = sum(current_asset_amounts.values())
            for name, weight in weights_dict.items(): current_asset_amounts[name] = total_assets * (weight/sum(weights_dict.values()))

        if debt_mode == "恆定維持率 (增貸再投資)":
            collateral_value = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library[n].get("type") == "Prototype"])
            if collateral_value > 0:
                current_margin_ratio = collateral_value / current_debt_amount if current_debt_amount > 0 else float('inf')
                if current_margin_ratio > target_margin_ratio:
                    new_loan = (collateral_value / target_margin_ratio) - current_debt_amount
                    if new_loan > 0:
                        current_debt_amount += new_loan
                        total_assets_now = sum(current_asset_amounts.values())
                        if total_assets_now > 0:
                            for n in current_asset_amounts.keys(): current_asset_amounts[n] += new_loan * (current_asset_amounts[n] / total_assets_now)
                
    num_years = len(strategy_annuals)
    cagr = (portfolio_equity ** (1 / num_years)) - 1 if num_years > 0 and not is_bankrupt else 0
    avg_annual_ret = sum(strategy_annuals.values()) / num_years if num_years > 0 else 0
    sharpe = (avg_annual_ret - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    
    # 💥 計算水下回撤與最大修復天數 (日級模擬逼近)
    df_curve = pd.DataFrame(equity_curve)
    if not df_curve.empty and portfolio_equity > 0:
        df_curve["最高淨值"] = df_curve["淨值"].cummax()
        df_curve["水下回撤"] = (df_curve["淨值"] / df_curve["最高淨值"]) - 1.0
        real_mdd = df_curve["水下回撤"].min()
        
        # 估算最大修復時間
        max_recovery_years = 0
        current_drop_years = 0
        for idx, row in df_curve.iterrows():
            if row["水下回撤"] < 0: current_drop_years += 1
            else:
                if current_drop_years > max_recovery_years: max_recovery_years = current_drop_years
                current_drop_years = 0
        max_recovery_days = int(max_recovery_years * 365)
    else:
        real_mdd = est_mdd; max_recovery_days = 9999 if is_bankrupt else 0

    calmar = cagr / abs(real_mdd) if real_mdd != 0 else 0
    type_label = "純大盤對照" if len([w for w in weights_dict.values() if w > 0]) == 1 else ("自訂戰略" if "🎯" in strategy_config.get("name", "") else "經典對照")
    
    return {
        "總權重": initial_total_weight, "負債模式": debt_mode, "再平衡": rebalance_type, "系統 Beta": sys_beta, 
        "年化淨報酬率(CAGR)": cagr, f"{num_years}年終值倍數": portfolio_equity, "年化波動率": est_vol,
        "最大回撤": real_mdd, "夏普值": sharpe, "卡瑪比率": calmar, "最大修復天數": max_recovery_days,
        "annuals": strategy_annuals, "curve": equity_curve, "類型": type_label, "有效年數": num_years
    }

# ==========================================
# 4. 介面渲染：側邊欄
# ==========================================
st.sidebar.title("⚙️ 全局設定與智能防呆")
new_lookback = st.sidebar.slider("歷史資料抓取範圍 (年)", 5, 30, st.session_state.lookback_years, 1)
if new_lookback != st.session_state.lookback_years:
    st.session_state.lookback_years = new_lookback; st.cache_data.clear()
    st.session_state.asset_library = load_default_assets(new_lookback); st.rerun()

align_inception = st.sidebar.checkbox("強制作為公平比較 (對齊最晚掛牌日)", value=True)
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 智能抓取新增資產")
with st.sidebar.form("auto_fetch_form"):
    ticker_input = st.text_input("輸入股票/ETF代號 (防呆自動補.TW)")
    if st.form_submit_button("抓取並新增") and ticker_input:
        with st.spinner("真實市場連線中..."):
            data, msg = fetch_asset_data(ticker_input, st.session_state.lookback_years)
            if data: st.session_state.asset_library[f"{ticker_input.upper()} (自訂)"] = data; st.success(msg)

# ==========================================
# 5. 主畫面：策略建構器 & 策略管理
# ==========================================
st.title("📊 頂級 CLEC 質押策略回測戰情室")

st.subheader("🛠️ 建立自訂組合戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱", f"策略模式 {len(st.session_state.custom_strategies)+1}")
    col_r, col_d = st.columns(2)
    with col_r: rebal_mode = st.selectbox("再平衡模組", ["CLEC", "傳統定時", "不執行"])
    with col_d: debt_mode = st.selectbox("負債運用模組", ["無", "恆定維持率 (增貸再投資)", "買借死 (提領生活費)"])
    
    st.write("精確輸入資產權重 (%)：")
    cols = st.columns(5)
    selected_assets = {}
    asset_opts = list(st.session_state.asset_library.keys())
    
    for i in range(5):
        with cols[i]:
            asset = st.selectbox(f"部位 {i+1}", asset_opts, index=0, key=f"sel_{i}")
            weight = st.number_input(f"權重 (%)", 0.0, 300.0, 0.0, 1.0, key=f"w_{i}")
            if asset != "無 (不配置)" and weight > 0: selected_assets[asset] = selected_assets.get(asset, 0) + weight

    if st.form_submit_button("📥 儲存策略並加入比較表") and selected_assets:
        st.session_state.custom_strategies[strat_name] = {"name": "🎯 " + strat_name, "wts": selected_assets, "rebal": rebal_mode, "debt_mode": debt_mode}
        st.success(f"已成功加入「{strat_name}」！")

# 💥 實裝：單一刪除與全刪管理區塊
if st.session_state.custom_strategies:
    st.markdown("#### 🗑️ 管理已儲存的自訂策略")
    col_del1, col_del2, col_del3 = st.columns([2, 1, 1])
    with col_del1: del_target = st.selectbox("選擇要刪除的策略", list(st.session_state.custom_strategies.keys()), label_visibility="collapsed")
    with col_del2: 
        if st.button("刪除單一策略", use_container_width=True): del st.session_state.custom_strategies[del_target]; st.rerun()
    with col_del3:
        if st.button("⚠️ 全數清空", type="primary", use_container_width=True): st.session_state.custom_strategies = {}; st.rerun()

st.markdown("---")

# ==========================================
# 6. 終極比較表與 4 大指標渲染
# ==========================================
st.subheader("🏆 戰略終極比較表")
comp_data = []
annual_chart_data = []
curve_chart_data = []

for name, config in st.session_state.benchmark_strategies.items():
    res = calculate_metrics(config, margin_rate, align_inception)
    res["策略名稱"] = name; comp_data.append(res)
    for year, ret in res["annuals"].items(): annual_chart_data.append({"策略名稱": name, "年份": year, "報酬率": ret, "類型": res["類型"]})
    for pt in res["curve"]: curve_chart_data.append({"策略名稱": name, "年份": pt["年份"], "淨值": pt["淨值"]})

for name, config in st.session_state.custom_strategies.items():
    res = calculate_metrics(config, margin_rate, align_inception)
    res["策略名稱"] = config["name"]; comp_data.append(res)
    for year, ret in res["annuals"].items(): annual_chart_data.append({"策略名稱": config["name"], "年份": year, "報酬率": ret, "類型": res["類型"]})
    for pt in res["curve"]: curve_chart_data.append({"策略名稱": config["name"], "年份": pt["年份"], "淨值": pt["淨值"]})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    terminal_col = [col for col in df_comp.columns if "終值倍數" in col][0]
    
    # 加入法人的兩大新指標：卡瑪比率、最大修復天數
    cols_order = ["類型", "策略名稱", "總權重", "負債模式", "再平衡", "系統 Beta", "年化淨報酬率(CAGR)", terminal_col, "年化波動率", "最大回撤", "夏普值", "卡瑪比率", "最大修復天數"]
    df_display = df_comp[cols_order].copy()
    
    df_display["總權重"] = df_display["總權重"].apply(lambda x: f"{x:.0f}%")
    df_display["系統 Beta"] = df_display["系統 Beta"].apply(lambda x: f"{x:.2f}")
    df_display["年化淨報酬率(CAGR)"] = df_display["年化淨報酬率(CAGR)"].apply(lambda x: f"{x*100:.2f}%")
    df_display[terminal_col] = df_display[terminal_col].apply(lambda x: f"{x:.1f}x")
    df_display["年化波動率"] = df_display["年化波動率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    df_display["卡瑪比率"] = df_display["卡瑪比率"].apply(lambda x: f"{x:.3f}")
    df_display["最大修復天數"] = df_display["最大修復天數"].apply(lambda x: f"{x:,} 天" if x < 9999 else "已斷頭破產")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 💥 實裝功能 1：資產成長折線圖 (Portfolio Growth Line Chart)
    st.markdown("---")
    st.subheader("📈 20年資產累積成長複利曲線 (Equity Curve)")
    if curve_chart_data:
        df_curves = pd.DataFrame(curve_chart_data)
        fig_curves = px.line(df_curves, x="年份", y="淨值", color="策略名稱", log_y=True) # 使用對數坐標軸更清晰
        fig_curves.update_layout(yaxis_title="資產增長倍數 (對數軸)", xaxis_title="年份", height=450)
        st.plotly_chart(fig_curves, use_container_width=True)

    # 💥 實裝功能 2：水下回撤折線圖 (Drawdown Profile)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("🛡️ 策略卡瑪比率排行 (每單位回撤帶來的利潤)")
        df_chart_calmar = df_comp.sort_values(by="卡瑪比率", ascending=True)
        fig_calmar = px.bar(df_chart_calmar, x="卡瑪比率", y="策略名稱", color="類型", orientation='h', text="卡瑪比率",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_calmar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig_calmar, use_container_width=True)
        
    with col2:
        st.subheader("⏳ 最長套牢修復期排行 (越短越好)")
        df_chart_rec = df_comp.sort_values(by="最大修復天數", ascending=False)
        fig_rec = px.bar(df_chart_rec, x="最大修復天數", y="策略名稱", color="類型", orientation='h', text="最大修復天數",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_rec.update_traces(texttemplate='%{text:,} 天', textposition='outside')
        st.plotly_chart(fig_rec, use_container_width=True)

    st.markdown("---")
    st.subheader("📆 歷年淨報酬率大亂鬥")
    if annual_chart_data:
        df_annual = pd.DataFrame(annual_chart_data).sort_values(by="年份")
        fig_annual = px.bar(df_annual, x="年份", y="報酬率", color="策略名稱", barmode="group")
        fig_annual.update_layout(yaxis_tickformat='.0%', yaxis_title="年度淨報酬率", xaxis_title="年份", height=450)
        st.plotly_chart(fig_annual, use_container_width=True)
