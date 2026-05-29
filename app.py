import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import datetime
import re

st.set_page_config(page_title="CLEC 質押策略績效戰情室", layout="wide")
RISK_FREE_RATE = 0.04

# ==========================================
# 1. 自動抓取市場數據函數 
# ==========================================
@st.cache_data(ttl=86400)
def fetch_asset_base_data(ticker, asset_type):
    try:
        ticker = ticker.strip().upper()
        if re.match(r'^\d+[A-Z]*$', ticker) and '.TW' not in ticker and '.TWO' not in ticker:
            ticker = ticker + '.TW'
            
        data = yf.download(ticker, start="1999-01-01", end=datetime.date.today(), progress=False)
        if data.empty:
            return None, f"找不到 {ticker} 的數據。"
        
        prices = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
        prices = prices.dropna()
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
            
        inception_date = prices.index[0].date()
        
        return {
            "prices": prices,
            "inception_date": inception_date,
            "type": asset_type
        }, f"成功抓取 {ticker}！掛牌日: {inception_date}"
    except Exception as e:
        return None, f"抓取失敗: {str(e)}"

# ==========================================
# 2. 初始化預設資產與策略
# ==========================================
def load_default_assets():
    lib = {
        "無 (不配置)": {"prices": pd.Series(dtype=float), "inception_date": datetime.date(1990, 1, 1), "beta": 0.0, "type": "None"},
        "現金": {"prices": pd.Series(dtype=float), "inception_date": datetime.date(1990, 1, 1), "beta": 0.0, "type": "Defensive"}
    }
    defaults = [
        ("QQQ", "QQQ (美股大盤)", "Prototype", 1.0),
        ("SPY", "SPY (標普大盤)", "Prototype", 1.0),
        ("QLD", "QLD (美股正2)", "Leverage", 2.0),
        ("0050.TW", "0050 (台股大盤)", "Prototype", 1.0),
        ("00662.TW", "00662 (NAS原型)", "Prototype", 1.0),
        ("00713.TW", "00713 (台股高息)", "Prototype", 0.65),
        ("00631L.TW", "00631L (台股正2)", "Leverage", 2.0),
        ("00670L.TW", "00670L (美股正2)", "Leverage", 2.0),
        ("SGOV", "SGOV (美股超短債)", "Defensive", 0.0),
        ("SHY", "SHY (1-3年短債)", "Defensive", 0.0),
        ("00865B.TW", "00865B (台股短債)", "Defensive", 0.0),
        ("00859B.TW", "00859B (台股投資級債)", "Defensive", 0.0)
    ]
    for ticker, display_name, a_type, beta in defaults:
        data, _ = fetch_asset_base_data(ticker, a_type)
        if data:
            data["beta"] = beta
            lib[display_name] = data
    return lib

if 'asset_library' not in st.session_state:
    st.session_state.asset_library = load_default_assets()

st.session_state.benchmark_strategies = {
    "純抱 SPY": {"wts": {"SPY (標普大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無", "target_margin": 6.0},
    "純抱 QQQ": {"wts": {"QQQ (美股大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無", "target_margin": 6.0},
    "經典 CLEC 433 (買借死)": {"wts": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SGOV (美股超短債)": 30.0}, "rebal": "CLEC", "debt_mode": "買借死 (提領生活費)", "target_margin": 6.0},
    "穩健 623 (恆定增貸)": {"wts": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SGOV (美股超短債)": 30.0}, "rebal": "CLEC", "debt_mode": "恆定維持率 (增貸再投資)", "target_margin": 6.0},
    "防禦 812 (恆定增貸)": {"wts": {"QQQ (美股大盤)": 80.0, "QLD (美股正2)": 10.0, "SGOV (美股超短債)": 20.0}, "rebal": "CLEC", "debt_mode": "恆定維持率 (增貸再投資)", "target_margin": 10.0}
}

if 'custom_strategies' not in st.session_state: st.session_state.custom_strategies = {}

# ==========================================
# 3. 智慧時間軸控制器 
# ==========================================
st.sidebar.markdown("### 歷史回測與分析引擎")

active_assets = set()
for strats in [st.session_state.benchmark_strategies, st.session_state.custom_strategies]:
    for config in strats.values():
        for asset_name, weight in config["wts"].items():
            if weight > 0 and asset_name in st.session_state.asset_library:
                active_assets.add(asset_name)

max_inception_date = datetime.date(1999, 1, 1)
for asset in active_assets:
    asset_data = st.session_state.asset_library.get(asset, {})
    inc_date = asset_data.get("inception_date", datetime.date(1999, 1, 1))
    if inc_date > max_inception_date and asset not in ["無 (不配置)", "現金"]: 
        max_inception_date = inc_date

enable_synthetic = st.sidebar.checkbox("🚀 啟用智能合成代理引擎 (解鎖發行日前回測)", value=False)

if 'start_date' not in st.session_state:
    st.session_state.start_date = max_inception_date
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.date.today()

min_historical_limit = datetime.date(1999, 1, 4)
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("回測起始日", value=st.session_state.start_date, min_value=min_historical_limit, max_value=datetime.date.today())
with col_d2:
    end_date = st.date_input("回測結束日", value=st.session_state.end_date, min_value=min_historical_limit, max_value=datetime.date.today())

st.session_state.start_date = start_date
st.session_state.end_date = end_date

min_allowed_date = min_historical_limit if enable_synthetic else max_inception_date

if start_date < min_allowed_date:
    if not enable_synthetic:
        st.sidebar.error(f"⚠️ **未啟用合成引擎**\n\n您選擇了 {start_date}，但目前配置中部位的最晚掛牌日為 {max_inception_date}。\n\n👉 **請勾選上方「🚀 啟用智能合成代理引擎」** 進行歷史數據回填，即可自由解鎖往前的年份！")
    else:
        st.sidebar.error(f"⚠️ 系統日線數據最早僅支援至 1999 年 1 月 4 日。")
    st.stop()

st.sidebar.markdown("---")
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("💰 買借死提領現金流設定")
init_capital = st.sidebar.number_input("初始試算本金 (元)", min_value=100000, value=10000000, step=1000000)
withdraw_mode = st.sidebar.selectbox("提領生活費模式", ["固定金額 (元)", "總資產百分比 (%)"])
if withdraw_mode == "總資產百分比 (%)":
    withdraw_value = st.sidebar.number_input("年提領比例 (%)", min_value=0.0, max_value=20.0, value=2.5, step=0.1) / 100.0
else:
    withdraw_value = st.sidebar.number_input("年提領金額 (元)", min_value=0, value=600000, step=50000)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 智能抓取新增資產")
with st.sidebar.form("auto_fetch_form"):
    ticker_input = st.text_input("輸入股票/ETF代號 (如: BND, NVDA)")
    custom_type = st.selectbox("核心資產屬性分類", ["原型資產 (Prototype)", "槓桿正2 (Leverage)", "防守短債 (Defensive)"])
    
    if st.form_submit_button("抓取並新增") and ticker_input:
        with st.spinner("真實市場連線中...與 QQQ 進行聯立矩陣運算中..."):
            type_map = {"原型資產 (Prototype)": "Prototype", "槓桿正2 (Leverage)": "Leverage", "防守短債 (Defensive)": "Defensive"}
            data, msg = fetch_asset_base_data(ticker_input, type_map[custom_type])
            if data: 
                calculated_beta = 1.0
                ticker_upper = ticker_input.strip().upper()
                if "QQQ (美股大盤)" in st.session_state.asset_library and ticker_upper != "QQQ":
                    qqq_p = st.session_state.asset_library["QQQ (美股大盤)"]["prices"]
                    asset_p = data["prices"]
                    common_idx = asset_p.index.intersection(qqq_p.index)
                    if len(common_idx) > 30:
                        asset_ret = asset_p.loc[common_idx].pct_change().dropna()
                        qqq_ret = qqq_p.loc[common_idx].pct_change().dropna()
                        common_idx2 = asset_ret.index.intersection(qqq_ret.index)
                        if len(common_idx2) > 30:
                            matrix = np.cov(asset_ret.loc[common_idx2], qqq_ret.loc[common_idx2])
                            calculated_beta = matrix[0][1] / matrix[1][1] if matrix[1][1] != 0 else 1.0
                
                data["beta"] = calculated_beta
                st.session_state.asset_library[f"{ticker_upper} (自訂)"] = data
                st.success(f"{msg} (系統自動精確對標 QQQ 計算之 Beta 值 = {calculated_beta:.2f})")
                st.rerun()

# ==========================================
# 4. 核心計算引擎
# ==========================================
def calculate_metrics(strategy_config, margin_rate, start_date, end_date, init_capital, withdraw_mode, withdraw_value):
    weights_dict = strategy_config["wts"]
    rebalance_type = strategy_config["rebal"]
    debt_mode = strategy_config["debt_mode"]
    target_margin_ratio = strategy_config.get("target_margin", 6.0) 
    
    initial_total_weight = sum(weights_dict.values())
    initial_debt_ratio = max(0, initial_total_weight - 100.0)
    sys_beta = 0.0
    
    master_prices = st.session_state.asset_library["QQQ (美股大盤)"]["prices"]
    master_slice = master_prices.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    trading_days = master_slice.index
    
    df_returns = pd.DataFrame(index=trading_days)
    
    for name, weight in weights_dict.items():
        if weight <= 0 or name not in st.session_state.asset_library:
            continue
        asset = st.session_state.asset_library[name]
        sys_beta += asset.get("beta", 0.0) * (weight / 100.0)
        
        if name in ["無 (不配置)", "現金"] or asset["prices"].empty:
            df_returns[name] = 0.0
        else:
            p_slice = asset["prices"].reindex(trading_days)
            real_returns = p_slice.pct_change().fillna(0.0)
            df_returns[name] = real_returns
            
    if len(trading_days) > 0:
        eoy_dates = set(df_returns.groupby(df_returns.index.year).apply(lambda x: x.index[-1]))
    else:
        eoy_dates = set()
        
    portfolio_equity = init_capital 
    current_debt_amount = (initial_debt_ratio / 100.0) * init_capital
    current_asset_amounts = {name: (weight/100.0) * init_capital for name, weight in weights_dict.items()}
    
    year_start_assets = {name: current_asset_amounts[name] for name in current_asset_amounts}
    prev_eoy_equity = init_capital
    
    strategy_annuals = {}
    equity_curve = [] 
    reg_margin_curve = [] 
    bond_margin_curve = [] 
    total_withdrawn = 0.0
    is_bankrupt = False
    bankruptcy_date = ""
    daily_interest_rate = margin_rate / 252.0

    for date in trading_days:
        if is_bankrupt:
            equity_curve.append({"日期": date, "淨值": 0.0, "負債": current_debt_amount})
            reg_margin_curve.append({"日期": date, "法規維持率": 0.0})
            bond_margin_curve.append({"日期": date, "純債維持率": 0.0})
            if date in eoy_dates: strategy_annuals[date.year] = -1.0
            continue
            
        for name, amount in current_asset_amounts.items():
            if name not in st.session_state.asset_library or amount <= 0:
                continue
            asset_info = st.session_state.asset_library[name]
            
            if name in ["無 (不配置)", "現金"]:
                ret = 0.02 / 252.0
            elif date.date() >= asset_info["inception_date"]:
                ret = df_returns.loc[date, name]
            else:
                if asset_info["type"] == "Defensive":
                    ret = 0.02 / 252.0
                elif asset_info["type"] == "Leverage":
                    qqq_ret = df_returns.loc[date, "QQQ (美股大盤)"] if "QQQ (美股大盤)" in df_returns.columns else 0.0
                    ret = qqq_ret * 2.0 - (0.012 / 252.0)
                else:
                    qqq_ret = df_returns.loc[date, "QQQ (美股大盤)"] if "QQQ (美股大盤)" in df_returns.columns else 0.0
                    ret = qqq_ret
                    
            current_asset_amounts[name] = amount * (1 + ret)
                
        interest_cost = current_debt_amount * daily_interest_rate
        current_debt_amount += interest_cost
        
        withdrawal_amount = 0
        if debt_mode == "買借死 (提領生活費)":
            if withdraw_mode == "總資產百分比 (%)":
                withdrawal_amount = (sum(current_asset_amounts.values()) * withdraw_value) / 252.0
            else:
                withdrawal_amount = withdraw_value / 252.0
                
        current_debt_amount += withdrawal_amount
        total_withdrawn += withdrawal_amount
            
        year_end_assets = sum(current_asset_amounts.values())
        portfolio_equity = year_end_assets - current_debt_amount
        
        legal_collateral = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library[n].get("type") in ["Prototype", "Defensive"]])
        bond_collateral = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library[n].get("type") == "Defensive"])
        
        current_reg_margin = legal_collateral / current_debt_amount if current_debt_amount > 0 else 10.0
        current_bond_margin = bond_collateral / current_debt_amount if current_debt_amount > 0 else 10.0
        
        display_reg = min(current_reg_margin, 10.0)
        display_bond = min(current_bond_margin, 10.0)
        
        if portfolio_equity <= 0:
            portfolio_equity = 0; is_bankrupt = True; bankruptcy_date = date.strftime("%Y-%m-%d")
        elif current_reg_margin < 1.4 and current_debt_amount > 0: 
            portfolio_equity = 0; is_bankrupt = True; bankruptcy_date = date.strftime("%Y-%m-%d")

        equity_curve.append({"日期": date, "淨值": portfolio_equity, "負債": current_debt_amount})
        reg_margin_curve.append({"日期": date, "法規維持率": display_reg})
        bond_margin_curve.append({"日期": date, "純債維持率": display_bond})
            
        if date in eoy_dates and not is_bankrupt:
            strategy_annuals[date.year] = (portfolio_equity / prev_eoy_equity) - 1.0 if prev_eoy_equity > 0 else 0
            prev_eoy_equity = portfolio_equity
            
            if rebalance_type == "CLEC":
                for name, amount in current_asset_amounts.items():
                    if st.session_state.asset_library[name].get("type") == "Leverage":
                        yr_ret = (amount / year_start_assets[name]) - 1.0 if year_start_assets[name] > 0 else 0
                        if yr_ret > 0:
                            extract = ((amount / (1+yr_ret)) * yr_ret) * 0.3
                            current_asset_amounts[name] -= extract
                            for d_name in current_asset_amounts.keys():
                                if st.session_state.asset_library[d_name].get("type") == "Defensive": current_asset_amounts[d_name] += extract; break
                        elif yr_ret < 0:
                            for d_name in current_asset_amounts.keys():
                                if st.session_state.asset_library[d_name].get("type") == "Defensive":
                                    rescue = current_asset_amounts[d_name] * 0.02
                                    current_asset_amounts[d_name] -= rescue; current_asset_amounts[name] += rescue; break
            elif rebalance_type == "傳統定時":
                total_assets = sum(current_asset_amounts.values())
                for name, weight in weights_dict.items(): current_asset_amounts[name] = total_assets * (weight/sum(weights_dict.values()))

            if debt_mode == "恆定維持率 (增貸再投資)":
                if legal_collateral > 0 and current_reg_margin > target_margin_ratio:
                    new_loan = (legal_collateral / target_margin_ratio) - current_debt_amount
                    if new_loan > 0:
                        current_debt_amount += new_loan
                        total_assets_now = sum(current_asset_amounts.values())
                        if total_assets_now > 0:
                            for n in current_asset_amounts.keys(): current_asset_amounts[n] += new_loan * (current_asset_amounts[n] / total_assets_now)
            
            for name in current_asset_amounts: year_start_assets[name] = current_asset_amounts[name]

    num_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    cagr = ((portfolio_equity / init_capital) ** (1 / num_years)) - 1 if num_years > 0 and not is_bankrupt and portfolio_equity > 0 else 0
    
    df_curve = pd.DataFrame(equity_curve)
    if not df_curve.empty and portfolio_equity > 0:
        df_curve["最高淨值"] = df_curve["淨值"].cummax()
        df_curve["水下回撤"] = (df_curve["淨值"] / df_curve["最高淨值"]) - 1.0 
        real_mdd = df_curve["水下回撤"].min()
        real_vol = df_curve["淨值"].pct_change().std() * np.sqrt(252) if len(df_curve) > 1 else 0.0
        sharpe = (cagr - RISK_FREE_RATE) / real_vol if real_vol > 0 else 0
        
        is_underwater = df_curve["水下回撤"] < 0
        drop_groups = (~is_underwater).cumsum()[is_underwater]
        max_recovery_days = int(drop_groups.value_counts().max() * (365.25 / 252)) if not drop_groups.empty else 0
        
        if is_underwater.any():
            drawdowns_pct = df_curve["水下回撤"] * 100
            ulcer_index = np.sqrt(np.mean(drawdowns_pct ** 2))
        else:
            ulcer_index = 0.0
    else:
        real_mdd = -1.0; real_vol = 0.0; sharpe = 0; max_recovery_days = 9999; ulcer_index = 99.9

    calmar = cagr / abs(real_mdd) if real_mdd != 0 else 0
    
    return {
        "總權重": initial_total_weight, "負債模式": debt_mode, "再平衡": rebalance_type, "系統 Beta": sys_beta, 
        "年化淨報酬率(CAGR)": cagr, "最終淨值": portfolio_equity, "年化波動率": real_vol,
        "最大回撤": real_mdd, "夏普值": sharpe, "卡瑪比率": calmar, "最大修復天數": max_recovery_days, "痛苦指數": ulcer_index,
        "累計提領生活費": total_withdrawn, "狀態": f"破產 ({bankruptcy_date} 觸及 140% 斷頭)" if is_bankrupt else "安全存活",
        "annuals": strategy_annuals, "curve": equity_curve, "reg_margin_curve": reg_margin_curve, "bond_margin_curve": bond_margin_curve, "有效年數": num_years,
        "類型": "純大盤對照" if len([w for w in weights_dict.values() if w > 0]) == 1 else ("自訂戰略" if "🎯" in strategy_config.get("name", "") else "經典對照")
    }

# ==========================================
# 5. 主畫面：策略建構器
# ==========================================
st.title("📊 CLEC 質押策略績效戰情室")

st.subheader("🛠 建立自訂組合戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱", f"策略模式 {len(st.session_state.custom_strategies)+1}")
    
    col_r, col_d, col_m = st.columns(3)
    with col_r: rebal_mode = st.selectbox("再平衡模組", ["CLEC", "傳統定時", "不執行"], index=0)
    with col_d: debt_mode = st.selectbox("負債運用模組", ["買借死 (提領生活費)", "恆定維持率 (增貸再投資)", "無"], index=0)
    with col_m: target_margin_input = st.number_input("目標維持率 (%)", min_value=140, max_value=2000, value=600, step=50)
    
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
        st.session_state.custom_strategies[strat_name] = {
            "name": "🎯 " + strat_name, 
            "wts": selected_assets, 
            "rebal": rebal_mode, 
            "debt_mode": debt_mode,
            "target_margin": target_margin_input / 100.0 
        }
        st.success(f"已成功加入「{strat_name}」！")
        st.rerun()

if st.session_state.custom_strategies:
    st.markdown("#### ⚙️ 管理與檢視自訂策略")
    with st.expander("🔍 點擊查看所有自訂策略的詳細配比", expanded=True):
        for s_name, config in st.session_state.custom_strategies.items():
            wts_str = " + ".join([f"{k.split(' ')[0]} ({v}%)" for k, v in config["wts"].items() if v > 0])
            margin_info = f" ｜ 目標維持率: {config.get('target_margin', 6.0) * 100:.0f}%" if config['debt_mode'] == "恆定維持率 (增貸再投資)" else ""
            st.info(f"**{s_name}**\n\n👉 配比：`{wts_str}`\n\n👉 設定：{config['rebal']} ｜ {config['debt_mode']}{margin_info}")

    col_del1, col_del2, col_del3 = st.columns([2, 1, 1])
    with col_del1: del_target = st.selectbox("選擇要刪除的策略", list(st.session_state.custom_strategies.keys()), label_visibility="collapsed")
    with col_del2: 
        if st.button("刪除單一策略", use_container_width=True): del st.session_state.custom_strategies[del_target]; st.rerun()
    with col_del3:
        if st.button("⚠️ 全數清空", type="primary", use_container_width=True): st.session_state.custom_strategies = {}; st.rerun()

st.markdown("---")

# ==========================================
# 6. 終極比較表與最佳配置儀表板
# ==========================================
comp_data = []
annual_chart_data = []
curve_chart_data = []
reg_margin_chart_data = []
bond_margin_chart_data = []

all_strategies = {}
for k, v in st.session_state.benchmark_strategies.items(): all_strategies[k] = v
for k, v in st.session_state.custom_strategies.items(): all_strategies[v["name"]] = v

with st.spinner("⏳ 核心引擎運作中..."):
    for name, config in all_strategies.items():
        res = calculate_metrics(config, margin_rate, start_date, end_date, init_capital=init_capital, withdraw_mode=withdraw_mode, withdraw_value=withdraw_value)
        res["策略名稱"] = name
        comp_data.append(res)
        for year, ret in res["annuals"].items(): annual_chart_data.append({"策略名稱": name, "年份": year, "報酬率": ret, "類型": res["類型"]})
        for pt in res["curve"]: curve_chart_data.append({"策略名稱": name, "日期": pt["日期"], "淨值": pt["淨值"]})
        for pt in res["reg_margin_curve"]: reg_margin_chart_data.append({"策略名稱": name, "日期": pt["日期"], "法規維持率": pt["法規維持率"]})
        for pt in res["bond_margin_curve"]: bond_margin_chart_data.append({"策略名稱": name, "日期": pt["日期"], "純債維持率": pt["純債維持率"]})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    
    st.markdown("### 📊 績效比較表")
    
    with st.expander("📖 點擊查看量化指標白話文說明"):
        st.markdown("""
        * **夏普值 (Sharpe Ratio)**：每承受 1 單位波動風險，能換取多少超額報酬。越高越好，大於 1 算優秀，代表這套策略「漲得穩」。
        * **卡瑪比率 (Calmar Ratio)**：年化報酬率除以最大回撤的絕對值。衡量你「每忍受 1% 的極限跌幅，每年能賺回多少利潤」。數值越高，代表遇到股災時的 CP 值越高。
        * **痛苦指數 (Ulcer Index)**：不只看跌多深，還看你在水下「憋氣套牢了多久」。數值越低越好，越低代表投資人晚上睡得越安穩。
        * **年化淨報酬率 (CAGR)**：在本系統包含現金流（提領生活費）的模型中，CAGR 的計算邏輯已非常貼近 IRR（內部報酬率）的實質意義。
        """)
    
    cols_order = [
        "策略名稱", "負債模式", "再平衡", "狀態", 
        "系統 Beta", "年化淨報酬率(CAGR)", "年化波動率", "最大回撤", "痛苦指數", 
        "夏普值", "卡瑪比率", "最大修復天數", "最終淨值", "累計提領生活費"
    ]
    df_display = df_comp[cols_order].copy()
    
    df_display["系統 Beta"] = df_display["系統 Beta"].apply(lambda x: f"{x:.2f}")
    df_display["年化淨報酬率(CAGR)"] = df_display["年化淨報酬率(CAGR)"].apply(lambda x: f"{x*100:.2f}%")
    df_display["年化波動率"] = df_display["年化波動率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["痛苦指數"] = df_display["痛苦指數"].apply(lambda x: f"{x:.2f}")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    df_display["卡瑪比率"] = df_display["卡瑪比率"].apply(lambda x: f"{x:.3f}")
    df_display["最大修復天數"] = df_display["最大修復天數"].apply(lambda x: f"{x:,} 天" if x < 9999 else "已斷頭破產")
    df_display["最終淨值"] = df_display["最終淨值"].apply(lambda x: f"NT$ {x:,.0f}")
    df_display["累計提領生活費"] = df_display["累計提領生活費"].apply(lambda x: f"NT$ {x:,.0f}")
    
    # 恢復為最簡潔的名詞，去除了括號內的雜訊
    df_display = df_display.rename(columns={
        "年化淨報酬率(CAGR)": "年化淨報酬率 (CAGR/IRR)",
        "最大回撤": "最大回撤 (MDD)"
    })
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    st.markdown("### 👑 最佳配置方案評估")
    
    valid_df = df_comp[df_comp["狀態"] == "安全存活"]
    if not valid_df.empty:
        best_cagr = valid_df.loc[valid_df["年化淨報酬率 (CAGR/IRR)"].str.rstrip('%').astype(float).idxmax()] if "年化淨報酬率 (CAGR/IRR)" in df_display.columns else valid_df.loc[valid_df["年化淨報酬率(CAGR)"].idxmax()]
        best_equity = valid_df.loc[valid_df["最終淨值"].idxmax()]
        best_mdd = valid_df.loc[valid_df["最大回撤"].idxmax()] 
        best_recovery = valid_df.loc[valid_df["最大修復天數"].idxmin()]
        best_ulcer = valid_df.loc[valid_df["痛苦指數"].idxmin()]
        best_sharpe = valid_df.loc[valid_df["夏普值"].idxmax()]
        best_calmar = valid_df.loc[valid_df["卡瑪比率"].idxmax()]

        st.markdown("##### 🏆 絕對收益視角")
        c1, c2 = st.columns(2)
        with c1: st.info(f"**最高最終餘額**\n### NT$ {best_equity['最終淨值']:,.0f}\n---\n#### 🏆 冠軍策略： `{best_equity['策略名稱']}`")
        with c2: st.info(f"**最高年化回報 (CAGR/IRR)**\n### {best_cagr['年化淨報酬率(CAGR)']*100:.2f}%\n---\n#### 🏆 冠軍策略： `{best_cagr['策略名稱']}`")

        st.markdown("##### 🛡️ 風險與回撤視角")
        c3, c4, c5 = st.columns(3)
        with c3: st.warning(f"**最低最大回撤 (MDD)**\n### {best_mdd['最大回撤']*100:.2f}%\n---\n#### 🏆 冠軍策略： `{best_mdd['策略名稱']}`")
        with c4: st.warning(f"**最短套牢修復期**\n### {best_recovery['最大修復天數']:,} 天\n---\n#### 🏆 冠軍策略： `{best_recovery['策略名稱']}`")
        with c5: st.warning(f"**最低痛苦指數 (Ulcer Index)**\n### {best_ulcer['痛苦指數']:.2f}\n---\n#### 🏆 冠軍策略： `{best_ulcer['策略名稱']}`")

        st.markdown("##### ⚖️ 風險收益比視角")
        c6, c7 = st.columns(2)
        with c6: st.success(f"**最高夏普比率 (Sharpe)**\n### {best_sharpe['夏普值']:.3f}\n---\n#### 🏆 冠軍策略： `{best_sharpe['策略名稱']}`")
        with c7: st.success(f"**最高卡瑪比率 (Calmar)**\n### {best_calmar['卡瑪比率']:.3f}\n---\n#### 🏆 冠軍策略： `{best_calmar['策略名稱']}`")
    else:
        st.error("⚠️ 壓力測試失敗：在您設定的條件下，所有策略均已宣告破產，無法產生最佳方案評估。")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💰 最終資產淨值排行 (NT$)")
        df_chart_multiple = df_comp.sort_values(by="最終淨值", ascending=True)
        max_val = df_chart_multiple["最終淨值"].max()
        fig_mult = px.bar(df_chart_multiple, x="最終淨值", y="策略名稱", color="類型", orientation='h', text="最終淨值",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_mult.update_layout(xaxis=dict(range=[0, max_val * 1.35]))
        fig_mult.update_traces(texttemplate='NT$ %{text:,.0f}', textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_mult, use_container_width=True)
    with col2:
        st.subheader("🛡 壓力測試：最大回撤 (MDD)")
        df_chart_mdd = df_comp.sort_values(by="最大回撤", ascending=True)
        min_mdd = df_chart_mdd["最大回撤"].min()
        fig_mdd = px.bar(df_chart_mdd, x="最大回撤", y="策略名稱", color="類型", orientation='h', text="最大回撤",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_mdd.update_layout(xaxis=dict(range=[min_mdd * 1.3, 0]), xaxis_tickformat='.0%')
        fig_mdd.update_traces(texttemplate='%{text:.2%}', textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_mdd, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🎯 策略卡瑪比率 (每單位回撤帶來的利潤)")
        df_chart_calmar = df_comp.sort_values(by="卡瑪比率", ascending=True)
        max_calmar = df_chart_calmar["卡瑪比率"].max()
        fig_calmar = px.bar(df_chart_calmar, x="卡瑪比率", y="策略名稱", color="類型", orientation='h', text="卡瑪比率",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_calmar.update_layout(xaxis=dict(range=[0, max_calmar * 1.25]))
        fig_calmar.update_traces(texttemplate='%{text:.3f}', textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_calmar, use_container_width=True)
    with col4:
        st.subheader("⏳ 最長套牢修復期 (越短越好)")
        df_chart_rec = df_comp.sort_values(by="最大修復天數", ascending=False)
        max_rec = df_chart_rec["最大修復天數"].max()
        fig_rec = px.bar(df_chart_rec, x="最大修復天數", y="策略名稱", color="類型", orientation='h', text="最大修復天數",
            color_discrete_map={"純大盤對照": "#7f7f7f", "經典對照": "#54A24B", "自訂戰略": "#E45756"})
        fig_rec.update_layout(xaxis=dict(range=[0, max_rec * 1.35]))
        df_display_rec_text = df_chart_rec["最大修復天數"].apply(lambda x: f"{x:,} 天" if x < 9999 else "已斷頭破產")
        fig_rec.update_traces(text=df_display_rec_text, textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_rec, use_container_width=True)

    st.subheader("📈 實質金額複利成長曲線 (Log Scale)")
    if curve_chart_data:
        df_curves = pd.DataFrame(curve_chart_data)
        df_curves_sampled = df_curves.iloc[::5, :]
        fig_curves = px.line(df_curves_sampled, x="日期", y="淨值", color="策略名稱", log_y=True)
        fig_curves.update_layout(yaxis_title="資產淨值 (NT$)", xaxis_title="日期", height=450)
        st.plotly_chart(fig_curves, use_container_width=True)
        
    st.markdown("---")
    st.subheader("🚨 雙軌風險防線追蹤圖 (維持率觀測)")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("1. 法規維持率追蹤線 (原型+短債)")
        if reg_margin_chart_data:
            df_reg = pd.DataFrame(reg_margin_chart_data)
            df_reg_sampled = df_reg.iloc[::5, :]
            fig_reg = px.line(df_reg_sampled, x="日期", y="法規維持率", color="策略名稱")
            fig_reg.add_hline(y=1.4, line_dash="dash", line_color="red", annotation_text="140% 斷頭線")
            fig_reg.update_layout(yaxis_tickformat='.0%', yaxis_title="維持率", xaxis_title="日期", height=400)
            fig_reg.update_yaxes(range=[0, 10])
            st.plotly_chart(fig_reg, use_container_width=True)
    with col_m2:
        st.subheader("2. 純債安全維持率追蹤線 (僅計短債)")
        if bond_margin_chart_data:
            df_bond = pd.DataFrame(bond_margin_chart_data)
            df_bond_sampled = df_bond.iloc[::5, :]
            fig_bond = px.line(df_bond_sampled, x="日期", y="純債維持率", color="策略名稱")
            fig_bond.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="100% 短債枯竭線")
            fig_bond.update_layout(yaxis_tickformat='.0%', yaxis_title="純債維持率", xaxis_title="日期", height=400)
            fig_bond.update_yaxes(range=[0, 10])
            st.plotly_chart(fig_bond, use_container_width=True)

    st.markdown("---")
    st.subheader("📆 歷年淨報酬率大亂鬥")
    if annual_chart_data:
        df_annual = pd.DataFrame(annual_chart_data).sort_values(by="年份")
        color_map = {
            "純抱 SPY": "#c7c7c7", 
            "純抱 QQQ": "#7f7f7f", 
            "經典 CLEC 433 (買借死)": "#1f77b4", 
            "穩健 623 (恆定增貸)": "#ff7f0e",
            "防禦 812 (恆定增貸)": "#2ca02c"
        }
        custom_colors = px.colors.sequential.Reds[3:] 
        for idx, custom_name in enumerate(st.session_state.custom_strategies.keys()): color_map["🎯 " + custom_name] = custom_colors[idx % len(custom_colors)]
        fig_annual = px.bar(df_annual, x="年份", y="報酬率", color="策略名稱", barmode="group", color_discrete_map=color_map)
        fig_annual.update_layout(yaxis_tickformat='.0%', yaxis_title="年度淨報酬率", xaxis_title="年份", height=450)
        st.plotly_chart(fig_annual, use_container_width=True)
        
    st.markdown("---")
    # 💥 全新改版：實戰白話文攻略報告
    st.markdown("### 📝 實戰攻略：如何調配出比「純抱 QQQ」更完美的比例？")
    st.info("""
    不想看學術名詞？直接給你三套「打敗大盤」的實戰調配指南：

    💡 **目標 1：追求更高的「夏普值」(CP值最高、漲得最穩)**
    * **破解思路**：純大盤的波動還是太大了，你需要「微量槓桿來保住報酬 + 大量防護來壓低波動」。
    * **建議配比**：試試 **`50% QQQ + 15% QLD + 35% SGOV`**。利用 15% 的正2維持整體向上的攻擊力，同時用高達 35% 的超短債強力吸收震盪，這能讓你的資金曲線變得非常平滑。

    🛡️ **目標 2：追求更低的「最大回撤」(抗跌第一、睡得安穩)**
    * **破解思路**：遇到股災要少跌，唯一解法就是擴大 SGOV (現金水庫)，並且「降低或捨棄」正2槓桿的耗損。
    * **建議配比**：試試 **`70% QQQ + 0% QLD + 30% SGOV`** 或內建的 **`防禦 812`** 策略。把 SGOV 比例拉高到 30% 甚至 40%，就算遇到 2008 年金融海嘯，你的帳戶跌幅也會比 QQQ 小非常多。

    🔥 **目標 3：追求極致的「最終淨值」(承受同等風險，但要賺更多)**
    * **破解思路**：如果你自認心臟夠大顆，能承受跟 QQQ 差不多的劇烈波動，那就透過短債借錢，讓正2來拉抬上限。
    * **建議配比**：直接選用內建的 **`經典 CLEC 433` (40% QQQ + 30% QLD + 30% SGOV)**。由短債負責繳利息與防斷頭，30% 的正2負責在牛市狂奔。只要不被斷頭，長線滾出來的複利會相當驚人。
    """)
