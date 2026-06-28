import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import datetime
import re
import time

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
            
        data = pd.DataFrame()
        for _ in range(3):
            temp_data = yf.download(ticker, start="1999-01-01", end=datetime.date.today(), progress=False)
            if not temp_data.empty:
                data = temp_data
                break
            time.sleep(0.5)
            
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
    
    qqq_data, _ = fetch_asset_base_data("QQQ", "Prototype")
    if qqq_data:
        qqq_data["beta"] = 1.0
        lib["QQQ (美股大盤)"] = qqq_data
        qqq_ret = qqq_data["prices"].pct_change().dropna()
    else:
        qqq_ret = pd.Series(dtype=float)

    defaults = [
        ("SPY", "SPY (標普大盤)", "Prototype"),
        ("QLD", "QLD (美股正2)", "Leverage"),
        ("TQQQ", "TQQQ (美股正3)", "Leverage"),
        ("0050.TW", "0050 (台股大盤)", "Prototype"),
        ("00662.TW", "00662 (NAS原型)", "Prototype"),
        ("009800.TW", "009800 (統一美網原型)", "Prototype"), 
        ("00646.TW", "00646 (標普原型)", "Prototype"), 
        ("00713.TW", "00713 (台股高息)", "Prototype"),
        ("00631L.TW", "00631L (台股正2)", "Leverage"),
        ("00670L.TW", "00670L (美股正2)", "Leverage"),
        ("SGOV", "SGOV (美股超短債)", "Defensive"),
        ("SHY", "SHY (美股1-3年短債)", "Defensive"),
        ("KMLM", "KMLM (趨勢追蹤期貨)", "Defensive"),
        ("DBMF", "DBMF (管理期貨策略)", "Defensive"),
        ("00865B.TW", "00865B (台股短債)", "Defensive"),
        ("00859B.TW", "00859B (台股投資級債)", "Defensive")
    ]
    
    for ticker, display_name, a_type in defaults:
        data, _ = fetch_asset_base_data(ticker, a_type)
        if data:
            calc_beta = 1.0
            if not qqq_ret.empty and a_type != "Defensive":
                asset_ret = data["prices"].pct_change().dropna()
                common_idx = asset_ret.index.intersection(qqq_ret.index)
                if len(common_idx) > 30:
                    matrix = np.cov(asset_ret.loc[common_idx], qqq_ret.loc[common_idx])
                    calc_beta = matrix[0][1] / matrix[1][1] if matrix[1][1] != 0 else 1.0
            elif a_type == "Defensive":
                asset_ret = data["prices"].pct_change().dropna()
                common_idx = asset_ret.index.intersection(qqq_ret.index)
                if len(common_idx) > 30:
                    matrix = np.cov(asset_ret.loc[common_idx], qqq_ret.loc[common_idx])
                    calc_beta = matrix[0][1] / matrix[1][1] if matrix[1][1] != 0 else 0.0
                else:
                    calc_beta = 0.0
                
            data["beta"] = calc_beta
            lib[display_name] = data
            
    return lib

if 'asset_library' not in st.session_state:
    st.session_state.asset_library = load_default_assets()

st.session_state.benchmark_strategies = {
    "純抱 SPY": {"wts": {"SPY (標普大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無", "target_margin": 6.0, "tactical": "無"},
    "純抱 QQQ": {"wts": {"QQQ (美股大盤)": 100.0}, "rebal": "不執行", "debt_mode": "無", "target_margin": 6.0, "tactical": "無"},
    "經典 CLEC 433 (買借死)": {"wts": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SGOV (美股超短債)": 30.0}, "rebal": "CLEC", "debt_mode": "買借死 (提領生活費)", "target_margin": 6.0, "tactical": "無"},
    "穩健 623 (恆定維持率 600%)": {"wts": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SGOV (美股超短債)": 30.0}, "rebal": "CLEC", "debt_mode": "恆定維持率 (增貸再投資)", "target_margin": 6.0, "tactical": "無"},
    "防禦 812 (年度常規 800%)": {"wts": {"QQQ (美股大盤)": 80.0, "QLD (美股正2)": 10.0, "SGOV (美股超短債)": 20.0}, "rebal": "CLEC", "debt_mode": "恆定維持率 (增貸再投資)", "target_margin": 8.0, "tactical": "無"},
    "彈性防禦 812 (防守型 800%)": {"wts": {"QQQ (美股大盤)": 80.0, "QLD (美股正2)": 10.0, "SGOV (美股超短債)": 20.0}, "rebal": "CLEC彈性(防守)", "debt_mode": "恆定維持率 (增貸再投資)", "target_margin": 8.0, "tactical": "無"},
    "QLD 50-50 (無負債)": {"wts": {"QLD (美股正2)": 50.0, "SGOV (美股超短債)": 50.0}, "rebal": "傳統定時", "debt_mode": "無", "target_margin": 6.0, "tactical": "無"},
    "TQQQ SGOV 333 (無負債)": {"wts": {"TQQQ (美股正3)": 33.3, "SGOV (美股超短債)": 66.7}, "rebal": "傳統定時", "debt_mode": "無", "target_margin": 6.0, "tactical": "無"}
}

if 'custom_strategies' not in st.session_state: st.session_state.custom_strategies = {}

# ==========================================
# 3. 智慧時間軸控制器 
# ==========================================
st.title("📊 CLEC 質押策略績效戰情室")

missing_assets = [name for name in ["QQQ (美股大盤)", "QLD (美股正2)", "SGOV (美股超短債)"] if name not in st.session_state.asset_library]
if missing_assets:
    st.error(f"🚨 **嚴重警告：核心數據抓取失敗！**\n\n系統未能從 Yahoo Finance 取得以下資產的歷史報價：`{', '.join(missing_assets)}`。這通常是 API 短暫限流或網路不穩所致。\n\n**這將導致回測績效嚴重失真！** 請點擊下方按鈕強制重新抓取。")
    if st.button("🔄 強制重新抓取並清除快取"):
        st.cache_data.clear()
        if 'asset_library' in st.session_state: del st.session_state.asset_library
        st.rerun()

st.sidebar.markdown("### 歷史回測與分析引擎")

active_assets = set()
for strats in [st.session_state.benchmark_strategies, st.session_state.custom_strategies]:
    for config in strats.values():
        for asset_name, weight in config["wts"].items():
            if asset_name in st.session_state.asset_library:
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
st.sidebar.subheader("🤖 AI 動態尋優設定")
target_ai_beta = st.sidebar.number_input("AI 尋優目標 Beta (預設 1.0)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
target_ai_debt_mode = st.sidebar.selectbox("AI 尋優指定負債模式", ["恆定維持率 (增貸再投資)", "買借死 (提領生活費)", "無"])
target_ai_tactical = st.sidebar.selectbox("AI 尋優戰術加碼模組", [
    "無", 
    "19/30股災加碼_賣出資產 (每次10%)", 
    "19/30股災加碼_賣出資產 (每次5%)",
    "19/30股災加碼_質押借貸 (每次10%)",
    "19/30股災加碼_質押借貸 (每次5%)"
])

st.sidebar.markdown("##### 🎯 AI 專屬尋優資產池")
ai_asset_opts = list(st.session_state.asset_library.keys())
def get_idx(name):
    return ai_asset_opts.index(name) if name in ai_asset_opts else 0

ai_proto_1 = st.sidebar.selectbox("AI 尋優原型資產 1", ai_asset_opts, index=get_idx("QQQ (美股大盤)"))
ai_proto_2 = st.sidebar.selectbox("AI 尋優原型資產 2 (選填)", ai_asset_opts, index=get_idx("無 (不配置)"))
ai_lev = st.sidebar.selectbox("AI 尋優槓桿資產", ai_asset_opts, index=get_idx("QLD (美股正2)"))
ai_def = st.sidebar.selectbox("AI 尋優防守資產", ai_asset_opts, index=get_idx("SGOV (美股超短債)"))

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
                st.success(f"{msg} (系統自動精精確對標 QQQ 計算之真實 Beta 值 = {calculated_beta:.2f})")
                st.rerun()

# ==========================================
# 4. 核心計算引擎 
# ==========================================
def calculate_metrics(strategy_config, margin_rate, start_date, end_date, init_capital, withdraw_mode, withdraw_value):
    weights_dict = strategy_config["wts"]
    rebalance_type = strategy_config.get("rebal", "不執行")
    debt_mode = strategy_config.get("debt_mode", "無")
    target_margin_ratio = strategy_config.get("target_margin", 6.0) 
    tactical_mode = strategy_config.get("tactical", "無")
    tactical_pct = strategy_config.get("tactical_pct", 10.0) / 100.0  
    
    initial_total_weight = sum(weights_dict.values())
    initial_debt_ratio = max(0, initial_total_weight - 100.0)
    sys_beta = 0.0
    
    # 智能判斷彈藥庫來源：優先找原型資產，若無原型資產則找防守資產(如SGOV)
    main_proto = next((n for n in weights_dict.keys() if st.session_state.asset_library.get(n, {}).get("type") == "Prototype"), None)
    if main_proto is None:
        main_proto = next((n for n in weights_dict.keys() if st.session_state.asset_library.get(n, {}).get("type") == "Defensive"), "QQQ (美股大盤)")
        
    main_lev = next((n for n in weights_dict.keys() if st.session_state.asset_library.get(n, {}).get("type") == "Leverage"), "QLD (美股正2)")
    
    master_prices = pd.Series(dtype=float)
    if "QQQ (美股大盤)" in st.session_state.asset_library and not st.session_state.asset_library["QQQ (美股大盤)"]["prices"].empty:
        master_prices = st.session_state.asset_library["QQQ (美股大盤)"]["prices"]
    elif "SPY (標普大盤)" in st.session_state.asset_library and not st.session_state.asset_library["SPY (標普大盤)"]["prices"].empty:
        master_prices = st.session_state.asset_library["SPY (標普大盤)"]["prices"]
    else:
        for name, asset in st.session_state.asset_library.items():
            if name not in ["無 (不配置)", "現金"] and not asset.get("prices", pd.Series(dtype=float)).empty:
                master_prices = asset["prices"]
                break
                
    if master_prices.empty:
        trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
    else:
        master_slice = master_prices.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
        trading_days = master_slice.index
    
    df_returns = pd.DataFrame(index=trading_days)
    
    for name, weight in weights_dict.items():
        if name not in st.session_state.asset_library:
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
    current_asset_amounts = {name: (weight/100.0) * init_capital for name, weight in weights_dict.items() if name in st.session_state.asset_library}
    
    if "19/30股災加碼" in tactical_mode:
        if main_proto not in current_asset_amounts: current_asset_amounts[main_proto] = 0.0
        if main_lev not in current_asset_amounts: current_asset_amounts[main_lev] = 0.0
    
    year_start_assets = {name: current_asset_amounts[name] for name in current_asset_amounts}
    prev_eoy_equity = init_capital
    last_rebal_equity = init_capital
    last_rebal_assets = current_asset_amounts.copy()
    
    strategy_annuals = {}
    equity_curve = [] 
    reg_margin_curve = [] 
    total_margin_curve = [] 
    bond_margin_curve = [] 
    total_withdrawn = 0.0
    is_bankrupt = False
    bankruptcy_date = ""
    bankruptcy_reason = "存活"
    daily_interest_rate = margin_rate / 252.0

    master_peak_price = 0.0
    lev_index = 1.0               
    lowest_entry_lev_index = 0.0  
    dynamic_fund_shifted = 0.0    
    tactical_borrowed_principal = 0.0 # 追蹤質押加碼借出的本金
    triggered_19 = False
    triggered_30 = False

    for date in trading_days:
        if is_bankrupt:
            equity_curve.append({"日期": date, "淨值": 0.0, "負債": current_debt_amount})
            reg_margin_curve.append({"日期": date, "法規維持率": 0.0})
            total_margin_curve.append({"日期": date, "總擔保維持率": 0.0})
            bond_margin_curve.append({"日期": date, "純債維持率": 0.0})
            if date in eoy_dates: strategy_annuals[date.year] = -1.0
            continue
            
        current_main_lev_ret = 0.0
        
        for name, amount in current_asset_amounts.items():
            if name not in st.session_state.asset_library: 
                continue
            asset_info = st.session_state.asset_library[name]
            
            if amount <= 0:
                ret = 0.0 
            elif name in ["無 (不配置)", "現金"]:
                ret = 0.02 / 252.0
            elif date.date() >= asset_info["inception_date"]:
                ret = df_returns.loc[date, name] if name in df_returns.columns else 0.0
            else:
                proxy_ret = 0.0
                if "QQQ (美股大盤)" in df_returns.columns: proxy_ret = df_returns.loc[date, "QQQ (美股大盤)"]
                elif "SPY (標普大盤)" in df_returns.columns: proxy_ret = df_returns.loc[date, "SPY (標普大盤)"]
                
                if asset_info.get("type") == "Defensive":
                    ret = 0.02 / 252.0
                elif asset_info.get("type") == "Leverage":
                    lev_mult = round(asset_info.get("beta", 2.0))
                    if lev_mult == 0: lev_mult = 2.0
                    ret = proxy_ret * lev_mult - (0.012 / 252.0)
                else:
                    ret = proxy_ret
            
            if name == main_lev:
                current_main_lev_ret = ret
                
            current_asset_amounts[name] = amount * (1 + ret)
                
        lev_index *= (1 + current_main_lev_ret)
        if dynamic_fund_shifted > 0:
            dynamic_fund_shifted *= (1 + current_main_lev_ret)
            
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
        
        if "19/30股災加碼" in tactical_mode:
            if not master_prices.empty and date in master_prices.index:
                current_master_val = master_prices.loc[date]
                # 創新高重置所有戰術狀態
                if current_master_val > master_peak_price:
                    master_peak_price = current_master_val
                    if dynamic_fund_shifted > 0.0:
                        current_asset_amounts[main_lev] -= dynamic_fund_shifted
                        if "質押借貸" in tactical_mode:
                            current_debt_amount -= tactical_borrowed_principal
                            current_asset_amounts[main_proto] += (dynamic_fund_shifted - tactical_borrowed_principal)
                            if current_debt_amount < 0:
                                current_asset_amounts[main_proto] += abs(current_debt_amount)
                                current_debt_amount = 0.0
                        else:
                            current_asset_amounts[main_proto] += dynamic_fund_shifted
                            
                        dynamic_fund_shifted = 0.0
                        tactical_borrowed_principal = 0.0
                        triggered_19 = False
                        triggered_30 = False
                        lowest_entry_lev_index = 0.0
                
                master_dd = (master_peak_price - current_master_val) / master_peak_price if master_peak_price > 0 else 0
                
                # 翻倍停利結算
                if triggered_19 and lowest_entry_lev_index > 0:
                    if lev_index >= lowest_entry_lev_index * 2.0:
                        current_asset_amounts[main_lev] -= dynamic_fund_shifted
                        if "質押借貸" in tactical_mode:
                            current_debt_amount -= tactical_borrowed_principal
                            current_asset_amounts[main_proto] += (dynamic_fund_shifted - tactical_borrowed_principal)
                            if current_debt_amount < 0:
                                current_asset_amounts[main_proto] += abs(current_debt_amount)
                                current_debt_amount = 0.0
                        else:
                            current_asset_amounts[main_proto] += dynamic_fund_shifted
                            
                        dynamic_fund_shifted = 0.0
                        tactical_borrowed_principal = 0.0
                        triggered_19 = False
                        triggered_30 = False
                        lowest_entry_lev_index = 0.0
                        master_peak_price = current_master_val 
                
                # 股災進場 (買入或質押)
                total_assets_now = sum(current_asset_amounts.values())
                if master_dd >= 0.19 and not triggered_19:
                    if "質押借貸" in tactical_mode:
                        shift_val = total_assets_now * tactical_pct
                        current_debt_amount += shift_val
                        current_asset_amounts[main_lev] += shift_val
                        tactical_borrowed_principal += shift_val
                    else:
                        shift_val = min(total_assets_now * tactical_pct, current_asset_amounts[main_proto])
                        current_asset_amounts[main_proto] -= shift_val
                        current_asset_amounts[main_lev] += shift_val
                        
                    dynamic_fund_shifted += shift_val
                    lowest_entry_lev_index = lev_index 
                    triggered_19 = True
                    
                if master_dd >= 0.30 and not triggered_30:
                    if "質押借貸" in tactical_mode:
                        shift_val = total_assets_now * tactical_pct
                        current_debt_amount += shift_val
                        current_asset_amounts[main_lev] += shift_val
                        tactical_borrowed_principal += shift_val
                    else:
                        shift_val = min(total_assets_now * tactical_pct, current_asset_amounts[main_proto])
                        current_asset_amounts[main_proto] -= shift_val
                        current_asset_amounts[main_lev] += shift_val
                        
                    dynamic_fund_shifted += shift_val
                    lowest_entry_lev_index = lev_index 
                    triggered_30 = True

        # 💥 修正：將 Prototype (原型) 與 Defensive (防守短債) 都計入法定擔保品 (legal_collateral)
        legal_collateral = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library.get(n, {}).get("type") in ["Prototype", "Defensive"]])
        
        if current_debt_amount > 0:
            current_reg_margin = legal_collateral / current_debt_amount
            if current_reg_margin < 1.60:
                target_debt = legal_collateral / 1.60
                shortfall = current_debt_amount - target_debt
                if shortfall > 0:
                    for d_name in current_asset_amounts.keys():
                        if st.session_state.asset_library.get(d_name, {}).get("type") == "Defensive" and current_asset_amounts[d_name] > 0:
                            repay = min(shortfall, current_asset_amounts[d_name])
                            current_asset_amounts[d_name] -= repay
                            current_debt_amount -= repay
                            shortfall -= repay
                            if shortfall <= 0: break

        # 💥 確保更新維持率時的分子也包含 Defensive
        legal_collateral = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library.get(n, {}).get("type") in ["Prototype", "Defensive"]])
        defensive_collateral = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library.get(n, {}).get("type") == "Defensive"])
        total_collateral = legal_collateral # 總擔保品與法定擔保品等價
        
        current_reg_margin = legal_collateral / current_debt_amount if current_debt_amount > 0 else 10.0
        current_total_margin = total_collateral / current_debt_amount if current_debt_amount > 0 else 10.0
        current_bond_margin = defensive_collateral / current_debt_amount if current_debt_amount > 0 else 10.0
        
        if portfolio_equity <= 0:
            portfolio_equity = 0; is_bankrupt = True; bankruptcy_date = date.strftime("%Y-%m-%d"); bankruptcy_reason = "淨值歸零"
        elif current_reg_margin < 1.4 and current_debt_amount > 0: 
            portfolio_equity = 0; is_bankrupt = True; bankruptcy_date = date.strftime("%Y-%m-%d"); bankruptcy_reason = "觸及 140% 斷頭"

        equity_curve.append({"日期": date, "淨值": portfolio_equity, "負債": current_debt_amount})
        reg_margin_curve.append({"日期": date, "法規維持率": min(current_reg_margin, 10.0)})
        total_margin_curve.append({"日期": date, "總擔保維持率": min(current_total_margin, 10.0)})
        bond_margin_curve.append({"日期": date, "純債維持率": min(current_bond_margin, 10.0)})
        
        is_tactical_active = (dynamic_fund_shifted > 0.0)

        if not is_bankrupt and not is_tactical_active:
            # 💥 新增：獨立的閾值再平衡機制 (監控槓桿部位是否漲跌 50%)
            if rebalance_type == "閾值平衡(±50%)":
                trigger_rebal = False
                for name, amount in current_asset_amounts.items():
                    if st.session_state.asset_library.get(name, {}).get("type") == "Leverage":
                        last_amt = last_rebal_assets.get(name, amount)
                        if last_amt > 0:
                            if amount >= last_amt * 1.50 or amount <= last_amt * 0.50:
                                trigger_rebal = True
                                break
                if trigger_rebal:
                    total_assets = sum(current_asset_amounts.values())
                    for name, weight in weights_dict.items(): 
                        if name in current_asset_amounts:
                            current_asset_amounts[name] = total_assets * (weight/sum([w for k,w in weights_dict.items() if k in current_asset_amounts]))
                    last_rebal_equity = portfolio_equity
                    last_rebal_assets = current_asset_amounts.copy()

            elif rebalance_type in ["CLEC彈性(防守)", "CLEC彈性(進取)"]:
                is_defensive = (rebalance_type == "CLEC彈性(防守)")
                threshold_up = 1.15 if is_defensive else 1.25
                threshold_down = 0.90 if is_defensive else 0.82
                extract_pct = 0.45 if is_defensive else 0.30
                rescue_pct = 0.0175 if is_defensive else 0.0275
                
                trigger_rebal = False
                
                if portfolio_equity >= last_rebal_equity * threshold_up:
                    for name, amount in current_asset_amounts.items():
                        if st.session_state.asset_library.get(name, {}).get("type") == "Leverage":
                            profit = amount - last_rebal_assets.get(name, amount)
                            if profit > 0:
                                extract = profit * extract_pct
                                current_asset_amounts[name] -= extract
                                for d_name in current_asset_amounts.keys():
                                    if st.session_state.asset_library.get(d_name, {}).get("type") == "Defensive":
                                        current_asset_amounts[d_name] += extract
                                        break
                    trigger_rebal = True
                    
                elif portfolio_equity <= last_rebal_equity * threshold_down:
                    total_assets_current = sum(current_asset_amounts.values())
                    for d_name in current_asset_amounts.keys():
                        if st.session_state.asset_library.get(d_name, {}).get("type") == "Defensive":
                            rescue = min(current_asset_amounts[d_name], total_assets_current * rescue_pct)
                            current_asset_amounts[d_name] -= rescue
                            for l_name in current_asset_amounts.keys():
                                if st.session_state.asset_library.get(l_name, {}).get("type") == "Leverage":
                                    current_asset_amounts[l_name] += rescue
                                    break
                            break
                    trigger_rebal = True
                    
                if trigger_rebal:
                    last_rebal_equity = portfolio_equity
                    last_rebal_assets = current_asset_amounts.copy()
            
        if date in eoy_dates and not is_bankrupt:
            strategy_annuals[date.year] = (portfolio_equity / prev_eoy_equity) - 1.0 if prev_eoy_equity > 0 else 0
            prev_eoy_equity = portfolio_equity
            
            if rebalance_type == "CLEC":
                for name, amount in current_asset_amounts.items():
                    if st.session_state.asset_library.get(name, {}).get("type") == "Leverage":
                        yr_ret = (amount / year_start_assets.get(name, amount)) - 1.0 if year_start_assets.get(name, amount) > 0 else 0
                        if yr_ret > 0:
                            extract = ((amount / (1+yr_ret)) * yr_ret) * 0.3
                            current_asset_amounts[name] -= extract
                            for d_name in current_asset_amounts.keys():
                                if st.session_state.asset_library.get(d_name, {}).get("type") == "Defensive": 
                                    current_asset_amounts[d_name] += extract; break
                        elif yr_ret < 0:
                            for d_name in current_asset_amounts.keys():
                                if st.session_state.asset_library.get(d_name, {}).get("type") == "Defensive":
                                    rescue = current_asset_amounts[d_name] * 0.02
                                    current_asset_amounts[d_name] -= rescue; current_asset_amounts[name] += rescue; break
                                        
            elif rebalance_type == "傳統定時":
                total_assets = sum(current_asset_amounts.values())
                for name, weight in weights_dict.items(): 
                    if name in current_asset_amounts:
                        current_asset_amounts[name] = total_assets * (weight/sum([w for k,w in weights_dict.items() if k in current_asset_amounts]))

            # 年底再平衡會將比例強制洗牌，此時將獨立追蹤的戰術帳戶清零（融入戰略池）
            if rebalance_type in ["CLEC", "傳統定時"]:
                dynamic_fund_shifted = 0.0
                tactical_borrowed_principal = 0.0
                lowest_entry_lev_index = 0.0

            if debt_mode == "恆定維持率 (增貸再投資)":
                if legal_collateral > 0 and current_reg_margin > target_margin_ratio:
                    new_loan = (legal_collateral / target_margin_ratio) - current_debt_amount
                    if new_loan > 0:
                        current_debt_amount += new_loan
                        total_assets_now = sum(current_asset_amounts.values())
                        if total_assets_now > 0:
                            for n in current_asset_amounts.keys(): current_asset_amounts[n] += new_loan * (current_asset_amounts[n] / total_assets_now)
            
            for name in current_asset_amounts: year_start_assets[name] = current_asset_amounts[name]

        # 虛擬帳戶回歸：常規計算結束後，將戰術資金加回總部位
        if dynamic_fund_shifted > 0.0 and main_lev in current_asset_amounts:
            current_asset_amounts[main_lev] += dynamic_fund_shifted

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
    final_status = f"破產 ({bankruptcy_date} {bankruptcy_reason})" if is_bankrupt else "安全存活"
    
    return {
        "總權重": initial_total_weight, "負債模式": debt_mode, "再平衡": rebalance_type, 
        "戰術加碼": tactical_mode,
        "初始借貸率": initial_debt_ratio / 100.0, "對標 Beta": sys_beta, 
        "CAGR": cagr, "最終淨值": portfolio_equity, "年化波動": real_vol,
        "最大回撤": real_mdd, "夏普值": sharpe, "卡瑪比率": calmar, "修復天數": max_recovery_days, "痛苦指數": ulcer_index,
        "累計提領": total_withdrawn, "狀態": final_status,
        "annuals": strategy_annuals, "curve": equity_curve, 
        "reg_margin_curve": reg_margin_curve, "total_margin_curve": total_margin_curve, "bond_margin_curve": bond_margin_curve, 
        "有效年數": num_years,
        "類型": "純大盤對照" if len([w for w in weights_dict.values() if w > 0]) == 1 else ("自訂戰略" if "🎯" in strategy_config.get("name", "") else "經典對照")
    }

# ==========================================
# 5. 主畫面：策略建構器
# ==========================================
st.subheader("🛠 建立自訂組合戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱", f"策略模式 {len(st.session_state.custom_strategies)+1}")
    
    col_r, col_t, col_d, col_m = st.columns(4)
    with col_r: rebal_mode = st.selectbox("常規再平衡模組", ["CLEC", "CLEC彈性(防守)", "CLEC彈性(進取)", "傳統定時", "不執行"], index=0)
    with col_t: tactical_ui = st.selectbox("外掛戰術模組", [
        "無", 
        "19/30股災加碼_賣出資產 (每次10%)", 
        "19/30股災加碼_賣出資產 (每次5%)",
        "19/30股災加碼_質押借貸 (每次10%)",
        "19/30股災加碼_質押借貸 (每次5%)"
    ], index=0)
    with col_d: debt_mode = st.selectbox("負債運用模組", ["買借死 (提領生活費)", "恆定維持率 (增貸再投資)", "無"], index=0)
    with col_m: target_margin_input = st.number_input("目標維持率 (%)", min_value=140, max_value=2000, value=600, step=50)
    
    if tactical_ui == "無":
        tactical_mode = "無"
        tactical_pct = 0.0
    else:
        if "質押借貸" in tactical_ui:
            tactical_mode = "19/30股災加碼_質押借貸(翻倍停利)"
        else:
            tactical_mode = "19/30股災加碼_賣出資產(翻倍停利)"
        tactical_pct = 10.0 if "10%" in tactical_ui else 5.0
        
    st.write("精確輸入資產權重 (%)：")
    cols = st.columns(5)
    selected_assets = {}
    asset_opts = list(st.session_state.asset_library.keys())
    
    for i in range(5):
        with cols[i]:
            asset = st.selectbox(f"部位 {i+1}", asset_opts, index=0, key=f"sel_{i}")
            weight = st.number_input(f"權重 (%)", 0.0, 300.0, 0.0, 1.0, key=f"w_{i}")
            if asset != "無 (不配置)" and weight >= 0: selected_assets[asset] = selected_assets.get(asset, 0) + weight

    if st.form_submit_button("📥 儲存策略並加入比較表") and selected_assets:
        st.session_state.custom_strategies[strat_name] = {
            "name": "🎯 " + strat_name, 
            "wts": selected_assets, 
            "rebal": rebal_mode, 
            "tactical": tactical_mode, 
            "tactical_pct": tactical_pct,
            "debt_mode": debt_mode,
            "target_margin": target_margin_input / 100.0 
        }
        st.success(f"已成功加入「{strat_name}」！")
        st.rerun()

if st.session_state.custom_strategies:
    st.markdown("#### ⚙️ 管理與檢視自訂策略")
    with st.expander("🔍 點擊查看所有自訂策略的詳細配比", expanded=True):
        for s_name, config in st.session_state.custom_strategies.items():
            wts_str = " + ".join([f"{k.split(' ')[0]} ({v}%)" for k, v in config["wts"].items()])
            margin_info = f" ｜ 目標維持率: {config.get('target_margin', 6.0) * 100:.0f}%" if config['debt_mode'] == "恆定維持率 (增貸再投資)" else ""
            tac_info = f" ｜ 戰術: {config.get('tactical', '無')} {f'({config.get('tactical_pct', 0)*100:.0f}%)' if config.get('tactical', '無') != '無' else ''}"
            st.info(f"**{s_name}**\n\n👉 配比：`{wts_str}`\n\n👉 設定：{config.get('rebal', '不執行')}{tac_info} ｜ {config.get('debt_mode', '無')}{margin_info}")

    col_del1, col_del2, col_del3 = st.columns([2, 1, 1])
    with col_del1: del_target = st.selectbox("選擇要刪除的策略", list(st.session_state.custom_strategies.keys()), label_visibility="collapsed")
    with col_del2: 
        if st.button("刪除單一策略", use_container_width=True): del st.session_state.custom_strategies[del_target]; st.rerun()
    with col_del3:
        if st.button("⚠️ 全數清空", type="primary", use_container_width=True): st.session_state.custom_strategies = {}; st.rerun()

st.markdown("---")

# ==========================================
# 6. 終極比較表與圖表渲染
# ==========================================
comp_data = []
annual_chart_data = []
curve_chart_data = []
reg_margin_chart_data = []
total_margin_chart_data = []
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
        for pt in res["total_margin_curve"]: total_margin_chart_data.append({"策略名稱": name, "日期": pt["日期"], "總擔保維持率": pt["總擔保維持率"]})
        for pt in res["bond_margin_curve"]: bond_margin_chart_data.append({"策略名稱": name, "日期": pt["日期"], "純債維持率": pt["純債維持率"]})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    
    st.markdown("### 📊 績效比較表")
    
    with st.expander("📖 點擊查看【再平衡心法】與【量化指標】白話文說明", expanded=False):
        st.markdown("""
        #### 🔄 再平衡與戰術模組說明
        * **傳統定時**：每年底固定強制將部位調回你設定的初始權重。
        * **CLEC (年度常規)**：每年底檢視。若槓桿部位賺錢，抽取 30% 獲利轉入短債；若虧損，從短債水庫抽取 2% 資金救援槓桿。
        * **CLEC 彈性 (防守/進取)**：不限年底，每日監控。總資產上漲達標就提早「獲利了結」；下跌達標啟動救援。
        * **19/30股災加碼 (翻倍停利)**：可與上述任何再平衡疊加的「戰術外掛」。當大盤從歷史高點回撤達 19% 時啟動加碼；若達 30% 再加碼一次。這筆戰術資金被**獨立隔離**不參與平時再平衡，直到其獲利翻倍(+100%) 或大盤創新高時，才將本利全數結算退回鎖潤。
            * **[賣出資產]**：賣出原型或防守資金(SGOV)來加碼。
            * **[質押借貸]**：不動用原有資金，以原有資產為擔保，直接開啟質押擴張資產負債表來借錢加碼。結算時先還清欠款，剩餘利潤灌入防守水庫。

        #### 📈 核心量化指標
        * **夏普值 (Sharpe Ratio)**：每承受 1 單位波動風險，能換取多少超額報酬。越大代表「漲得越穩」。
        * **卡瑪比率 (Calmar Ratio)**：年化報酬率 ÷ 最大回撤。衡量「每忍受 1% 跌幅，每年能賺回多少利潤」，抗跌且能漲的策略數值最高。
        * **痛苦指數 (Ulcer Index)**：不只看跌多深，還看你在水下「憋氣套牢了多久」。數值越低越好，越低代表晚上睡得越安穩。
        * **CAGR (年化報酬率)**：在本系統包含現金流（提領生活費）的模型中，此數據即等同於投資人的實質 IRR（內部報酬率）。
        """)
    
    df_comp["綜合再平衡"] = df_comp.apply(lambda x: f"{x['再平衡']} + {x['戰術加碼']}" if x['戰術加碼'] != '無' else x['再平衡'], axis=1)

    cols_order = [
        "策略名稱", "負債模式", "綜合再平衡", "狀態", 
        "初始借貸率", "對標 Beta", "CAGR", "年化波動", "最大回撤", "痛苦指數", 
        "夏普值", "卡瑪比率", "修復天數", "最終淨值", "累計提領"
    ]
    df_display = df_comp[cols_order].copy()
    
    df_display["初始借貸率"] = df_display["初始借貸率"].apply(lambda x: f"{x*100:.1f}%")
    df_display["對標 Beta"] = df_display["對標 Beta"].apply(lambda x: f"{x:.2f}")
    df_display["CAGR"] = df_display["CAGR"].apply(lambda x: f"{x*100:.2f}%")
    df_display["年化波動"] = df_display["年化波動"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["痛苦指數"] = df_display["痛苦指數"].apply(lambda x: f"{x:.2f}")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    df_display["卡瑪比率"] = df_display["卡瑪比率"].apply(lambda x: f"{x:.3f}")
    df_display["修復天數"] = df_display["修復天數"].apply(lambda x: f"{x:,} 天" if x < 9999 else "破產")
    df_display["最終淨值"] = df_display["最終淨值"].apply(lambda x: f"NT$ {x:,.0f}")
    df_display["累計提領"] = df_display["累計提領"].apply(lambda x: f"NT$ {x:,.0f}")
    
    df_display = df_display.rename(columns={
        "綜合再平衡": "再平衡與戰術",
        "對標 Beta": "對標 Beta\n(設定值)",
        "CAGR": "年化淨報酬\n(CAGR/IRR)",
        "年化波動": "年化\n波動率",
        "最大回撤": "最大回撤\n(MDD)",
        "痛苦指數": "痛苦\n指數",
        "夏普值": "夏普值\n(Sharpe)",
        "卡瑪比率": "卡瑪比率\n(Calmar)",
        "修復天數": "最大修復\n(天)",
        "累計提領": "累計提領\n生活費"
    })
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    st.markdown("### 👑 最佳配置方案評估")
    
    valid_df = df_comp[df_comp["狀態"] == "安全存活"]
    if not valid_df.empty:
        best_cagr = valid_df.loc[valid_df["CAGR"].idxmax()]
        best_equity = valid_df.loc[valid_df["最終淨值"].idxmax()]
        best_mdd = valid_df.loc[valid_df["最大回撤"].idxmax()] 
        best_recovery = valid_df.loc[valid_df["修復天數"].idxmin()]
        best_ulcer = valid_df.loc[valid_df["痛苦指數"].idxmin()]
        best_sharpe = valid_df.loc[valid_df["夏普值"].idxmax()]
        best_calmar = valid_df.loc[valid_df["卡瑪比率"].idxmax()]

        st.markdown("##### 🏆 絕對收益視角")
        c1, c2 = st.columns(2)
        with c1: st.info(f"**最高最終餘額**\n### NT$ {best_equity['最終淨值']:,.0f}\n---\n#### 🏆 冠軍策略： `{best_equity['策略名稱']}`")
        with c2: st.info(f"**最高年化回報 (CAGR)**\n### {best_cagr['CAGR']*100:.2f}%\n---\n#### 🏆 冠軍策略： `{best_cagr['策略名稱']}`")

        st.markdown("##### 🛡️ 風險與回撤視角")
        c3, c4, c5 = st.columns(3)
        with c3: st.warning(f"**最低最大回撤**\n### {best_mdd['最大回撤']*100:.2f}%\n---\n#### 🏆 冠軍策略： `{best_mdd['策略名稱']}`")
        with c4: st.warning(f"**最短修復期**\n### {best_recovery['修復天數']:,} 天\n---\n#### 🏆 冠軍策略： `{best_recovery['策略名稱']}`")
        with c5: st.warning(f"**最低痛苦指數**\n### {best_ulcer['痛苦指數']:.2f}\n---\n#### 🏆 冠軍策略： `{best_ulcer['策略名稱']}`")

        st.markdown("##### ⚖️ 風險收益比視角")
        c6, c7 = st.columns(2)
        with c6: st.success(f"**最高夏普值**\n### {best_sharpe['夏普值']:.3f}\n---\n#### 🏆 冠軍策略： `{best_sharpe['策略名稱']}`")
        with c7: st.success(f"**最高卡瑪比率**\n### {best_calmar['卡瑪比率']:.3f}\n---\n#### 🏆 冠軍策略： `{best_calmar['策略名稱']}`")
    else:
        st.error("⚠️ 壓力測試失敗：在您設定的條件下，所有策略均已宣告破產，無法產生最佳方案評估。")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💰 最終資產淨值排行 (NT$)")
        df_chart_multiple = df_comp.sort_values(by="最終淨值", ascending=True)
        max_val = df_chart_multiple["最終淨值"].max()
        color_map = {
            "純抱 SPY": "#c7c7c7", "純抱 QQQ": "#7f7f7f", 
            "經典 CLEC 433 (買借死)": "#1f77b4",
            "穩健 623 (恆定維持率 600%)": "#ff7f0e",
            "防禦 812 (年度常規 800%)": "#2ca02c",
            "彈性防禦 812 (防守型 800%)": "#059669",
            "QLD 50-50 (無負債)": "#9467bd",
            "TQQQ SGOV 333 (無負債)": "#8c564b"
        }
        custom_colors = px.colors.sequential.Reds[3:] 
        for idx, custom_name in enumerate(st.session_state.custom_strategies.keys()): color_map["🎯 " + custom_name] = custom_colors[idx % len(custom_colors)]
        
        df_chart_multiple["最終淨值_str"] = df_chart_multiple["最終淨值"].apply(lambda x: f"NT$ {x:,.0f}")
        fig_mult = px.bar(df_chart_multiple, x="最終淨值", y="策略名稱", color="策略名稱", orientation='h', text="最終淨值_str", color_discrete_map=color_map)
        fig_mult.update_layout(xaxis=dict(range=[0, max_val * 1.35]), showlegend=False)
        fig_mult.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_mult, use_container_width=True)
        
    with col2:
        st.subheader("🛡 壓力測試：最大回撤 (MDD)")
        df_chart_mdd = df_comp.sort_values(by="最大回撤", ascending=True)
        min_mdd = df_chart_mdd["最大回撤"].min()
        df_chart_mdd["最大回撤_str"] = df_chart_mdd["最大回撤"].apply(lambda x: f"{x:.2%}")
        fig_mdd = px.bar(df_chart_mdd, x="最大回撤", y="策略名稱", color="策略名稱", orientation='h', text="最大回撤_str", color_discrete_map=color_map)
        fig_mdd.update_layout(xaxis=dict(range=[min_mdd * 1.3, 0]), xaxis_tickformat='.0%', showlegend=False)
        fig_mdd.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_mdd, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🎯 策略卡瑪比率 (每單位回撤帶來的利潤)")
        df_chart_calmar = df_comp.sort_values(by="卡瑪比率", ascending=True)
        max_calmar = df_chart_calmar["卡瑪比率"].max()
        df_chart_calmar["卡瑪比率_str"] = df_chart_calmar["卡瑪比率"].apply(lambda x: f"{x:.3f}")
        fig_calmar = px.bar(df_chart_calmar, x="卡瑪比率", y="策略名稱", color="策略名稱", orientation='h', text="卡瑪比率_str", color_discrete_map=color_map)
        fig_calmar.update_layout(xaxis=dict(range=[0, max_calmar * 1.25]), showlegend=False)
        fig_calmar.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_calmar, use_container_width=True)
        
    with col4:
        st.subheader("⏳ 最長套牢修復期 (越短越好)")
        df_chart_rec = df_comp.sort_values(by="修復天數", ascending=False)
        max_rec = df_chart_rec["修復天數"].max()
        df_chart_rec["修復_str"] = df_chart_rec["修復天數"].apply(lambda x: f"{x:,} 天" if x < 9999 else "已斷頭破產")
        fig_rec = px.bar(df_chart_rec, x="修復天數", y="策略名稱", color="策略名稱", orientation='h', text="修復_str", color_discrete_map=color_map)
        fig_rec.update_layout(xaxis=dict(range=[0, max_rec * 1.35]), showlegend=False)
        fig_rec.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_rec, use_container_width=True)

    st.subheader("📈 實質金額複利成長曲線 (Log Scale)")
    if curve_chart_data:
        df_curves = pd.DataFrame(curve_chart_data)
        df_curves_sampled = df_curves.iloc[::5, :]
        fig_curves = px.line(df_curves_sampled, x="日期", y="淨值", color="策略名稱", log_y=True, color_discrete_map=color_map)
        fig_curves.update_layout(yaxis_title="資產淨值 (NT$)", xaxis_title="日期", height=450)
        st.plotly_chart(fig_curves, use_container_width=True)
        
    st.markdown("---")
    
    st.subheader("🚨 三軌風險防線追蹤圖 (維持率觀測)")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    legend_style = dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title="")
    
    with col_m1:
        st.subheader("1. 法規維持率 (僅計原型)")
        if reg_margin_chart_data:
            df_reg = pd.DataFrame(reg_margin_chart_data)
            df_reg_sampled = df_reg.iloc[::5, :]
            fig_reg = px.line(df_reg_sampled, x="日期", y="法規維持率", color="策略名稱", color_discrete_map=color_map)
            fig_reg.add_hline(y=1.4, line_dash="dash", line_color="red", annotation_text="140% 斷頭線")
            fig_reg.update_layout(yaxis_tickformat='.0%', yaxis_title="法規維持率", xaxis_title="日期", height=480, showlegend=True, legend=legend_style)
            fig_reg.update_yaxes(range=[0, 10])
            st.plotly_chart(fig_reg, use_container_width=True)
            
    with col_m2:
        st.subheader("2. 總擔保維持率 (原型+短債)")
        if total_margin_chart_data:
            df_total = pd.DataFrame(total_margin_chart_data)
            df_total_sampled = df_total.iloc[::5, :]
            fig_total = px.line(df_total_sampled, x="日期", y="總擔保維持率", color="策略名稱", color_discrete_map=color_map)
            fig_total.update_layout(yaxis_tickformat='.0%', yaxis_title="總擔保維持率", xaxis_title="日期", height=480, showlegend=True, legend=legend_style)
            fig_total.update_yaxes(range=[0, 10])
            st.plotly_chart(fig_total, use_container_width=True)
            
    with col_m3:
        st.subheader("3. 純債安全維持率 (僅計短債)")
        if bond_margin_chart_data:
            df_bond = pd.DataFrame(bond_margin_chart_data)
            df_bond_sampled = df_bond.iloc[::5, :]
            fig_bond = px.line(df_bond_sampled, x="日期", y="純債維持率", color="策略名稱", color_discrete_map=color_map)
            fig_bond.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="100% 短債枯竭線")
            fig_bond.update_layout(yaxis_tickformat='.0%', yaxis_title="純債維持率", xaxis_title="日期", height=480, showlegend=True, legend=legend_style)
            fig_bond.update_yaxes(range=[0, 10])
            st.plotly_chart(fig_bond, use_container_width=True)

    st.markdown("---")
    st.subheader("📆 歷年淨報酬率大亂鬥")
    if annual_chart_data:
        df_annual = pd.DataFrame(annual_chart_data).sort_values(by="年份")
        fig_annual = px.bar(df_annual, x="年份", y="報酬率", color="策略名稱", barmode="group", color_discrete_map=color_map)
        fig_annual.update_layout(yaxis_tickformat='.0%', yaxis_title="年度淨報酬率", xaxis_title="年份", height=450)
        st.plotly_chart(fig_annual, use_container_width=True)
        
    st.markdown("---")
    
    st.markdown("### 🤖 系統判斷與優化建議 (AI 動態尋優)")
    
    if not valid_df.empty:
        qqq_baseline = df_comp[df_comp["策略名稱"] == "純抱 QQQ"]
        has_qqq_baseline = not qqq_baseline.empty and qqq_baseline.iloc[0]["狀態"] == "安全存活"
        
        if has_qqq_baseline:
            qqq_stats = qqq_baseline.iloc[0]
            
            main_proto = ai_proto_1 if ai_proto_1 != "無 (不配置)" else "QQQ (美股大盤)"
            sec_proto = ai_proto_2 if ai_proto_2 != "無 (不配置)" else None
            main_lev = ai_lev if ai_lev != "無 (不配置)" else "QLD (美股正2)"
            main_def = ai_def if ai_def != "無 (不配置)" else "SGOV (美股超短債)"

            beta_main_proto = st.session_state.asset_library.get(main_proto, {}).get("beta", 1.0)
            beta_sec_proto = st.session_state.asset_library.get(sec_proto, {}).get("beta", 1.0) if sec_proto else 1.0
            beta_lev = st.session_state.asset_library.get(main_lev, {}).get("beta", 2.0)
            
            pool_str = f"{main_proto.split(' ')[0]}"
            if sec_proto: pool_str += f", {sec_proto.split(' ')[0]}"
            pool_str += f", {main_lev.split(' ')[0]}, {main_def.split(' ')[0]}"
            
            with st.spinner(f"⏳ 系統正在背景進行極限參數網格搜索，根據您指定的專屬資產池 [{pool_str}] 進行動態權重解算..."):
                
                ai_results = []
                target_beta_scaled = target_ai_beta * 100.0
                
                if target_ai_tactical == "無":
                    tac_mode = "無"
                    tac_pct = 0.0
                else:
                    if "質押借貸" in target_ai_tactical:
                        tac_mode = "19/30股災加碼_質押借貸(翻倍停利)"
                    else:
                        tac_mode = "19/30股災加碼_賣出資產(翻倍停利)"
                    tac_pct = 10.0 if "10%" in target_ai_tactical else 5.0
                
                search_rebals = ["CLEC", "CLEC彈性(防守)", "CLEC彈性(進取)", "傳統定時"]
                
                for rebal_ai in search_rebals:
                    max_lev_wts = (target_beta_scaled / beta_lev) + 5.0 if beta_lev > 0 else 5.0
                    for w_lev in np.arange(0.0, max_lev_wts, 5.0):
                        remaining_beta = target_beta_scaled - (w_lev * beta_lev)
                        if remaining_beta < 0: continue
                        
                        ratios = [0.0, 0.3, 0.5, 0.7, 1.0] if sec_proto else [0.0]
                        for r in ratios:
                            if sec_proto and r > 0:
                                denom = (1.0 - r) * beta_main_proto + r * beta_sec_proto
                                w_total_proto = remaining_beta / denom if denom != 0 else 0.0
                                w_sec = w_total_proto * r
                                w_main = w_total_proto * (1.0 - r)
                            else:
                                w_main = remaining_beta / beta_main_proto if beta_main_proto != 0 else 0.0
                                w_sec = 0.0
                                
                            for w_def in np.arange(0.0, 65.0, 5.0):
                                
                                w_main_rounded = round(w_main / 5.0) * 5.0
                                w_sec_rounded = round(w_sec / 5.0) * 5.0
                                w_lev_rounded = round(w_lev / 5.0) * 5.0
                                w_def_rounded = round(w_def / 5.0) * 5.0

                                debt = (w_main_rounded + w_sec_rounded + w_lev_rounded + w_def_rounded) - 100.0
                                
                                if target_ai_debt_mode == "無":
                                    if debt != 0: continue 
                                    actual_target_margin = 999.0
                                else:
                                    if debt <= 0: continue 
                                    w_legal = w_main_rounded + w_sec_rounded
                                    actual_target_margin = w_legal / debt if debt > 0 else 999.0
                                    if actual_target_margin < 4.0: continue
                                
                                tmp_wts = {}
                                if w_main_rounded > 0: tmp_wts[main_proto] = w_main_rounded
                                if w_sec_rounded > 0: tmp_wts[sec_proto] = w_sec_rounded
                                if w_lev_rounded > 0: tmp_wts[main_lev] = w_lev_rounded
                                if w_def_rounded > 0: tmp_wts[main_def] = w_def_rounded
                                
                                config = {
                                    "wts": tmp_wts,
                                    "rebal": rebal_ai,
                                    "tactical": tac_mode,
                                    "tactical_pct": tac_pct,
                                    "debt_mode": target_ai_debt_mode,
                                    "target_margin": actual_target_margin
                                }
                                
                                res = calculate_metrics(config, margin_rate, start_date, end_date, init_capital, withdraw_mode, withdraw_value)
                                res["wts_config"] = config["wts"]
                                res["rebal_config"] = config["rebal"]
                                res["tactical_config"] = config["tactical"]
                                res["tactical_pct"] = config["tactical_pct"]
                                res["debt_config"] = config["debt_mode"]
                                res["target_margin_pct"] = actual_target_margin * 100 
                                ai_results.append(res)
                        
                df_ai = pd.DataFrame(ai_results)
                df_ai_valid = df_ai[(df_ai["狀態"] == "安全存活") & (df_ai["最大回撤"] > -0.95)] if not df_ai.empty else pd.DataFrame()
                
                def format_ai_wts(row):
                    wts_str = " + ".join([f"{k.split(' ')[0]} {v}%" for k, v in row["wts_config"].items() if v > 0])
                    debt_short = "無負債" if row['debt_config'] == "無" else ("恆定維持率" if "恆定" in row['debt_config'] else "買借死")
                    margin_str = "" if row['debt_config'] == "無" else f" ｜ {debt_short} {int(row['target_margin_pct'])}%"
                    tac_str = f" + {row['tactical_config']} ({row.get('tactical_pct', 0)*100:.0f}%)" if row['tactical_config'] != '無' else ""
                    return f"**`{wts_str}`** (再平衡: {row['rebal_config']}{tac_str}{margin_str})"

                st.info(f"系統已根據您專屬的資產池 `{pool_str}` 進行解算。在**「保證絕對存活」**且**「鎖定目標 Beta = {target_ai_beta:.1f}」** 的前提下，為您找出以下實戰黃金比例：")

                if not df_ai_valid.empty:
                    # 目標 1
                    ai_best_sharpe = df_ai_valid.loc[df_ai_valid["夏普值"].idxmax()]
                    st.markdown("""<div style="background-color: rgba(74, 222, 128, 0.2); padding: 8px 15px; border-radius: 5px; color: #4ade80; font-weight: bold; margin-bottom: 10px;">💡 目標：更高的 CP 值 (漲得穩)</div>""", unsafe_allow_html=True)
                    if ai_best_sharpe["夏普值"] > qqq_stats["夏普值"]:
                        st.markdown(f"相比純抱 QQQ (夏普值 {qqq_stats['夏普值']:.3f})，系統找到以下最佳平衡點：\n* **✨ AI 推薦最優配比**：{format_ai_wts(ai_best_sharpe)}\n* **模擬成效**：成功將夏普值推升至 **{ai_best_sharpe['夏普值']:.3f}** (年化報酬 {ai_best_sharpe['CAGR']*100:.2f}%)。")
                    else:
                        st.markdown(f"系統算盡此條件下的所有組合，發現 `純抱 QQQ` (夏普值 {qqq_stats['夏普值']:.3f}) 的風險收益比仍難以被超越。以下是系統為您找出的**亞軍配比**：\n* **✨ AI 推薦次優配比**：{format_ai_wts(ai_best_sharpe)}\n* **模擬成效**：夏普值達 **{ai_best_sharpe['夏普值']:.3f}** (年化報酬 {ai_best_sharpe['CAGR']*100:.2f}%)。")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # 目標 2
                    ai_best_mdd = df_ai_valid.loc[df_ai_valid["最大回撤"].idxmax()]
                    st.markdown("""<div style="background-color: rgba(250, 204, 21, 0.2); padding: 8px 15px; border-radius: 5px; color: #facc15; font-weight: bold; margin-bottom: 10px;">🛡️ 目標：更低的最大回撤 (睡得安穩)</div>""", unsafe_allow_html=True)
                    if ai_best_mdd["最大回撤"] > qqq_stats["最大回撤"]:
                        st.markdown(f"若您覺得純 QQQ 的跌幅 ({qqq_stats['最大回撤']*100:.2f}%) 太高，系統為您找到以下最佳鐵壁防禦：\n* **✨ AI 推薦最優配比**：{format_ai_wts(ai_best_mdd)}\n* **模擬成效**：成功將極限回撤壓低至 **{ai_best_mdd['最大回撤']*100:.2f}%**，痛苦指數降至 **{ai_best_mdd['痛苦指數']:.2f}**。")
                    else:
                        st.markdown(f"在此 Beta 區間內，`純抱 QQQ` ({qqq_stats['最大回撤']*100:.2f}%) 的防禦力為榜首。以下是系統找出的**最強防禦亞軍**：\n* **✨ AI 推薦次優配比**：{format_ai_wts(ai_best_mdd)}\n* **模擬成效**：成功將極限回撤控制在 **{ai_best_mdd['最大回撤']*100:.2f}%** (痛苦指數 {ai_best_mdd['痛苦指數']:.2f})。")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # 目標 3
                    ai_best_equity = df_ai_valid.loc[df_ai_valid["最終淨值"].idxmax()]
                    st.markdown("""<div style="background-color: rgba(248, 113, 113, 0.2); padding: 8px 15px; border-radius: 5px; color: #f87171; font-weight: bold; margin-bottom: 10px;">🔥 目標：極致的最終淨值 (賺得比 QQQ 更多)</div>""", unsafe_allow_html=True)
                    if ai_best_equity["最終淨值"] > qqq_stats["最終淨值"]:
                        st.markdown(f"在您設定的限制下，系統發現能創造更高絕對獲利的配置：\n* **✨ AI 推薦最優配比**：{format_ai_wts(ai_best_equity)}\n* **模擬成效**：將最終淨值推升至 **NT$ {ai_best_equity['最終淨值']:,.0f}** (勝過 QQQ 的 NT$ {qqq_stats['最終淨值']:,.0f})！")
                    else:
                        st.markdown(f"系統推演後確認，`純抱 QQQ` 仍是這段時間內的獲利王 (最終淨值 NT$ {qqq_stats['最終淨值']:,.0f})。以下是**獲利亞軍配比**：\n* **✨ AI 推薦次優配比**：{format_ai_wts(ai_best_equity)}\n* **模擬成效**：最終淨值達 **NT$ {ai_best_equity['最終淨值']:,.0f}** (年化報酬 {ai_best_equity['CAGR']*100:.2f}%)。")
                else:
                    st.error(f"⚠️ 系統在進行背景網格尋優時，發現在此區間內，無法找到符合安全存活的策略。建議調降 Beta 目標或選擇恆定維持率模式。")
        else:
            st.info("⚠️ 若要啟用 AI 網格尋優對比，請確保 `純抱 QQQ` 策略在您的回測區間內處於安全存活狀態。")
