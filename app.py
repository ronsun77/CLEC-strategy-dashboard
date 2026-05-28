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
        
        # 轉換為以年為單位的收盤價字典
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
# 3. 核心計算引擎 (解耦並實裝絕對金額提領壓力測試)
# ==========================================
def calculate_metrics(strategy_config, margin_rate, align_inception=True, target_margin_ratio=6.0, init_capital=10000000.0, withdraw_mode="總資產百分比 (%)", withdraw_value=0.025):
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
    portfolio_equity = init_capital 
    current_debt_amount = (initial_debt_ratio / 100.0) * init_capital
    current_asset_amounts = {name: (weight/100.0) * init_capital for name, weight in weights_dict.items()}
    
    equity_curve = [] 
    total_withdrawn = 0.0
    is_bankrupt = False
    bankruptcy_reason = "存活"
    bankruptcy_year = ""

    for year in valid_years:
        if is_bankrupt:
            strategy_annuals[year] = 0; equity_curve.append({"年份": year, "淨值": 0.0, "負債": current_debt_amount})
            continue
            
        year_start_assets = sum(current_asset_amounts.values())
        
        # 1. 資產增長
        for name, amount in current_asset_amounts.items():
            if name in st.session_state.asset_library and amount > 0:
                ret = st.session_state.asset_library[name].get("annuals", {}).get(year, 0)
                if ret == 0 and st.session_state.asset_library[name].get("type") == "Defensive": ret = 0.02
                current_asset_amounts[name] = amount * (1 + ret)
                
        # 2. 利息計算
        interest_cost = current_debt_amount * margin_rate
        current_debt_amount += interest_cost
        
        # 3. 買借死動態生活費提領
        withdrawal_amount = 0
        if debt_mode == "買借死 (提領生活費)":
            if withdraw_mode == "總資產百分比 (%)":
                withdrawal_amount = sum(current_asset_amounts.values()) * withdraw_value
            else:
                withdrawal_amount = withdraw_value # 固定金額提領
            
            current_debt_amount += withdrawal_amount
            total_withdrawn += withdrawal_amount
            
        year_end_assets = sum(current_asset_amounts.values())
        portfolio_equity = year_end_assets - current_debt_amount
        
        # 4. 風控檢查：維持率檢查 (原型資產 / 負債)
        collateral_val = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library[n].get("type") == "Prototype"])
        current_margin_ratio = collateral_val / current_debt_amount if current_debt_amount > 0 else float('inf')
        
        if portfolio_equity <= 0:
            portfolio_equity = 0; is_bankrupt = True; bankruptcy_reason = "淨值歸零"; bankruptcy_year = year
            strategy_annuals[year] = -1.0; equity_curve.append({"年份": year, "淨值": 0.0, "負債": current_debt_amount})
            continue
            
        if current_margin_ratio < 1.4: # 跌破 140% 強制斷頭
            portfolio_equity = 0; is_bankrupt = True; bankruptcy_reason = "維持率低於140%斷頭"; bankruptcy_year = year
            strategy_annuals[year] = -1.0; equity_curve.append({"年份": year, "淨值": 0.0, "負債": current_debt_amount})
            continue
            
        net_year_return = (portfolio_equity - (year_start_assets - (current_debt_amount - interest_cost - withdrawal_amount))) / (year_start_assets - (current_debt_amount - interest_cost - withdrawal_amount)) if year_start_assets > 0 else 0
        strategy_annuals[year] = net_year_return
        equity_curve.append({"年份": year, "淨值": portfolio_equity, "負債": current_debt_amount})
        
        # 5. 再平衡模組
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

        # 6. 恆定維持率模組 (動態增貸再投資)
        if debt_mode == "恆定維持率 (增貸再投資)":
            if collateral_val > 0 and current_margin_ratio > target_margin_ratio:
                new_loan = (collateral_val / target_margin_ratio) - current_debt_amount
                if new_loan > 0:
                    current_debt_amount += new_loan
                    total_assets_now = sum(current_asset_amounts.values())
                    if total_assets_now > 0:
                        for n in current_asset_amounts.keys(): current_asset_amounts[n] += new_loan * (current_asset_amounts[n] / total_assets_now)
                
    num_years = len(strategy_annuals)
    cagr = ((portfolio_equity / init_capital) ** (1 / num_years)) - 1 if num_years > 0 and not is_bankrupt and portfolio_equity > 0 else 0
    avg_annual_ret = sum(strategy_annuals.values()) / num_years if num_years > 0 else 0
    sharpe = (avg_annual_ret - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    
    # 計算最大回撤
    df_curve = pd.DataFrame(equity_curve)
    if not df_curve.empty and portfolio_equity > 0:
        df_curve["最高淨值"] = df_curve["淨值"].cummax()
        df_curve["水下回撤"] = (df_curve["淨值"] / df_curve["最高淨值"]) - 1.0
        real_mdd = df_curve["水下回撤"].min()
    else:
        real_mdd = -1.0 if is_bankrupt else 0.0

    calmar = cagr / abs(real_mdd) if real_mdd != 0 else 0
    
    return {
        "總權重": initial_total_weight, "負債模式": debt_mode, "再平衡": rebalance_type, "系統 Beta": sys_beta, 
        "年化淨報酬率(CAGR)": cagr, "最終淨值": portfolio_equity, "年化波動率": est_vol,
        "最大回撤": real_mdd, "夏普值": sharpe, "卡瑪比率": calmar, "累計提領生活費": total_withdrawn,
        "狀態": f"破產 ({bankruptcy_year}年 {bankruptcy_reason})" if is_bankrupt else "安全存活",
        "annuals": strategy_annuals, "curve": equity_curve, "有效年數": num_years
    }

# ==========================================
# 4. 介面渲染：側邊欄 (全面升級壓力測試參數)
# ==========================================
st.sidebar.title("⚙️ 全局設定與智能防呆")
new_lookback = st.sidebar.slider("歷史資料抓取範圍 (年)", 5, 30, st.session_state.lookback_years, 1)
if new_lookback != st.session_state.lookback_years:
    st.session_state.lookback_years = new_lookback; st.cache_data.clear()
    st.session_state.asset_library = load_default_assets(new_lookback); st.rerun()

align_inception = st.sidebar.checkbox("強制作為公平比較 (對齊最晚掛牌日)", value=True)
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

# 💥 新增：買借死提領壓力測試控制區
st.sidebar.markdown("---")
st.sidebar.subheader("💰 買借死提領現金流設定")
init_capital = st.sidebar.number_input("初始試算本金 (元)", min_value=100000, value=10000000, step=1000000)
withdraw_mode = st.sidebar.selectbox("提領生活費模式", ["總資產百分比 (%)", "固定金額 (元)"])
if withdraw_mode == "總資產百分比 (%)":
    withdraw_value = st.sidebar.number_input("年提領比例 (%)", min_value=0.0, max_value=20.0, value=2.5, step=0.1) / 100.0
else:
    withdraw_value = st.sidebar.number_input("年提領金額 (元)", min_value=0, value=250000, step=50000)

# ==========================================
# 5. 主畫面：策略建構器 (支援無限策略疊加)
# ==========================================
st.title("📊 頂級 CLEC 質押策略回測戰情室")

st.subheader("🛠   建立自訂組合戰略")
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

if st.session_state.custom_strategies:
    st.markdown("#### 🗑   管理已儲存的自訂策略")
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
curve_chart_data = []

# 計算對照組與自訂策略
all_strategies = {}
for k, v in st.session_state.benchmark_strategies.items(): all_strategies[k] = v
for k, v in st.session_state.custom_strategies.items(): all_strategies[v["name"]] = v

for name, config in all_strategies.items():
    res = calculate_metrics(config, margin_rate, align_inception, init_capital=init_capital, withdraw_mode=withdraw_mode, withdraw_value=withdraw_value)
    res["策略名稱"] = name
    comp_data.append(res)
    for idx, pt in enumerate(res["curve"]): 
        curve_chart_data.append({"策略名稱": name, "年份": pt["年份"], "淨值": pt["淨值"], "負債": pt["負債"]})

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    # 重新調整欄位順序，突顯「狀態」與「累計提領金額」
    cols_order = ["策略名稱", "負債模式", "再平衡", "狀態", "年化淨報酬率(CAGR)", "最終淨值", "累計提領生活費", "最大回撤", "夏普值", "卡瑪比率", "有效年數"]
    df_display = df_comp[cols_order].copy()
    
    df_display["年化淨報酬率(CAGR)"] = df_display["年化淨報酬率(CAGR)"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最終淨值"] = df_display["最終淨值"].apply(lambda x: f"NT$ {x:,.0f}")
    df_display["累計提領生活費"] = df_display["累計提領生活費"].apply(lambda x: f"NT$ {x:,.0f}")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    df_display["卡瑪比率"] = df_display["卡瑪比率"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 📈 繪製資產成長與負債對比曲線
    st.markdown("---")
    st.subheader("📈 20年實質金額複利與負債擴張曲線 (壓力測試)")
    if curve_chart_data:
        df_curves = pd.DataFrame(curve_chart_data)
        fig_curves = px.line(df_curves, x="年份", y="淨值", color="策略名稱", title="帳戶實質淨值走勢 (元)")
        st.plotly_chart(fig_curves, use_container_width=True)
