import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import numpy as np
import datetime

st.set_page_config(page_title="Pro 級質押戰略戰情室", layout="wide")
RISK_FREE_RATE = 0.04

# ==========================================
# 1. 自動抓取市場數據函數 (Yahoo Finance)
# ==========================================
def fetch_asset_data(ticker):
    try:
        # 抓取過去 5 年的日線資料
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=5*365)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, "找不到該代號的數據，請確認代號 (台股請加 .TW，如 006208.TW)"
        
        # 使用收盤價計算
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0] # 處理 yfinance 多欄位問題
            
        daily_returns = close_prices.pct_change().dropna()
        
        # 計算年化報酬率 (CAGR)
        total_return = close_prices.iloc[-1] / close_prices.iloc[0]
        years = (close_prices.index[-1] - close_prices.index[0]).days / 365.25
        ann_return = (total_return ** (1 / years)) - 1
        
        # 計算年化波動率
        ann_vol = daily_returns.std() * np.sqrt(252)
        
        # 計算最大回撤 (Max Drawdown)
        rolling_max = close_prices.cummax()
        drawdown = (close_prices / rolling_max) - 1.0
        mdd = drawdown.min()
        
        # Beta 預設為 1.0 (實務上需與大盤做共變異數計算，此處為保持效能先設常數)
        beta = 1.0 
        
        return {"ret": float(ann_return), "beta": beta, "vol": float(ann_vol), "mdd": float(mdd)}, "成功"
    except Exception as e:
        return None, f"抓取失敗: {str(e)}"

# ==========================================
# 2. 初始化 Session State
# ==========================================
if 'asset_library' not in st.session_state:
    st.session_state.asset_library = {
        "無 (不配置)": {"ret": 0.0, "beta": 0.0, "vol": 0.0, "mdd": 0.0},
        "QQQ (美股大盤)": {"ret": 0.15, "beta": 1.0, "vol": 0.18, "mdd": -0.33},
        "QLD (美股正2)": {"ret": 0.26, "beta": 2.0, "vol": 0.36, "mdd": -0.60},
        "00713 (台股高息)": {"ret": 0.10, "beta": 0.65, "vol": 0.12, "mdd": -0.15},
        "SGOV (短債)": {"ret": 0.045, "beta": 0.0, "vol": 0.02, "mdd": -0.01}
    }

if 'benchmark_strategies' not in st.session_state:
    st.session_state.benchmark_strategies = {
        "經典 CLEC 433": {"QQQ (美股大盤)": 40.0, "QLD (美股正2)": 30.0, "SGOV (短債)": 30.0},
        "穩健 623 (防禦質押)": {"QQQ (美股大盤)": 60.0, "QLD (美股正2)": 20.0, "SGOV (短債)": 30.0}
    }

if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}

# ==========================================
# 3. 核心計算引擎 (加入 MDD 計算)
# ==========================================
def calculate_metrics(weights_dict, margin_rate):
    total_weight = sum(weights_dict.values())
    debt_ratio = max(0, total_weight - 100.0)
    
    asset_ret, sys_beta, est_vol, est_mdd = 0.0, 0.0, 0.0, 0.0
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            asset = st.session_state.asset_library[name]
            w_pct = weight / 100.0
            asset_ret += asset["ret"] * w_pct
            sys_beta += asset["beta"] * w_pct
            est_vol += asset["vol"] * w_pct
            est_mdd += asset.get("mdd", 0) * w_pct # 質押槓桿會自動放大回撤
            
    debt_cost = (debt_ratio / 100.0) * margin_rate
    net_return = asset_ret - debt_cost
    sharpe = (net_return - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    
    return {
        "總權重": total_weight, "實質負債": debt_ratio,
        "淨報酬率": net_return, "最大回撤": est_mdd,
        "波動率": est_vol, "夏普值": sharpe
    }

# ==========================================
# 4. 側邊欄：智能抓取與資產庫
# ==========================================
st.sidebar.title("⚙️ 系統設定與智能資產庫")
margin_rate = st.sidebar.number_input("質押借貸利率 (%)", 0.0, 10.0, 2.5, 0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 智能抓取新增資產")
st.sidebar.caption("輸入代號 (如 AAPL, QQQ, 0050.TW)，系統將自動抓取 5 年數據並計算參數。")

with st.sidebar.form("auto_fetch_form"):
    ticker_input = st.text_input("Yahoo Finance 股票代號")
    fetch_btn = st.form_submit_button("自動抓取並新增")
    
    if fetch_btn and ticker_input:
        with st.spinner(f"正在抓取 {ticker_input} 的歷史數據..."):
            data, msg = fetch_asset_data(ticker_input.upper())
            if data:
                st.session_state.asset_library[ticker_input.upper()] = data
                st.success(f"成功新增 {ticker_input.upper()}！\n(報酬: {data['ret']*100:.1f}%, 回撤: {data['mdd']*100:.1f}%)")
            else:
                st.error(msg)

with st.sidebar.expander("查看當前資產庫參數"):
    lib_df = pd.DataFrame(st.session_state.asset_library).T
    lib_df.columns = ["年化報酬", "Beta", "波動率", "最大回撤"]
    st.dataframe(lib_df.style.format("{:.2%}"))

# ==========================================
# 5. 主畫面：策略建構器
# ==========================================
st.title("📊 頂級質押戰略戰情室")

st.subheader("🛠️ 建立新的自訂戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱", "自訂新戰略")
    st.write("精確輸入資產權重 (%)，加總超過 100% 視為質押借款：")
    
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
# 6. 終極比較表與圖表 (含最大回撤)
# ==========================================
st.subheader("🏆 戰略終極比較表")
comp_data = []

for name, wts in st.session_state.benchmark_strategies.items():
    res = calculate_metrics(wts, margin_rate)
    res["策略名稱"] = name; res["類型"] = "經典對照"
    comp_data.append(res)

for name, wts in st.session_state.custom_strategies.items():
    res = calculate_metrics(wts, margin_rate)
    res["策略名稱"] = "🎯 " + name; res["類型"] = "自訂戰略"
    comp_data.append(res)

df_comp = pd.DataFrame(comp_data)

if not df_comp.empty:
    cols_order = ["類型", "策略名稱", "總權重", "實質負債", "淨報酬率", "最大回撤", "夏普值"]
    df_comp = df_comp[cols_order]
    
    df_display = df_comp.copy()
    df_display["總權重"] = df_display["總權重"].apply(lambda x: f"{x:.0f}%")
    df_display["實質負債"] = df_display["實質負債"].apply(lambda x: f"{x:.0f}%")
    df_display["淨報酬率"] = df_display["淨報酬率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["最大回撤"] = df_display["最大回撤"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 繪製最大回撤長條圖
    st.subheader("🛡️ 壓力測試：最大回撤 (越接近 0 越好)")
    df_chart = df_comp.sort_values(by="最大回撤", ascending=True)
    
    fig = px.bar(
        df_chart, x="最大回撤", y="策略名稱", color="類型", orientation='h', text="最大回撤",
        color_discrete_map={"經典對照": "#54A24B", "自訂戰略": "#E45756"}
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(xaxis_tickformat='.0%', xaxis_title="最大回撤 (MDD)")
    st.plotly_chart(fig, use_container_width=True)
