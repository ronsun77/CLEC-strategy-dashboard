import streamlit as st
import pandas as pd
import plotly.express as px

# --- 參數設定 ---
st.set_page_config(page_title="進階質押戰略總覽", layout="wide")

# 定義資產屬性字典 (Return, Beta, Volatility)
ASSETS = {
    "0050/006208 (台股大盤)": {"ret": 0.10, "beta": 1.0, "vol": 0.16},
    "00713 (台股高息)": {"ret": 0.10, "beta": 0.65, "vol": 0.12},
    "00662/QQQ (美股科技)": {"ret": 0.15, "beta": 1.0, "vol": 0.18},
    "00670L/QLD (美股正2)": {"ret": 0.26, "beta": 2.0, "vol": 0.36},
    "00631L (台股正2)": {"ret": 0.20, "beta": 2.0, "vol": 0.32},
    "00865B/SGOV (短債)": {"ret": 0.045, "beta": 0.0, "vol": 0.02},
    "現金": {"ret": 0.0, "beta": 0.0, "vol": 0.0}
}
RISK_FREE_RATE = 0.04

# --- 側邊欄：使用者輸入 ---
st.sidebar.header("⚙️ 投資組合參數設定")
margin_rate = st.sidebar.slider("質押借貸利率 (%)", 1.0, 4.0, 2.5, 0.1) / 100

st.sidebar.subheader("1. 原型 ETF (雙重選擇)")
proto1_name = st.sidebar.selectbox("原型選擇 1", ["0050/006208 (台股大盤)", "00713 (台股高息)"])
proto1_weight = st.sidebar.slider(f"{proto1_name} 權重 (%)", 0, 100, 40) / 100

proto2_name = st.sidebar.selectbox("原型選擇 2", ["00662/QQQ (美股科技)"])
proto2_weight = st.sidebar.slider(f"{proto2_name} 權重 (%)", 0, 100, 20) / 100

st.sidebar.subheader("2. 槓桿型 ETF (雙重選擇)")
lev1_name = st.sidebar.selectbox("槓桿選擇 1", ["00670L/QLD (美股正2)"])
lev1_weight = st.sidebar.slider(f"{lev1_name} 權重 (%)", 0, 100, 20) / 100

lev2_name = st.sidebar.selectbox("槓桿選擇 2", ["00631L (台股正2)"])
lev2_weight = st.sidebar.slider(f"{lev2_name} 權重 (%)", 0, 100, 0) / 100

st.sidebar.subheader("3. 防守型資產")
def_name = st.sidebar.selectbox("防守選擇", ["00865B/SGOV (短債)", "現金"])
def_weight = st.sidebar.slider(f"{def_name} 權重 (%)", 0, 100, 30) / 100

# --- 後台計算邏輯 ---
total_weight = proto1_weight + proto2_weight + lev1_weight + lev2_weight + def_weight
debt_ratio = max(0, total_weight - 1.0) # 超過 100% 的部分視為負債

# 加權計算 (假設總本金為 1)
asset_return = (
    proto1_weight * ASSETS[proto1_name]["ret"] +
    proto2_weight * ASSETS[proto2_name]["ret"] +
    lev1_weight * ASSETS[lev1_name]["ret"] +
    lev2_weight * ASSETS[lev2_name]["ret"] +
    def_weight * ASSETS[def_name]["ret"]
)
debt_cost = debt_ratio * margin_rate
net_return = asset_return - debt_cost

system_beta = (
    proto1_weight * ASSETS[proto1_name]["beta"] +
    proto2_weight * ASSETS[proto2_name]["beta"] +
    lev1_weight * ASSETS[lev1_name]["beta"] +
    lev2_weight * ASSETS[lev2_name]["beta"] +
    def_weight * ASSETS[def_name]["beta"]
)

# 簡化版波動率估算 (線性加權)
est_vol = (
    proto1_weight * ASSETS[proto1_name]["vol"] +
    proto2_weight * ASSETS[proto2_name]["vol"] +
    lev1_weight * ASSETS[lev1_name]["vol"] +
    lev2_weight * ASSETS[lev2_name]["vol"] +
    def_weight * ASSETS[def_name]["vol"]
)

sharpe_ratio = (net_return - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0

# --- 主畫面：數據展示 ---
st.title("📊 CLEC 質押戰略總覽建構器")

col1, col2, col3, col4 = st.columns(4)
col1.metric("預估年化報酬率", f"{net_return*100:.2f}%")
col2.metric("系統總曝險 (Beta)", f"{system_beta:.2f}")
col3.metric("預估年化波動率", f"{est_vol*100:.2f}%")
col4.metric("夏普值 (Sharpe)", f"{sharpe_ratio:.2f}")

st.info(f"💡 **槓桿狀態分析**：當前總配置比例為 **{total_weight*100:.0f}%**。實質質押負債為 **{debt_ratio*100:.0f}%**，每年預估利息成本為 **{debt_cost*100:.2f}%**。")

# --- 圓餅圖繪製 ---
data = {
    "資產類別": [proto1_name, proto2_name, lev1_name, lev2_name, def_name],
    "權重": [proto1_weight, proto2_weight, lev1_weight, lev2_weight, def_weight]
}
df = pd.DataFrame(data)
df = df[df["權重"] > 0] # 只顯示大於 0 的項目

fig = px.pie(df, values="權重", names="資產類別", title="資產配置分佈圖 (含槓桿部位)", hole=0.4)
st.plotly_chart(fig, use_container_width=True)
