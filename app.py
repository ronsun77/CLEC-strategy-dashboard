import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Pro 級質押戰略戰情室", layout="wide")

# ==========================================
# 1. 初始化 Session State (短期記憶體)
# ==========================================
RISK_FREE_RATE = 0.04

# 預設資產字典
if 'asset_library' not in st.session_state:
    st.session_state.asset_library = {
        "無 (不配置)": {"ret": 0.0, "beta": 0.0, "vol": 0.0},
        "0050/006208 (台股大盤)": {"ret": 0.10, "beta": 1.0, "vol": 0.16},
        "00713 (台股高息)": {"ret": 0.10, "beta": 0.65, "vol": 0.12},
        "QQQ/00662 (美股原型)": {"ret": 0.15, "beta": 1.0, "vol": 0.18},
        "QLD/00670L (美股正2)": {"ret": 0.26, "beta": 2.0, "vol": 0.36},
        "00631L (台股正2)": {"ret": 0.20, "beta": 2.0, "vol": 0.32},
        "SGOV/00865B (短債)": {"ret": 0.045, "beta": 0.0, "vol": 0.02},
        "現金": {"ret": 0.0, "beta": 0.0, "vol": 0.0}
    }

# 預設經典策略 (做為 Benchmark)
if 'benchmark_strategies' not in st.session_state:
    st.session_state.benchmark_strategies = {
        "經典 CLEC 433 (無借貸)": {"QQQ/00662 (美股原型)": 40.0, "QLD/00670L (美股正2)": 30.0, "SGOV/00865B (短債)": 30.0},
        "穩健 622 (無借貸)": {"QQQ/00662 (美股原型)": 60.0, "QLD/00670L (美股正2)": 20.0, "SGOV/00865B (短債)": 20.0},
        "效率 623 (防禦型質押)": {"QQQ/00662 (美股原型)": 60.0, "QLD/00670L (美股正2)": 20.0, "SGOV/00865B (短債)": 30.0},
        "攻擊 722 (攻擊型質押)": {"QQQ/00662 (美股原型)": 70.0, "QLD/00670L (美股正2)": 20.0, "SGOV/00865B (短債)": 20.0}
    }

# 儲存使用者建立的自訂策略清單
if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}


# ==========================================
# 2. 核心計算引擎 (財務工程邏輯)
# ==========================================
def calculate_metrics(weights_dict, margin_rate):
    total_weight = sum(weights_dict.values())
    debt_ratio = max(0, total_weight - 100.0)
    
    asset_return, system_beta, est_vol = 0.0, 0.0, 0.0
    for name, weight in weights_dict.items():
        if name in st.session_state.asset_library and weight > 0:
            asset = st.session_state.asset_library[name]
            w_pct = weight / 100.0
            asset_return += asset["ret"] * w_pct
            system_beta += asset["beta"] * w_pct
            est_vol += asset["vol"] * w_pct
            
    debt_cost = (debt_ratio / 100.0) * margin_rate
    net_return = asset_return - debt_cost
    sharpe = (net_return - RISK_FREE_RATE) / est_vol if est_vol > 0 else 0
    
    return {
        "總權重": f"{total_weight:.1f}%",
        "實質負債": f"{debt_ratio:.1f}%",
        "淨報酬率": net_return,
        "系統 Beta": system_beta,
        "波動率": est_vol,
        "夏普值": sharpe
    }

# ==========================================
# 3. 介面渲染：側邊欄 (全局設定與資產庫)
# ==========================================
st.sidebar.title("⚙️ 系統設定與資產庫")
margin_rate_input = st.sidebar.number_input("質押借貸利率 (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
margin_rate = margin_rate_input / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("➕ 新增自訂資產標的")
with st.sidebar.form("add_asset_form"):
    new_name = st.text_input("標的名稱 (如: 00981A)")
    new_ret = st.number_input("預估年化報酬率 (%)", value=5.0, step=0.1)
    new_beta = st.number_input("系統 Beta 值", value=1.0, step=0.1)
    new_vol = st.number_input("預估波動率 (%)", value=10.0, step=0.1)
    
    submitted = st.form_submit_button("新增至資產庫")
    if submitted and new_name:
        st.session_state.asset_library[new_name] = {
            "ret": new_ret / 100.0, "beta": new_beta, "vol": new_vol / 100.0
        }
        st.success(f"已新增 {new_name}！")

st.sidebar.markdown("---")
with st.sidebar.expander("查看當前資產庫 (Asset Library)"):
    # 顯示為 dataframe 方便檢視
    lib_df = pd.DataFrame(st.session_state.asset_library).T
    lib_df.columns = ["預期報酬", "Beta", "波動率"]
    st.dataframe(lib_df.style.format("{:.2f}"))

# ==========================================
# 4. 主畫面：策略建構器 (Strategy Builder)
# ==========================================
st.title("📊 Pro 級多重質押戰略戰情室")
st.markdown("突破限制！精確定義你的多組投資組合，並與經典 CLEC 進行終極排行。")

# 建立自訂策略的表單
st.subheader("🛠️ 建立新的自訂戰略")
with st.form("create_strategy_form"):
    strat_name = st.text_input("自訂策略名稱 (如: 冬眠期防守陣型)", "我的新戰略")
    
    # 允許使用者最多配置 5 檔標的 (可以透過增加 range 放大)
    st.write("精確輸入資產權重 (總和小於 100% 為閒置，大於 100% 視為質押)")
    cols = st.columns(5)
    selected_assets = {}
    
    asset_options = list(st.session_state.asset_library.keys())
    
    for i in range(5):
        with cols[i]:
            # 預設帶入一些標的，方便快速操作
            default_index = i if i < len(asset_options) else 0
            asset = st.selectbox(f"部位 {i+1}", asset_options, index=default_index, key=f"sel_{i}")
            weight = st.number_input(f"權重 (%)", min_value=0.0, max_value=300.0, value=0.0, step=1.0, key=f"w_{i}")
            if asset != "無 (不配置)" and weight > 0:
                selected_assets[asset] = selected_assets.get(asset, 0) + weight

    submit_strat = st.form_submit_button("📥 儲存策略並加入比較表")
    if submit_strat:
        if not selected_assets:
            st.warning("請至少配置一項大於 0% 的資產！")
        else:
            st.session_state.custom_strategies[strat_name] = selected_assets
            st.success(f"策略 '{strat_name}' 已成功加入！")

# 提供清除自訂策略的按鈕
if st.session_state.custom_strategies:
    if st.button("🗑️ 清空所有自訂策略"):
        st.session_state.custom_strategies = {}
        st.rerun()

st.markdown("---")

# ==========================================
# 5. 主畫面：終極比較表 (Master Comparison)
# ==========================================
st.subheader("🏆 戰略終極比較表 (Benchmark vs Custom)")

comparison_data = []

# 計算並加入 Benchmark 策略
for name, weights in st.session_state.benchmark_strategies.items():
    res = calculate_metrics(weights, margin_rate)
    res["策略名稱"] = name
    res["類型"] = "經典對照組"
    comparison_data.append(res)

# 計算並加入 User Custom 策略
for name, weights in st.session_state.custom_strategies.items():
    res = calculate_metrics(weights, margin_rate)
    res["策略名稱"] = "🎯 " + name
    res["類型"] = "我的自訂戰略"
    comparison_data.append(res)

df_comp = pd.DataFrame(comparison_data)

if not df_comp.empty:
    # 重新排列欄位順序
    cols_order = ["類型", "策略名稱", "總權重", "實質負債", "淨報酬率", "系統 Beta", "波動率", "夏普值"]
    df_comp = df_comp[cols_order]
    
    # 建立用來顯示的格式化 DataFrame
    df_display = df_comp.copy()
    df_display["淨報酬率"] = df_display["淨報酬率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["系統 Beta"] = df_display["系統 Beta"].apply(lambda x: f"{x:.2f}")
    df_display["波動率"] = df_display["波動率"].apply(lambda x: f"{x*100:.2f}%")
    df_display["夏普值"] = df_display["夏普值"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 繪製夏普值排行長條圖
    st.subheader("📈 策略效率排行 (夏普值)")
    df_chart = df_comp.sort_values(by="夏普值", ascending=True)
    
    fig = px.bar(
        df_chart, 
        x="夏普值", 
        y="策略名稱", 
        color="類型",
        orientation='h',
        text="夏普值",
        color_discrete_map={"經典對照組": "#83C9FF", "我的自訂戰略": "#FF4B4B"}
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(showlegend=True, xaxis_title="夏普值 (越高越好)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
