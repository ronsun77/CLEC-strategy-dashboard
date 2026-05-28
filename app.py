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
# 1. 自動抓取市場數據函數
# ==========================================
def fetch_asset_data(ticker, lookback_years=20):
    try:
        ticker = ticker.strip().upper()

        if re.match(r'^\d+[A-Z]*$', ticker) and '.TW' not in ticker and '.TWO' not in ticker:
            ticker = ticker + '.TW'

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=lookback_years * 365)

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        if data.empty:
            return None, f"找不到 {ticker} 的數據。"

        close_prices = data['Close']

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        daily_returns = close_prices.pct_change().dropna()

        total_return = close_prices.iloc[-1] / close_prices.iloc[0]

        years = (
            close_prices.index[-1] - close_prices.index[0]
        ).days / 365.25

        ann_return = (total_return ** (1 / years)) - 1

        ann_vol = daily_returns.std() * np.sqrt(252)

        rolling_max = close_prices.cummax()

        drawdown = (close_prices / rolling_max) - 1.0

        mdd = drawdown.min()

        annual_data = close_prices.resample('YE').last()

        annual_returns = {
            str(year.year): float(val)
            for year, val in annual_data.pct_change().dropna().items()
        }

        inception_year = (
            min([int(y) for y in annual_returns.keys()])
            if annual_returns
            else datetime.date.today().year
        )

        return {
            "ret": float(ann_return),
            "beta": 1.0,
            "vol": float(ann_vol),
            "mdd": float(mdd),
            "annuals": annual_returns,
            "inception_year": inception_year,
            "prices": close_prices,
            "type": (
                "Leverage"
                if "L" in ticker or "正2" in ticker
                else (
                    "Defensive"
                    if "債" in ticker or "SHY" in ticker
                    else "Prototype"
                )
            )
        }, f"成功抓取 {ticker}！"

    except Exception as e:
        return None, f"抓取失敗: {str(e)}"


# ==========================================
# 2. 初始化真實數據
# ==========================================
@st.cache_data(ttl=86400)
def load_default_assets(lookback=20):

    lib = {
        "無 (不配置)": {
            "ret": 0.0,
            "beta": 0.0,
            "vol": 0.0,
            "mdd": 0.0,
            "annuals": {},
            "inception_year": 0,
            "type": "None",
            "prices": pd.Series()
        },
        "現金": {
            "ret": 0.0,
            "beta": 0.0,
            "vol": 0.0,
            "mdd": 0.0,
            "annuals": {},
            "inception_year": 0,
            "type": "Defensive",
            "prices": pd.Series()
        }
    }

    defaults = {
        "SPY": "SPY (標普大盤)",
        "QQQ": "QQQ (美股大盤)",
        "QLD": "QLD (美股正2)",
        "00713.TW": "00713 (台股高息)",
        "SHY": "SHY (1-3年短債)"
    }

    for ticker, display_name in defaults.items():

        data, _ = fetch_asset_data(ticker, lookback)

        if data:

            if "QLD" in ticker:
                data["beta"] = 2.0
                data["type"] = "Leverage"

            if "00713" in ticker:
                data["beta"] = 0.65
                data["type"] = "Prototype"

            if "SHY" in ticker:
                data["beta"] = 0.0
                data["type"] = "Defensive"

            lib[display_name] = data

    return lib


if 'lookback_years' not in st.session_state:
    st.session_state.lookback_years = 20

if 'asset_library' not in st.session_state:
    st.session_state.asset_library = load_default_assets(
        st.session_state.lookback_years
    )

if 'benchmark_strategies' not in st.session_state:

    st.session_state.benchmark_strategies = {

        "純抱 SPY": {
            "wts": {
                "SPY (標普大盤)": 100.0
            },
            "rebal": "不執行",
            "debt_mode": "無"
        },

        "純抱 QQQ": {
            "wts": {
                "QQQ (美股大盤)": 100.0
            },
            "rebal": "不執行",
            "debt_mode": "無"
        },

        "經典 CLEC 433 (買借死)": {
            "wts": {
                "QQQ (美股大盤)": 40.0,
                "QLD (美股正2)": 30.0,
                "SHY (1-3年短債)": 30.0
            },
            "rebal": "CLEC",
            "debt_mode": "買借死 (提領生活費)"
        },

        "穩健 623 (恆定增貸)": {
            "wts": {
                "QQQ (美股大盤)": 60.0,
                "QLD (美股正2)": 20.0,
                "SHY (1-3年短債)": 30.0
            },
            "rebal": "CLEC",
            "debt_mode": "恆定維持率 (增貸再投資)"
        }
    }

if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}


# ==========================================
# 3. 核心計算引擎
# ==========================================
def calculate_metrics(
    strategy_config,
    margin_rate,
    align_inception=True,
    target_margin_ratio=6.0,
    init_capital=10000000.0,
    withdraw_mode="固定金額 (元)",
    withdraw_value=600000.0
):

    weights_dict = strategy_config["wts"]
    rebalance_type = strategy_config["rebal"]
    debt_mode = strategy_config["debt_mode"]

    initial_total_weight = sum(weights_dict.values())

    initial_debt_ratio = max(
        0,
        initial_total_weight - 100.0
    )

    sys_beta = 0.0
    est_vol = 0.0

    all_years = set()

    max_inception_year = 0

    for name, weight in weights_dict.items():

        if (
            name in st.session_state.asset_library
            and weight > 0
        ):

            asset = st.session_state.asset_library[name]

            all_years.update(
                asset.get("annuals", {}).keys()
            )

            if (
                asset.get("inception_year", 0)
                > max_inception_year
                and name not in ["無 (不配置)", "現金"]
            ):
                max_inception_year = asset.get(
                    "inception_year",
                    0
                )

            sys_beta += asset["beta"] * (
                weight / 100.0
            )

            est_vol += asset["vol"] * (
                weight / 100.0
            )

    if align_inception and max_inception_year > 0:

        valid_years = sorted([
            y for y in all_years
            if int(y) >= max_inception_year
        ])

    else:
        valid_years = sorted(all_years)

    portfolio_equity = init_capital

    current_debt_amount = (
        initial_debt_ratio / 100.0
    ) * init_capital

    current_asset_amounts = {
        name: (weight / 100.0) * init_capital
        for name, weight in weights_dict.items()
    }

    equity_curve = []

    reg_margin_curve = []

    bond_margin_curve = []

    total_withdrawn = 0.0

    is_bankrupt = False

    bankruptcy_reason = "存活"

    bankruptcy_year = ""

    for year in valid_years:

        if is_bankrupt:

            equity_curve.append({
                "年份": year,
                "淨值": 0.0,
                "負債": current_debt_amount
            })

            reg_margin_curve.append({
                "年份": year,
                "法規維持率": 0.0
            })

            bond_margin_curve.append({
                "年份": year,
                "純債維持率": 0.0
            })

            continue

        # ======================================
        # 1. 資產增長
        # ======================================

        for name, amount in current_asset_amounts.items():

            if (
                name in st.session_state.asset_library
                and amount > 0
            ):

                ret = st.session_state.asset_library[
                    name
                ].get(
                    "annuals",
                    {}
                ).get(year, 0)

                if (
                    ret == 0
                    and st.session_state.asset_library[
                        name
                    ].get("type") == "Defensive"
                ):
                    ret = 0.02

                current_asset_amounts[name] = (
                    amount * (1 + ret)
                )

        # ======================================
        # 2. 利息與提領
        # ======================================

        interest_cost = (
            current_debt_amount * margin_rate
        )

        current_debt_amount += interest_cost

        withdrawal_amount = 0

        if debt_mode == "買借死 (提領生活費)":

            if withdraw_mode == "總資產百分比 (%)":

                withdrawal_amount = (
                    sum(current_asset_amounts.values())
                    * withdraw_value
                )

            else:
                withdrawal_amount = withdraw_value

        current_debt_amount += withdrawal_amount

        total_withdrawn += withdrawal_amount

        year_end_assets = sum(
            current_asset_amounts.values()
        )

        portfolio_equity = (
            year_end_assets - current_debt_amount
        )

        # ======================================
        # 3. 維持率
        # ======================================

        legal_collateral = sum([
            amount
            for n, amount in current_asset_amounts.items()
            if st.session_state.asset_library[n].get(
                "type"
            ) in ["Prototype", "Defensive"]
        ])

        bond_collateral = sum([
            amount
            for n, amount in current_asset_amounts.items()
            if st.session_state.asset_library[n].get(
                "type"
            ) == "Defensive"
        ])

        current_reg_margin = (
            legal_collateral / current_debt_amount
            if current_debt_amount > 0
            else 10.0
        )

        current_bond_margin = (
            bond_collateral / current_debt_amount
            if current_debt_amount > 0
            else 10.0
        )

        display_reg = min(current_reg_margin, 10.0)

        display_bond = min(current_bond_margin, 10.0)

        # ======================================
        # 4. 破產檢查
        # ======================================

        if portfolio_equity <= 0:

            portfolio_equity = 0

            is_bankrupt = True

            bankruptcy_reason = "淨值歸零"

            bankruptcy_year = year

        if (
            current_reg_margin < 1.4
            and current_debt_amount > 0
        ):

            portfolio_equity = 0

            is_bankrupt = True

            bankruptcy_reason = "法規維持率低於140%斷頭"

            bankruptcy_year = year

        equity_curve.append({
            "年份": year,
            "淨值": portfolio_equity,
            "負債": current_debt_amount
        })

        reg_margin_curve.append({
            "年份": year,
            "法規維持率": display_reg
        })

        bond_margin_curve.append({
            "年份": year,
            "純債維持率": display_bond
        })

        # ======================================
        # 5. 再平衡
        # ======================================

        if rebalance_type == "CLEC":

            for name, amount in current_asset_amounts.items():

                if st.session_state.asset_library[name].get(
                    "type"
                ) == "Leverage":

                    ret = st.session_state.asset_library[
                        name
                    ].get(
                        "annuals",
                        {}
                    ).get(year, 0)

                    if ret > 0:

                        extract = (
                            ((amount / (1 + ret)) * ret)
                            * 0.3
                        )

                        current_asset_amounts[name] -= extract

                        for d_name in current_asset_amounts.keys():

                            if st.session_state.asset_library[
                                d_name
                            ].get("type") == "Defensive":

                                current_asset_amounts[d_name] += extract

                                break

                    elif ret < 0:

                        for d_name in current_asset_amounts.keys():

                            if st.session_state.asset_library[
                                d_name
                            ].get("type") == "Defensive":

                                rescue = (
                                    current_asset_amounts[d_name]
                                    * 0.02
                                )

                                current_asset_amounts[d_name] -= rescue

                                current_asset_amounts[name] += rescue

                                break

        elif rebalance_type == "傳統定時":

            total_assets = sum(
                current_asset_amounts.values()
            )

            for name, weight in weights_dict.items():

                current_asset_amounts[name] = (
                    total_assets
                    * (weight / sum(weights_dict.values()))
                )

        # ======================================
        # 6. 恆定維持率增貸
        # ======================================

        if debt_mode == "恆定維持率 (增貸再投資)":

            if (
                legal_collateral > 0
                and current_reg_margin > target_margin_ratio
            ):

                new_loan = (
                    (legal_collateral / target_margin_ratio)
                    - current_debt_amount
                )

                if new_loan > 0:

                    current_debt_amount += new_loan

                    total_assets_now = sum(
                        current_asset_amounts.values()
                    )

                    if total_assets_now > 0:

                        for n in current_asset_amounts.keys():

                            current_asset_amounts[n] += (
                                new_loan
                                * (
                                    current_asset_amounts[n]
                                    / total_assets_now
                                )
                            )

    # ==========================================
    # 年度報酬率（修復 KeyError）
    # ==========================================
    annuals = {}

    for i in range(1, len(equity_curve)):

        prev_equity = equity_curve[i - 1]["淨值"]

        curr_equity = equity_curve[i]["淨值"]

        if prev_equity > 0:

            annual_ret = (
                curr_equity / prev_equity
            ) - 1

            annuals[str(
                equity_curve[i]["年份"]
            )] = annual_ret

    # ==========================================
    # 指標計算
    # ==========================================
    num_years = len(valid_years)

    if (
        num_years > 0
        and not is_bankrupt
        and portfolio_equity > 0
    ):

        cagr = (
            (portfolio_equity / init_capital)
            ** (1 / num_years)
        ) - 1

    else:
        cagr = 0

    df_curve = pd.DataFrame(equity_curve)

    if (
        not df_curve.empty
        and portfolio_equity > 0
    ):

        df_curve["最高淨值"] = (
            df_curve["淨值"].cummax()
        )

        df_curve["水下回撤"] = (
            df_curve["淨值"]
            / df_curve["最高淨值"]
        ) - 1.0

        real_mdd = df_curve["水下回撤"].min()

        real_vol = (
            df_curve["淨值"]
            .pct_change()
            .std()
        )

        sharpe = (
            (cagr - RISK_FREE_RATE)
            / real_vol
            if real_vol > 0
            else 0
        )

        max_recovery_years = 0

        current_drop_years = 0

        for idx, row in df_curve.iterrows():

            if row["水下回撤"] < 0:

                current_drop_years += 1

            else:

                if current_drop_years > max_recovery_years:

                    max_recovery_years = current_drop_years

                current_drop_years = 0

        if current_drop_years > max_recovery_years:

            max_recovery_years = current_drop_years

        max_recovery_days = int(
            max_recovery_years * 365
        )

    else:

        real_mdd = -1.0

        real_vol = est_vol

        sharpe = 0

        max_recovery_days = 9999

    calmar = (
        cagr / abs(real_mdd)
        if real_mdd != 0
        else 0
    )

    return {

        "總權重": initial_total_weight,

        "負債模式": debt_mode,

        "再平衡": rebalance_type,

        "系統 Beta": sys_beta,

        "年化淨報酬率(CAGR)": cagr,

        "最終淨值": portfolio_equity,

        "年化波動率": real_vol,

        "最大回撤": real_mdd,

        "夏普值": sharpe,

        "卡瑪比率": calmar,

        "最大修復天數": max_recovery_days,

        "累計提領生活費": total_withdrawn,

        "狀態": (
            f"破產 ({bankruptcy_year} {bankruptcy_reason})"
            if is_bankrupt
            else "安全存活"
        ),

        # 修復 KeyError
        "annuals": annuals,

        "curve": equity_curve,

        "reg_margin_curve": reg_margin_curve,

        "bond_margin_curve": bond_margin_curve,

        "有效年數": num_years,

        "類型": (
            "純大盤對照"
            if len([
                w for w in weights_dict.values()
                if w > 0
            ]) == 1
            else (
                "自訂戰略"
                if "🎯" in strategy_config.get("name", "")
                else "經典對照"
            )
        )
    }
