import pandas as pd
import streamlit as st

# 假設這是在 Streamlit 環境中運行
# 此處為您優化後的每日迭代邏輯核心區塊
def run_simulation_loop(current_asset_amounts, current_debt_amount, master_prices, master_peak_price, 
                        tactical_mode, tactical_pct, lev_index, lev_index_history, 
                        dynamic_fund_shifted, triggered_19, triggered_30, lowest_entry_lev_index,
                        main_lev, main_proto, rebalance_type, eoy_dates, date, portfolio_equity, 
                        last_rebal_equity, last_rebal_assets, year_start_assets, debt_mode, 
                        withdraw_mode, withdraw_value, total_withdrawn, equity_curve, 
                        reg_margin_curve, total_margin_curve, bond_margin_curve, 
                        strategy_annuals, prev_eoy_equity, target_margin_ratio):

    # 1. 生活費提領邏輯
    withdrawal_amount = 0
    if debt_mode == "買借死 (提領生活費)":
        if withdraw_mode == "總資產百分比 (%)":
            withdrawal_amount = (sum(current_asset_amounts.values()) * withdraw_value) / 252.0
        else:
            withdrawal_amount = withdraw_value / 252.0
            
    current_debt_amount += withdrawal_amount
    total_withdrawn += withdrawal_amount
        
    # 2. 戰術外掛邏輯 (獨立虛擬帳戶操作)
    if tactical_mode == "19/30股災加碼(翻倍停利)":
        if not master_prices.empty and date in master_prices.index:
            current_master_val = master_prices.loc[date]
            if current_master_val > master_peak_price:
                master_peak_price = current_master_val
                # 創新高時，強制結算戰術部位
                if dynamic_fund_shifted > 0.0:
                    current_asset_amounts[main_lev] -= dynamic_fund_shifted
                    current_asset_amounts[main_proto] += dynamic_fund_shifted
                    dynamic_fund_shifted = 0.0
                    triggered_19 = False; triggered_30 = False; lowest_entry_lev_index = 0.0
            
            master_dd = (master_peak_price - current_master_val) / master_peak_price if master_peak_price > 0 else 0
            
            # 翻倍停利
            if triggered_19 and lowest_entry_lev_index > 0:
                if lev_index >= lowest_entry_lev_index * 2.0:
                    current_asset_amounts[main_lev] -= dynamic_fund_shifted
                    current_asset_amounts[main_proto] += dynamic_fund_shifted
                    dynamic_fund_shifted = 0.0
                    triggered_19 = False; triggered_30 = False; lowest_entry_lev_index = 0.0
            
            # 股災進場
            total_assets_now = sum(current_asset_amounts.values())
            if master_dd >= 0.19 and not triggered_19:
                shift_val = min(total_assets_now * tactical_pct, current_asset_amounts[main_proto])
                current_asset_amounts[main_proto] -= shift_val
                current_asset_amounts[main_lev] += shift_val
                dynamic_fund_shifted += shift_val
                lowest_entry_lev_index = lev_index 
                triggered_19 = True
            if master_dd >= 0.30 and not triggered_30:
                shift_val = min(total_assets_now * tactical_pct, current_asset_amounts[main_proto])
                current_asset_amounts[main_proto] -= shift_val
                current_asset_amounts[main_lev] += shift_val
                dynamic_fund_shifted += shift_val
                lowest_entry_lev_index = lev_index 
                triggered_30 = True

    # 3. 風險控管與法規維持率計算
    legal_collateral = sum([amount for n, amount in current_asset_amounts.items() if st.session_state.asset_library.get(n, {}).get("type") == "Prototype"])
    
    # 強制還債機制
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

    # 4. 虛擬帳戶隔離：將戰術資金從常規部位暫時抽離，以免干擾再平衡計算
    if dynamic_fund_shifted > 0.0 and main_lev in current_asset_amounts:
        current_asset_amounts[main_lev] = max(0.0, current_asset_amounts[main_lev] - dynamic_fund_shifted)

    # 5. 常規再平衡邏輯 (運作在「剩餘」資產上)
    # ... (此處接續您原有的 CLEC 或定時平衡邏輯) ...

    # 6. 虛擬帳戶回歸：再平衡結束後，把戰術資金加回來
    if dynamic_fund_shifted > 0.0 and main_lev in current_asset_amounts:
        current_asset_amounts[main_lev] += dynamic_fund_shifted

    return current_asset_amounts, current_debt_amount, dynamic_fund_shifted, triggered_19, triggered_30, lowest_entry_lev_index
