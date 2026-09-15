from eda_analyzer import fit_parameters
from dp_model import BellmanInventoryDP
from config import (MAX_CAPACITY, FIXED_ORDER_COST, UNIT_ORDER_COST, 
                    HOLDING_COST, SHORTAGE_COST, DISCOUNT_FACTOR, TOLERANCE)

def simulate_policy(test_data, s_reorder, S_target):
    """
    Giả lập môi trường thời gian thực (Simulation environment) 
    để tính toán tổng chi phí vận hành.
    """
    current_inventory = S_target
    total_cost = 0
    total_holding = 0
    total_shortage = 0
    total_ordering = 0
    
    for actual_demand in test_data['Demand']:
        # Kiểm tra đặt hàng
        if current_inventory <= s_reorder:
            order_qty = S_target - current_inventory
            cost = FIXED_ORDER_COST + UNIT_ORDER_COST * order_qty
            total_ordering += cost
            total_cost += cost
            current_inventory += order_qty
            
        current_inventory -= actual_demand
        
        if current_inventory > 0:
            cost = HOLDING_COST * current_inventory
            total_holding += cost
            total_cost += cost
            current_inventory = min(current_inventory, MAX_CAPACITY)
        else:
            cost = SHORTAGE_COST * abs(current_inventory)
            total_shortage += cost
            total_cost += cost
            current_inventory = 0
            
    return total_cost, total_ordering, total_holding, total_shortage

def perform_time_series_cv(train_data, n_splits=5):
    """
    Bước 5: Huấn luyện & Tinh chỉnh mô hình (Training & Cross-Validation)
    Thực hiện Time-Series Cross-Validation (Rolling window) linh hoạt theo độ dài dữ liệu.
    """
    total_len = len(train_data)
    val_size = max(10, total_len // (n_splits + 2))
    
    print(f"{'Fold':<8} {'Train Window':<15} {'Val Window':<15} {'Mean':<6} {'Var':<6} {'Policy (s,S)':<15} {'Val Cost':<10}")
    print("-" * 80)
    
    for i in range(1, n_splits + 1):
        train_end = total_len - (n_splits - i + 1) * val_size
        val_end = train_end + val_size
        
        if train_end <= 10:
            continue
            
        fold_train = train_data.iloc[:train_end]
        fold_val = train_data.iloc[train_end:val_end]
        
        mean = fold_train['Demand'].mean()
        var = fold_train['Demand'].var()
        
        n_est, p_est = fit_parameters(mean, var, verbose=False)
        
        dp = BellmanInventoryDP(MAX_CAPACITY, FIXED_ORDER_COST, UNIT_ORDER_COST, 
                                HOLDING_COST, SHORTAGE_COST, DISCOUNT_FACTOR, TOLERANCE,
                                n_est, p_est)
        dp.value_iteration(verbose=False)
        s, S = dp.get_optimal_policy()
        
        total_cost, _, _, _ = simulate_policy(fold_val, s, S)
        
        print(f"Fold {i:<3} 1 - {train_end:<11} {train_end+1} - {val_end:<8} {mean:<6.2f} {var:<6.2f} s={s:<2}, S={S:<6} ${total_cost:,.2f}")
