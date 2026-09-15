import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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


def predict_ordering_decisions(data, s_reorder, S_target, n_est, p_est):
    """
    Mô phỏng chính sách (s, S) qua từng ngày và phân loại quyết định đặt hàng.
    
    Returns:
        y_true: Nhãn thực tế — ngày nào thực sự CẦN đặt hàng (1) hay không (0).
                Cách xác định: nếu không đặt hàng mà tồn kho cuối ngày < 0 → cần đặt.
        y_pred: Quyết định của chính sách (s,S) — ngày nào mô hình ĐÃ đặt hàng (1) hay không (0).
        demand_actual: Nhu cầu thực tế từng ngày.
        demand_predicted: Nhu cầu kỳ vọng từ phân phối NegBin (hằng số = n*(1-p)/p).
    """
    current_inventory = S_target
    y_true = []
    y_pred = []
    demand_actual = []
    
    # Kỳ vọng nhu cầu từ phân phối Negative Binomial: E[D] = n*(1-p)/p
    expected_demand = n_est * (1 - p_est) / p_est
    demand_predicted = [expected_demand] * len(data)
    
    for actual_demand in data['Demand']:
        demand_actual.append(actual_demand)
        
        # Chính sách (s,S) quyết định: có đặt hàng không?
        did_order = 1 if current_inventory <= s_reorder else 0
        y_pred.append(did_order)
        
        # Xác định nhãn thực tế: có CẦN đặt hàng không?
        # Nếu không đặt hàng, tồn kho sau khi trừ demand sẽ âm → đáng lẽ phải đặt
        inventory_if_no_order = current_inventory - actual_demand
        should_have_ordered = 1 if inventory_if_no_order < 0 else 0
        y_true.append(should_have_ordered)
        
        # Cập nhật tồn kho theo chính sách thực tế
        if did_order:
            current_inventory = S_target
        current_inventory -= actual_demand
        current_inventory = max(current_inventory, 0)
        current_inventory = min(current_inventory, MAX_CAPACITY)
    
    return (np.array(y_true), np.array(y_pred), 
            np.array(demand_actual), np.array(demand_predicted))


def calculate_rmsse(actual, predicted, train_data):
    """
    Tính Root Mean Squared Scaled Error (RMSSE).
    Mẫu số là naive forecast error trên tập train (dùng giá trị ngày hôm trước làm dự báo).
    
    RMSSE = sqrt( mean((actual - predicted)^2) / mean((train[t] - train[t-1])^2) )
    """
    train_demands = train_data['Demand'].values
    
    # Naive forecast error trên train: sai số khi dùng y_{t-1} dự báo y_t
    naive_errors = np.diff(train_demands)
    denominator = np.mean(naive_errors ** 2)
    
    if denominator == 0:
        denominator = 1e-6  # Tránh chia cho 0
    
    numerator = np.mean((actual - predicted) ** 2)
    rmsse = np.sqrt(numerator / denominator)
    return rmsse


def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """
    Vẽ và lưu ma trận nhầm lẫn (Confusion Matrix) ra file PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    labels = ['Không đặt hàng (0)', 'Đặt hàng (1)']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           title=f'Ma trận Nhầm lẫn — {dataset_name}',
           ylabel='Nhãn Thực tế (Actual)',
           xlabel='Dự đoán Mô hình (Predicted)')
    
    # Hiển thị giá trị số trong từng ô
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    
    fig.tight_layout()
    filename = f'confusion_matrix_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"   Đã lưu ma trận nhầm lẫn ra file '{filename}'.")
    return filename


def evaluate_model(data, s_reorder, S_target, n_est, p_est, train_data_for_scale, dataset_name="Test"):
    """
    Bước 7: Đánh giá toàn diện mô hình.
    Tính và in: Accuracy, F1 Score, RMSSE, Confusion Matrix.
    
    Args:
        data: DataFrame chứa cột 'Demand' để đánh giá.
        s_reorder, S_target: Chính sách (s, S) từ Bellman DP.
        n_est, p_est: Tham số NegBin đã ước lượng.
        train_data_for_scale: Tập Train dùng làm mẫu số cho RMSSE.
        dataset_name: Tên tập dữ liệu (để hiển thị).
    """
    print(f"\n   --- Đánh giá trên tập {dataset_name} ({len(data)} ngày) ---")
    
    # 1. Chạy mô phỏng và phân loại
    y_true, y_pred, demand_actual, demand_predicted = predict_ordering_decisions(
        data, s_reorder, S_target, n_est, p_est)
    
    # 2. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"   Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    
    # 3. F1 Score (xử lý trường hợp không có class positive)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"   F1 Score  : {f1:.4f}")
    
    # 4. RMSSE (so sánh demand kỳ vọng vs demand thực tế)
    rmsse = calculate_rmsse(demand_actual, demand_predicted, train_data_for_scale)
    print(f"   RMSSE     : {rmsse:.4f}")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"   Ma trận nhầm lẫn (Confusion Matrix):")
    print(f"                        Predicted")
    print(f"                   Không ĐH    Đặt hàng")
    print(f"   Actual Không ĐH   {cm[0][0]:>5}      {cm[0][1]:>5}")
    if cm.shape[0] > 1:
        print(f"   Actual Đặt hàng   {cm[1][0]:>5}      {cm[1][1]:>5}")
    
    # 6. Lưu biểu đồ confusion matrix
    plot_confusion_matrix(y_true, y_pred, dataset_name)
    
    return acc, f1, rmsse
