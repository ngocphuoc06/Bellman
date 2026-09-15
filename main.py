import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import *
from data_manager import generate_walmart_synthetic_data, split_data
from eda_analyzer import perform_eda, fit_parameters
from dp_model import BellmanInventoryDP
from evaluator import perform_time_series_cv, simulate_policy

def main():
    print("===== BƯỚC 1: THU THẬP DỮ LIỆU ====")
    df_raw = generate_walmart_synthetic_data(days=TOTAL_DAYS, n_true=6.9, p_true=0.25)
    print(f"Đã sinh tập dữ liệu với {len(df_raw)} dòng.")
    
    print("\n=== BƯỚC 2: DATA SPLITTING ===")
    train_data, test_data = split_data(df_raw)
    print(f"Tập huấn luyện (Train): {len(train_data)} ngày.")
    print(f"Tập kiểm thử (Test): {len(test_data)} ngày (đã chặn Data Leakage).")
    
    print("\n================ BƯỚC 3: EDA (KHÁM PHÁ DỮ LIỆU) ================")
    train_mean, train_var = perform_eda(train_data)
    
    print("\n================ BƯỚC 4: TIỀN XỬ LÝ & TÌM THAM SỐ (FIT) ================")
    n_estimated, p_estimated = fit_parameters(train_mean, train_var)
    
    print("\n================ BƯỚC 5: HUẤN LUYỆN MÔ HÌNH BELLMAN DP ================")
    print(">> Đang chạy Kiểm định chéo chuỗi thời gian (Rolling-window 5-fold CV)...")
    perform_time_series_cv(train_data)
    
    print("\n>> Đang huấn luyện mô hình cuối cùng trên toàn bộ tập Train...")
    dp_model = BellmanInventoryDP(MAX_CAPACITY, FIXED_ORDER_COST, UNIT_ORDER_COST, 
                                  HOLDING_COST, SHORTAGE_COST, DISCOUNT_FACTOR, TOLERANCE,
                                  n_estimated, p_estimated)
    dp_model.value_iteration(verbose=True)
    s_reorder, S_target = dp_model.get_optimal_policy()
    print(f"=> Mô hình trả về Chính sách (s, S): Đặt hàng khi Tồn kho <= {s_reorder}. Mục tiêu nạp đầy tới {S_target}.")
    
    print("\n================ BƯỚC 6: FINAL EVALUATION (MÔ PHỎNG KIỂM THỬ) ================")
    total_cost, ordering, holding, shortage = simulate_policy(test_data, s_reorder, S_target)
    print(f"Đã chạy mô phỏng qua {len(test_data)} ngày của tập test (Hold-out).")
    print(f"Chi tiết chi phí:")
    print(f" - Phí Đặt hàng: ${ordering:,.2f}")
    print(f" - Phí Lưu kho : ${holding:,.2f}")
    print(f" - Phí Cạn kho : ${shortage:,.2f}")
    print(f"TỔNG CHI PHÍ VẬN HÀNH THỰC TẾ: ${total_cost:,.2f}")

if __name__ == "__main__":
    main()
