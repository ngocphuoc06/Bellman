# config.py

# ==========================================
# CẤU HÌNH HỆ THỐNG & SIÊU THAM SỐ (DP)
# ==========================================

# Tham số không gian trạng thái
MAX_CAPACITY = 40               # Ngưỡng vật lý của không gian trạng thái (M = 40 theo tài liệu)

# Chi phí vận hành
FIXED_ORDER_COST = 10           # Chi phí đặt hàng cố định (K)
UNIT_ORDER_COST = 0             # Chi phí biến đổi (c)
HOLDING_COST = 0.5              # Chi phí lưu kho (h)
SHORTAGE_COST = 20              # Chi phí phạt cạn kho (p)

# Tham số hội tụ
DISCOUNT_FACTOR = 0.95          # Hệ số chiết khấu (gamma)
TOLERANCE = 1e-4                # Ngưỡng hội tụ (theta - 10^-4 theo tài liệu)

# Dữ liệu
TOTAL_DAYS = 1913               # Tổng số ngày trong bộ dữ liệu M5
TEST_DAYS = 100                 # Hold-out set (Tập kiểm thử) 100 ngày cuối cùng
