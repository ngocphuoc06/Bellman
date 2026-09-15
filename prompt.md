# Prompt tái tạo dự án Bellman Inventory DP

Hãy xây dựng hệ thống quản trị hàng tồn kho ngẫu nhiên (Stochastic Inventory Control) bằng **Quy hoạch động Bellman (Value Iteration)** với phân phối **Negative Binomial**, theo kiến trúc modular Python. Ngôn ngữ hiển thị: **Tiếng Việt**. Thư viện: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`.

---

## Cấu trúc dự án — 6 file

```
config.py | data_manager.py | eda_analyzer.py | dp_model.py | evaluator.py | main.py
```

---

## 1. `config.py` — Hằng số

```python
MAX_CAPACITY = 40          # Không gian trạng thái S = {0..40}
FIXED_ORDER_COST = 10      # K
UNIT_ORDER_COST = 0        # c
HOLDING_COST = 0.5         # h
SHORTAGE_COST = 20         # p
DISCOUNT_FACTOR = 0.95     # gamma
TOLERANCE = 1e-4           # theta hội tụ
TOTAL_DAYS = 1913
TEST_DAYS = 100
```

---

## 2. `data_manager.py` — 4 hàm

### `fetch_online_retail_real_data(stock_code='20675')`
- Tải CSV từ URL `https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv` (encoding latin1).
- Lọc `Quantity > 0`, lọc theo `StockCode`, gom nhóm `Quantity` theo ngày → cột `['Date', 'Demand']`.
- Tạo chuỗi ngày liên tục (`date_range`), fill ngày trống = 0.
- Cache vào `data/real_demand_daily.csv`, lần sau đọc từ cache.
- Fallback sang `generate_walmart_synthetic_data()` nếu lỗi.

### `load_custom_csv_data(filepath, date_col='Date', demand_col='Demand')`
- Đọc CSV người dùng, chuẩn hóa 2 cột `Date`, `Demand`.

### `generate_walmart_synthetic_data(days=1913, n_true=6.9, p_true=0.25)`
- Sinh dữ liệu từ `scipy.stats.nbinom.rvs(n_true, p_true, size=days)`.

### `split_data(df, test_days=100)`
- Chia chronological (không random). Nếu `len(df) <= test_days*2` thì `train_size = 80%`, ngược lại `train_size = len(df) - test_days`.

---

## 3. `eda_analyzer.py` — 2 hàm

### `perform_eda(data)`
- Tính `mean`, `var` của cột `Demand`.
- In kết quả. Nếu `var > mean` → in cảnh báo Overdispersion.
- Vẽ histogram lưu ra `demand_histogram.png`.
- Return `(mean, var)`.

### `fit_parameters(mean, var, verbose=True)`
- Moment Matching cho Negative Binomial: `p = mean / var`, `n = mean² / (var - mean)`.
- Nếu `var <= mean` thì set `var = mean + 0.01`.
- Return `(n, p)`.

---

## 4. `dp_model.py` — Class `BellmanInventoryDP`

### `__init__(self, max_capacity, K, c, h, p, gamma, theta, demand_n, demand_p)`
- Trạng thái: `np.arange(M+1)`.
- Cache xác suất: `demand_probs = [nbinom.pmf(d, n, p) for d in range(M*2)]`.
- Khởi tạo `V = zeros(M+1)`, `policy = zeros(M+1, int)`.

### `expected_cost(state, action)`
- `inventory_after = state + action`
- `order_cost = K + c*action if action > 0 else 0`
- Duyệt `demand 0..max_demand`: tính `expected_holding`, `expected_shortage`, `expected_future_value` (dùng `self.V[next_state]`).
- `next_state = min(max(inventory_after - demand, 0), M)`
- Return `order_cost + holding + shortage + gamma * future_value`.

### `value_iteration(verbose=False)`
- Lặp đến khi `max|V_new - V| < theta`.
- Với mỗi state s: thử mọi action `a ∈ {0..M-s}`, chọn action có cost min → `V[s]`, `policy[s]`.

### `get_optimal_policy()`
- `S_target = policy[0]` (lượng đặt khi kho rỗng).
- `s_reorder` = state lớn nhất có `policy[state] > 0`.
- Return `(s_reorder, S_target)`.

---

## 5. `evaluator.py` — 6 hàm

### `simulate_policy(data, s_reorder, S_target)`
- Bắt đầu `inventory = S_target`.
- Mỗi ngày: nếu `inventory <= s_reorder` → đặt hàng lên S_target (+ chi phí K). Trừ demand. Nếu `inventory > 0` → holding cost. Nếu `<= 0` → shortage cost, reset = 0. Clamp ≤ M.
- Return `(total_cost, ordering, holding, shortage)`.

### `perform_time_series_cv(train_data, n_splits=5)`
- Rolling-window CV: `val_size = max(10, len // (n_splits+2))`.
- Mỗi fold: `fold_train = [:train_end]`, `fold_val = [train_end:val_end]`.
- Fit parameters → tạo BellmanDP → value_iteration → get policy → simulate trên fold_val → in bảng kết quả.

### `predict_ordering_decisions(data, s_reorder, S_target, n_est, p_est)`
- Mô phỏng policy từng ngày. Xây dựng binary classification:
  - `y_pred = 1` nếu `inventory <= s_reorder` (mô hình đặt hàng).
  - `y_true = 1` nếu `inventory - actual_demand < 0` (đáng lẽ phải đặt).
- `demand_predicted = n*(1-p)/p` (kỳ vọng NegBin, hằng số).
- Cập nhật inventory: nếu đặt hàng → set = S_target, trừ demand, clamp [0, M].
- Return `(y_true, y_pred, demand_actual, demand_predicted)` dạng numpy array.

### `calculate_rmsse(actual, predicted, train_data)`
- `RMSSE = sqrt(MSE(actual, predicted) / mean(diff(train_demand)²))`.
- Tránh chia 0: denominator = max(denominator, 1e-6).

### `plot_confusion_matrix(y_true, y_pred, dataset_name)`
- Vẽ heatmap `imshow` với colorbar, hiển thị số trong ô, lưu `confusion_matrix_{name}.png`.

### `evaluate_model(data, s_reorder, S_target, n_est, p_est, train_data_for_scale, dataset_name)`
- Gọi `predict_ordering_decisions` → tính `accuracy_score`, `f1_score(zero_division=0)`, `calculate_rmsse`, in confusion matrix dạng bảng text, gọi `plot_confusion_matrix`.
- Return `(accuracy, f1, rmsse)`.

---

## 6. `main.py` — Pipeline 7 bước

```
def main(use_real_data=True, custom_csv_path=None):
```

| Bước | Mô tả | Hàm gọi |
|------|-------|---------|
| 1 | Thu thập dữ liệu | `fetch_online_retail_real_data` / `load_custom_csv` / `generate_synthetic` |
| 2 | EDA trên dữ liệu thô | `perform_eda(df_raw)` → `raw_mean, raw_var` |
| 3 | Chia Train/Test | `split_data(df_raw, TEST_DAYS)` |
| 4 | Fit tham số NegBin | `fit_parameters(raw_mean, raw_var)` → `n_est, p_est` |
| 5 | Huấn luyện: CV + Bellman DP | `perform_time_series_cv(train)` → `BellmanInventoryDP(...)` → `value_iteration()` → `get_optimal_policy()` |
| 6 | Mô phỏng chi phí trên Test | `simulate_policy(test, s, S)` → in chi phí |
| 7 | Đánh giá mô hình | `evaluate_model(train, ...)` + `evaluate_model(test, ...)` → Accuracy, F1, RMSSE, Confusion Matrix |

- Đầu file: `sys.stdout.reconfigure(encoding='utf-8')` chống lỗi font Windows.
- Tất cả print tiếng Việt có Unicode.

---

## Yêu cầu bổ sung

- Viết README.md đặc tả chi tiết: kiến trúc, nguồn dữ liệu, công thức toán Bellman, mô tả từng hàm, bảng kết quả CV, bảng confusion matrix, ví dụ tính toán, hướng dẫn cài đặt.
- Sử dụng Mermaid flowchart cho sơ đồ 7 bước.
- Comment và docstring bằng tiếng Việt.
