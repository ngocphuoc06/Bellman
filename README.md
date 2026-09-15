# 📦 Bellman Inventory DP — Hệ thống Quản trị Hàng tồn kho bằng Quy hoạch Động

> **Đóng vai trò chuyên gia Machine Learning & Operations Research**, dự án này áp dụng thuật toán **Value Iteration (Lặp Giá trị Bellman)** để giải quyết bài toán kiểm soát hàng tồn kho ngẫu nhiên theo chuẩn quy trình **Machine Learning 6 bước**.

---

## 📁 Cấu trúc dự án (Modular Architecture)

```
Bellman/
├── config.py          # Siêu tham số hệ thống
├── data_manager.py    # Bước 1 & 2: Thu thập & Chia tách dữ liệu
├── eda_analyzer.py    # Bước 3 & 4: EDA & Học tham số
├── dp_model.py        # Class lõi: Thuật toán Bellman DP
├── evaluator.py       # Bước 5 & 6: CV & Đánh giá cuối cùng
└── main.py            # Luồng thực thi chính (Orchestrator)
```

> ❓ **Tại sao lại chia nhỏ thành nhiều file?** Vì mỗi module chỉ làm đúng một việc (Single Responsibility Principle). Điều này giúp dễ dàng kiểm thử từng bộ phận, thay thế module mà không ảnh hưởng các phần còn lại, và nhiều người có thể cùng làm việc song song.

---

## 🔧 `config.py` — Trung tâm Cấu hình

```python
MAX_CAPACITY = 40        # M = 40
FIXED_ORDER_COST = 10    # K = 10
HOLDING_COST = 0.5       # h = 0.5
SHORTAGE_COST = 20       # p = 20
DISCOUNT_FACTOR = 0.95   # γ = 0.95
TOLERANCE = 1e-4         # θ = 0.0001
```

**Mục đích:** Tách biệt dữ liệu cấu hình ra khỏi logic code, để khi cần điều chỉnh tham số, bạn chỉ sửa đúng một nơi duy nhất.

**Tại sao chọn các giá trị này?**

| Tham số | Ý nghĩa | Lý do chọn giá trị |
|---|---|---|
| `MAX_CAPACITY = 40` | Kho chứa tối đa 40 đơn vị | Giới hạn không gian trạng thái để tránh "Lời nguyền số chiều" (Curse of Dimensionality) |
| `FIXED_ORDER_COST = 10` | Chi phí cố định mỗi lần đặt hàng (bất kể số lượng) | Tạo động lực để gộp đơn hàng thay vì đặt hàng mỗi ngày |
| `HOLDING_COST = 0.5` | Chi phí lưu kho mỗi đơn vị mỗi ngày | Thấp nhưng tích lũy theo thời gian → Tránh giữ hàng quá nhiều |
| `SHORTAGE_COST = 20` | Phạt khi hết hàng, mỗi đơn vị thiếu | Cao gấp 40 lần holding cost → Ưu tiên không để cạn kho |
| `DISCOUNT_FACTOR = 0.95` | Mức độ "quan tâm" đến chi phí tương lai | γ = 0.95 nghĩa là 1 đồng chi phí ngày mai chỉ "đau" bằng 0.95 đồng hôm nay |
| `TOLERANCE = 1e-4` | Ngưỡng hội tụ | Đủ chính xác cho bài toán vận hành |

---

## 📊 `data_manager.py` — Thu thập & Chia tách Dữ liệu

### Hàm `generate_walmart_synthetic_data(days, n_true, p_true)`

```python
from scipy.stats import nbinom
demands = nbinom.rvs(n_true, p_true, size=days)
```

**Tại sao dùng Phân phối Nhị thức Âm (Negative Binomial)?**

Dữ liệu nhu cầu hàng ngày thường có **Variance > Mean** (Siêu phân tán / Overdispersion). Phân phối Poisson (phổ biến hơn) chỉ phù hợp khi Variance ≈ Mean. Negative Binomial được thiết kế đặc biệt để xử lý trường hợp Overdispersion.

**Ví dụ trực quan:**
```
Phân phối Poisson (λ=20): Mean = 20, Variance = 20  ✗ (Không phù hợp)
Phân phối NegBinom (n=6.9, p=0.25): Mean ≈ 20, Variance ≈ 80  ✓ (Phù hợp)
```

### Hàm `split_data(df)`

```python
train_size = len(df) - TEST_DAYS  # 1913 - 100 = 1813 ngày train
train_data = df.iloc[:train_size]  # Lấy từ đầu đến mốc
test_data  = df.iloc[train_size:]  # Lấy từ mốc đến cuối
```

**Tại sao KHÔNG dùng `shuffle=True` (ngẫu nhiên)?**

Dữ liệu chuỗi thời gian có tính thứ tự. Nếu xáo trộn ngẫu nhiên, dữ liệu từ tháng 12 (tương lai) sẽ lọt vào tập Train để dự đoán cho tháng 1 (quá khứ) → **Data Leakage (Rò rỉ dữ liệu)** → Kết quả đánh giá ảo, mô hình vô dụng trong thực tế.

```
Đúng:   [====TRAIN (1813 ngày)====] [=TEST (100 ngày)=]   ← Thời gian →
Sai:    [========TRỘN LẪN NGẪU NHIÊN========]             ← Gian lận! ←
```

---

## 🔍 `eda_analyzer.py` — Khám phá Dữ liệu & Học Tham số

### Hàm `perform_eda(train_data)`

```python
train_mean = train_data['Demand'].mean()  # Trung bình nhu cầu
train_var  = train_data['Demand'].var()   # Phương sai nhu cầu
if train_var > train_mean:
    print("=> Phát hiện Siêu phân tán!")
```

**Ý nghĩa của kiểm tra Overdispersion:**
Đây là bước "thám tử" dữ liệu. Thay vì mù quáng gán phân phối, ta kiểm tra thực nghiệm xem dữ liệu thực sự tuân theo quy luật gì, rồi mới chọn công cụ phù hợp.

### Hàm `fit_parameters(train_mean, train_var)` — Moment Matching

```python
p_estimated = train_mean / train_var
n_estimated = (train_mean ** 2) / (train_var - train_mean)
```

**Tại sao dùng Moment Matching thay vì MLE (Maximum Likelihood Estimation)?**

Moment Matching tính trực tiếp từ công thức giải tích → **Nhanh hơn** và **ổn định hơn** cho dữ liệu chuỗi thời gian. Cả hai phương pháp cho kết quả gần tương đương khi tập dữ liệu đủ lớn (>1000 điểm).

**Ví dụ tính tay:**
```
train_mean = 21.20, train_var = 91.46
p = 21.20 / 91.46 = 0.2319
n = (21.20²) / (91.46 - 21.20) = 449.44 / 70.26 = 6.40
```

---

## 🤖 `dp_model.py` — Thuật toán Bellman DP (Trái tim Hệ thống)

### Class `BellmanInventoryDP`

Đây là nơi toàn bộ "trí tuệ" của hệ thống được đặt. Bài toán được mô hình hóa dưới dạng **Markov Decision Process (MDP)**:

```
Trạng thái (State s):  Mức tồn kho hiện tại (0, 1, 2, ..., 40)
Hành động (Action a):  Số đơn vị hàng cần đặt thêm
Phần thưởng (Reward):  Chi phí âm (ta muốn TỐI THIỂU hóa chi phí)
```

### Hàm `_demand_probability(d)` — Xác suất Nhu cầu

```python
return nbinom.pmf(d, self.demand_n, self.demand_p)
```

**PMF = Probability Mass Function (Hàm khối xác suất)**. Trả về xác suất nhu cầu đúng bằng `d` đơn vị trong một ngày.

**Tối ưu hiệu năng với Cache:**
```python
# Trong __init__: Tính TRƯỚC, lưu vào mảng
self.demand_probs = np.array([self._demand_probability(d) for d in range(self.max_demand)])

# Trong vòng lặp: Chỉ TRA CỨU, không tính lại
prob = self.demand_probs[demand]  # O(1) thay vì O(n)
```
Nếu không cache: Gọi `nbinom.pmf()` tổng cộng **41 states × 41 actions × 80 demands × 203 iterations ≈ 27 triệu lần**. Với cache: Chỉ tính **80 lần** một lần duy nhất.

### Hàm `expected_cost(state, action)` — Phương trình Bellman

Đây là trái tim của thuật toán. Công thức toán học:

$$Q(s, a) = C_{order}(a) + \sum_{d=0}^{\infty} P(D=d) \left[ h \cdot \max(s+a-d, 0) + p \cdot \max(d-s-a, 0) + \gamma \cdot V(s') \right]$$

**Giải thích từng thành phần bằng ví dụ:**

Giả sử: Kho hiện có `s = 5` đơn vị, quyết định đặt thêm `a = 10` → Sau khi nhập hàng: `5 + 10 = 15` đơn vị.

```python
inventory_after_order = 5 + 10 = 15

# Kịch bản 1: Nhu cầu = 12, xác suất = 0.08
ending_inventory = 15 - 12 = 3 (dương → CÒN HÀNG)
holding_cost     = 0.5 × 3 = 1.5$
next_state       = 3

# Kịch bản 2: Nhu cầu = 20, xác suất = 0.03
ending_inventory = 15 - 20 = -5 (âm → HẾT HÀNG)
shortage_cost    = 20 × 5 = 100$  # Phạt nặng!
next_state       = 0

# Tổng chi phí kỳ vọng = order_cost + Σ(prob × holding) + Σ(prob × shortage) + γ × Σ(prob × V[next])
```

### Hàm `value_iteration()` — Thuật toán Lặp Giá trị

```python
while True:
    delta = 0
    for s in self.states:          # Duyệt qua 41 trạng thái
        costs = []
        for a in range(max_action + 1):  # Thử từng hành động có thể
            cost_a = self.expected_cost(s, a)
            costs.append(cost_a)
        V_new[s] = min(costs)      # Chọn hành động tốt nhất
        policy[s] = argmin(costs)  # Ghi nhớ hành động đó
        delta = max(delta, |V[s] - V_new[s]|)  # Đo mức độ thay đổi
    if delta < tolerance:
        break  # HỘI TỤ!
```

**Tại sao thuật toán này đảm bảo hội tụ?** Đây là ứng dụng của **Định lý Điểm cố định Banach (Banach Fixed-Point Theorem)**. Toán tử Bellman là một ánh xạ co (contraction mapping) với hệ số co γ = 0.95. Mỗi vòng lặp, sai số giảm đi ít nhất 5%, đảm bảo hội tụ sau hữu hạn bước.

**Ví dụ trực quan qua 3 vòng lặp đầu:**
```
Vòng 1: V[0] = 2500, V[5] = 1800, V[40] = 100  (Ước lượng ban đầu thô)
Vòng 2: V[0] = 2380, V[5] = 1720, V[40] = 98   (Bắt đầu tinh chỉnh)
Vòng 3: V[0] = 2361, V[5] = 1705, V[40] = 97   (Tiến gần đến tối ưu)
...
Vòng 243: Hội tụ! Delta < 0.0001
```

### Hàm `get_optimal_policy()` — Trích xuất Chính sách (s, S)

```python
S_target = self.policy[0]  # Khi kho TRỐNG, đặt bao nhiêu? → Đó là S
s_reorder = -1
for state, action in enumerate(self.policy):
    if action > 0:
        s_reorder = state  # Mức kho cao nhất mà vẫn còn đặt hàng → Đó là s
```

**Chính sách (s, S) — Ví dụ với kết quả s=31, S=40:**
```
Kho = 29 → Đặt 11 đơn vị (29 + 11 = 40)  ✓ Đặt hàng
Kho = 30 → Đặt 10 đơn vị (30 + 10 = 40)  ✓ Đặt hàng
Kho = 31 → Đặt  9 đơn vị (31 +  9 = 40)  ✓ Đặt hàng (điểm s)
Kho = 32 → Đặt  0 đơn vị                  ✗ Không đặt
Kho = 40 → Đặt  0 đơn vị                  ✗ Không đặt
```

---

## 📐 `evaluator.py` — Kiểm định & Đánh giá

### Hàm `perform_time_series_cv(train_data)` — Kiểm định chéo Chuỗi thời gian

**Tại sao cần Cross-Validation?** Để chống overfitting: Đảm bảo chính sách (s, S) hoạt động tốt trên nhiều "cửa sổ" thời gian khác nhau, không chỉ trên một đoạn dữ liệu cụ thể.

```
Rolling-window 5-fold CV (Expanding Window Strategy):
                                        
Fold 1: [====Train (1300 ngày)====][Val (100)]
Fold 2: [======Train (1400 ngày)======][Val (100)]
Fold 3: [========Train (1500 ngày)========][Val (100)]
Fold 4: [==========Train (1600 ngày)==========][Val (100)]
Fold 5: [============Train (1700 ngày)============][Val (100)]
```

**Tại sao dùng Expanding Window (mở rộng) thay vì Sliding Window (trượt)?** Expanding window tận dụng toàn bộ lịch sử dữ liệu, phù hợp hơn khi mô hình không bị overfit theo thời gian.

### Hàm `simulate_policy(test_data, s_reorder, S_target)` — Mô phỏng Thực tế

```python
for actual_demand in test_data['Demand']:
    # 1. Đầu ngày: Kiểm tra có cần đặt hàng không?
    if current_inventory <= s_reorder:
        order_qty = S_target - current_inventory
        total_cost += FIXED_ORDER_COST + UNIT_ORDER_COST * order_qty
        current_inventory += order_qty
    
    # 2. Trong ngày: Bán hàng
    current_inventory -= actual_demand
    
    # 3. Cuối ngày: Tính chi phí tồn đọng hoặc phạt thiếu hàng
    if current_inventory > 0:
        total_cost += HOLDING_COST * current_inventory
    else:
        total_cost += SHORTAGE_COST * abs(current_inventory)
        current_inventory = 0
```

**Ví dụ mô phỏng 3 ngày:**
```
Ngày 1: Kho = 35, Nhu cầu = 28 → Kho còn 7. Chi phí lưu kho = 0.5 × 7 = 3.5$
Ngày 2: Kho = 7 <= s(31) → Đặt 40-7=33 đơn vị. CF đặt hàng = 10$. Bán 29. Kho còn 11.
Ngày 3: Kho = 11 <= s(31) → Đặt 40-11=29 đơn vị. CF đặt hàng = 10$. Bán 35. Kho còn 5.
```

---

## 🎬 `main.py` — Luồng thực thi chính

File này là "Đạo diễn" — không tự làm gì cả, chỉ gọi các module theo đúng thứ tự:

```python
# Bước 1: Thu thập
df_raw = generate_walmart_synthetic_data(days=1913)

# Bước 2: Chia tách (Ngay từ đầu, không đụng vào test_data cho đến Bước 6)
train_data, test_data = split_data(df_raw)

# Bước 3: EDA (Chỉ trên train_data)
train_mean, train_var = perform_eda(train_data)

# Bước 4: Học tham số (Chỉ từ train_data)
n_estimated, p_estimated = fit_parameters(train_mean, train_var)

# Bước 5: Huấn luyện = CV + Fit cuối cùng trên toàn bộ train
perform_time_series_cv(train_data)
dp_model = BellmanInventoryDP(..., demand_n=n_estimated, demand_p=p_estimated)
dp_model.value_iteration()
s_reorder, S_target = dp_model.get_optimal_policy()

# Bước 6: Đánh giá FINAL trên test_data (Chỉ chạm vào test_data đúng một lần này)
total_cost = simulate_policy(test_data, s_reorder, S_target)
```

---

## 🚀 Cách chạy dự án

```bash
# Cài đặt thư viện
pip install numpy pandas scipy scikit-learn matplotlib

# Chạy chương trình
python main.py
```

**Kết quả mẫu:**
```
=== BƯỚC 1: THU THẬP DỮ LIỆU ===
Đã sinh tập dữ liệu với 1913 dòng.

=== BƯỚC 2: DATA SPLITTING ===
Tập huấn luyện (Train): 1813 ngày.
Tập kiểm thử (Test): 100 ngày.

=== BƯỚC 3: EDA ===
Mean: 21.20, Variance: 91.46
=> Phát hiện Siêu phân tán (Overdispersion)!

=== BƯỚC 4: TIỀN XỬ LÝ ===
p_estimated = 0.2319, n_estimated = 6.40

=== BƯỚC 5: HUẤN LUYỆN ===
Fold 1  s=31, S=40  Val Cost: $2,314.00
Fold 2  s=31, S=40  Val Cost: $2,942.50
...
Mô hình hội tụ sau 243 vòng lặp.
Chính sách (s, S): s=31, S=40

=== BƯỚC 6: FINAL EVALUATION ===
TỔNG CHI PHÍ VẬN HÀNH: $2,251.50
```

---

## 📚 Công thức Toán học Tổng kết

| Ký hiệu | Tên | Công thức |
|---|---|---|
| $V^*(s)$ | Hàm giá trị tối ưu | $V^*(s) = \min_a Q(s,a)$ |
| $Q(s,a)$ | Hàm hành động-giá trị | $C_{order} + \mathbb{E}_D[h \cdot (s+a-D)^+ + p \cdot (D-s-a)^+ + \gamma V^*(s')]$ |
| $\delta$ | Sai số Bellman | $\delta = \max_s \|V_{new}(s) - V_{old}(s)\|$ |
| $(s, S)$ | Chính sách tối ưu | Đặt hàng khi kho $\leq s$, nạp đầy lên $S$ |

---

*Dự án được xây dựng theo chuẩn **Production-Ready ML Pipeline** — Phù hợp tham khảo cho các bài toán Vận trù học & Chuỗi cung ứng thực tế.*
