import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.stats import nbinom

class BellmanInventoryDP:
    def __init__(self, max_capacity, fixed_order_cost, unit_order_cost, 
                 holding_cost, shortage_cost, discount_factor, tolerance,
                 demand_n=6.9, demand_p=0.25):
        # 1. Siêu tham số & Chi phí
        self.M = max_capacity                # Ngưỡng vật lý của không gian trạng thái (tránh Curse of Dimensionality)
        self.K = fixed_order_cost            # Chi phí đặt hàng cố định
        self.c = unit_order_cost             # Chi phí trên mỗi đơn vị hàng
        self.h = holding_cost                # Chi phí lưu kho
        self.p = shortage_cost               # Chi phí phạt cạn kho
        self.gamma = discount_factor         # Hệ số chiết khấu (Tầm nhìn dài hạn)
        self.theta = tolerance               # Ngưỡng hội tụ
        self.demand_n = demand_n
        self.demand_p = demand_p
        
        # 2. Không gian Trạng thái (S) và Hàm Giá trị (V)
        self.states = np.arange(self.M + 1)
        self.V = np.zeros(self.M + 1)        # Khởi tạo V(s) = 0
        self.policy = np.zeros(self.M + 1, dtype=int)
        
        self.max_demand = self.M * 2
        self.demand_probs = np.array([self._demand_probability(d) for d in range(self.max_demand)])
        
    def _demand_probability(self, d):
        """Mô phỏng phân phối nhu cầu bằng Nhị thức Âm (Negative Binomial) do dữ liệu bị Siêu phân tán"""
        return nbinom.pmf(d, self.demand_n, self.demand_p)

    def expected_cost(self, state, action):
        """Tính hàm mục tiêu: Tổng chi phí kỳ vọng khi thực thi hành động a tại trạng thái s"""
        inventory_after_order = state + action
        expected_holding = 0
        expected_shortage = 0
        expected_future_value = 0
        
        # Chi phí đặt hàng
        order_cost = self.K + self.c * action if action > 0 else 0
        
        # Quét qua các kịch bản nhu cầu có thể xảy ra (giới hạn đuôi phân phối ở mức an toàn)
        for demand in range(self.max_demand):
            prob = self.demand_probs[demand]
            if prob < 1e-6: continue
            
            # Trạng thái kho cuối ngày
            ending_inventory = inventory_after_order - demand
            
            if ending_inventory > 0:
                expected_holding += prob * (self.h * ending_inventory)
                next_state = min(ending_inventory, self.M)
            else:
                expected_shortage += prob * (self.p * abs(ending_inventory))
                next_state = 0
                
            # Giá trị chiết khấu tương lai (Ánh xạ co Bellman)
            expected_future_value += prob * self.V[next_state]
            
        return order_cost + expected_holding + expected_shortage + self.gamma * expected_future_value

    def value_iteration(self):
        """Quá trình cập nhật đệ quy để tìm hàm giá trị tối ưu"""
        iteration = 0
        while True:
            delta = 0
            V_new = np.zeros_like(self.V)
            
            for s in self.states:
                # Không gian hành động A: Lượng đặt hàng tối đa không vượt quá sức chứa M
                max_action = self.M - s
                costs = []
                
                for a in range(max_action + 1):
                    cost_a = self.expected_cost(s, a)
                    costs.append(cost_a)
                
                # Phương trình Tối ưu Bellman (Tối thiểu hóa chi phí)
                best_action_cost = min(costs)
                V_new[s] = best_action_cost
                self.policy[s] = np.argmin(costs)
                
                # Đo lường khoảng cách theo chuẩn cực đại (Supremum norm)
                delta = max(delta, abs(self.V[s] - V_new[s]))
                
            self.V = np.copy(V_new)
            iteration += 1
            
            # Định lý Banach: Dừng khi sai số Bellman nhỏ hơn ngưỡng hội tụ theta
            if delta < self.theta:
                print(f"Thuật toán hội tụ tại vòng lặp thứ {iteration}.")
                break
                
    def get_optimal_policy(self):
        return self.policy

if __name__ == "__main__":
    # Khởi tạo tham số
    max_capacity = 50
    fixed_order_cost = 100
    unit_order_cost = 10
    holding_cost = 2
    shortage_cost = 20
    discount_factor = 0.95
    tolerance = 0.01

    print("Khởi tạo mô hình DP...")
    dp = BellmanInventoryDP(max_capacity, fixed_order_cost, unit_order_cost, 
                            holding_cost, shortage_cost, discount_factor, tolerance)
    
    print("Bắt đầu lặp giá trị (Value Iteration)...")
    dp.value_iteration()
    
    policy = dp.get_optimal_policy()
    
    # Tự động quét mảng policy để tìm s và S
    S_target = policy[0]
    s_reorder = -1
    for state, action in enumerate(policy):
        if action > 0:
            s_reorder = state
            
    print(f"Chính sách Tồn kho (s, S):")
    print(f" - Điểm đặt hàng (s) = {s_reorder}")
    print(f" - Mức tồn kho mục tiêu (S) = {S_target}")