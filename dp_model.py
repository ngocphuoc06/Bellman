import numpy as np
from scipy.stats import nbinom

class BellmanInventoryDP:
    """
    Class lõi giải quyết bài toán kiểm soát hàng tồn kho bằng Quy hoạch động Bellman.
    Hỗ trợ mô hình hóa dưới dạng Markov Decision Process (MDP).
    """
    def __init__(self, max_capacity, fixed_order_cost, unit_order_cost, 
                 holding_cost, shortage_cost, discount_factor, tolerance,
                 demand_n, demand_p):
        self.M = max_capacity
        self.K = fixed_order_cost
        self.c = unit_order_cost
        self.h = holding_cost
        self.p = shortage_cost
        self.gamma = discount_factor
        self.theta = tolerance
        self.demand_n = demand_n
        self.demand_p = demand_p
        
        self.states = np.arange(self.M + 1)
        self.V = np.zeros(self.M + 1)
        self.policy = np.zeros(self.M + 1, dtype=int)
        
        # Cache xác suất nhu cầu để tối ưu hiệu năng
        self.max_demand = self.M * 2
        self.demand_probs = np.array([self._demand_probability(d) for d in range(self.max_demand)])
        
    def _demand_probability(self, d):
        return nbinom.pmf(d, self.demand_n, self.demand_p)

    def expected_cost(self, state, action):
        inventory_after_order = state + action
        expected_holding = 0
        expected_shortage = 0
        expected_future_value = 0
        
        order_cost = self.K + self.c * action if action > 0 else 0
        
        for demand in range(self.max_demand):
            prob = self.demand_probs[demand]
            if prob < 1e-6: continue
            
            ending_inventory = inventory_after_order - demand
            
            if ending_inventory > 0:
                expected_holding += prob * (self.h * ending_inventory)
                next_state = min(ending_inventory, self.M)
            else:
                expected_shortage += prob * (self.p * abs(ending_inventory))
                next_state = 0
                
            expected_future_value += prob * self.V[next_state]
            
        return order_cost + expected_holding + expected_shortage + self.gamma * expected_future_value

    def value_iteration(self, verbose=False):
        """
        Quá trình Value Iteration (Lặp giá trị) để tìm phương trình tối ưu Bellman.
        """
        iteration = 0
        while True:
            delta = 0
            V_new = np.zeros_like(self.V)
            
            for s in self.states:
                max_action = self.M - s
                costs = []
                
                for a in range(max_action + 1):
                    cost_a = self.expected_cost(s, a)
                    costs.append(cost_a)
                
                best_action_cost = min(costs)
                V_new[s] = best_action_cost
                self.policy[s] = np.argmin(costs)
                
                delta = max(delta, abs(self.V[s] - V_new[s]))
                
            self.V = np.copy(V_new)
            iteration += 1
            
            if delta < self.theta:
                if verbose:
                    print(f"Thuật toán hội tụ tại vòng lặp thứ {iteration}.")
                break
                
    def get_optimal_policy(self):
        """
        Trích xuất (s, S) policy (Base-stock policy).
        """
        S_target = self.policy[0]
        s_reorder = -1
        for state, action in enumerate(self.policy):
            if action > 0:
                s_reorder = state
        return s_reorder, S_target
