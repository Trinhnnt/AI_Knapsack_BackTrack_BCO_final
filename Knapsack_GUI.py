import gradio as gr
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import heapq

# =============================================================================
# PHẦN 1: HÀM SINH DỮ LIỆU
# =============================================================================

def generate_knapsack_problem(num_items, max_weight=20, max_value=100, capacity_ratio=0.5, seed=None):
    """Tạo bài toán knapsack ngẫu nhiên"""
    if seed is not None:
        random.seed(seed)
    values = [random.randint(1, max_value) for _ in range(num_items)]
    weights = [random.randint(1, max_weight) for _ in range(num_items)]
    capacity = int(sum(weights) * capacity_ratio)
    return values, weights, capacity

# =============================================================================
# PHẦN 2: CÁC THUẬT TOÁN (GIỮ NGUYÊN CODE CŨ)
# =============================================================================
def run_brute_force(num_items, max_weight, max_value, capacity_ratio, seed):
    """1. Brute Force - TOÀN BỘ QUY TRÌNH"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    max_value_result = 0
    best_combination = []
    convergence_steps = []
    nodes_visited = 0
    
    start_time = time.time()
    
    for r in range(n + 1):
        for combination in combinations(range(n), r):
            nodes_visited += 1
            total_weight = sum(weights[i] for i in combination)
            total_value = sum(values[i] for i in combination)
            
            if total_weight <= capacity and total_value > max_value_result:
                max_value_result = total_value
                best_combination = combination
            
            convergence_steps.append(max_value_result)
    
    execution_time = time.time() - start_time
    
    solution = [1 if i in best_combination else 0 for i in range(n)]
    total_weight = sum(weights[i] for i in best_combination)
    
    threshold = 0.9 * max_value_result
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': solution,
        'max_value': max_value_result,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': nodes_visited,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_simple_backtracking(num_items, max_weight, max_value, capacity_ratio, seed):
    """2. Simple Backtracking - TOÀN BỘ QUY TRÌNH"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    best = {"value": 0, "weight": 0, "solution": [0] * n}
    convergence_steps = []
    nodes_visited = 0
    
    start_time = time.time()
    
    def backtrack(i, current_weight, current_value, selected):
        nonlocal nodes_visited
        nodes_visited += 1
        convergence_steps.append(best["value"])
        
        if i == n:
            if current_weight <= capacity and current_value > best["value"]:
                best["value"] = current_value
                best["weight"] = current_weight
                best["solution"] = selected[:]
            return
        
        selected[i] = 0
        backtrack(i + 1, current_weight, current_value, selected)
        
        if current_weight + weights[i] <= capacity:
            selected[i] = 1
            backtrack(i + 1, current_weight + weights[i], current_value + values[i], selected)
    
    selected = [0] * n
    backtrack(0, 0, 0, selected)
    execution_time = time.time() - start_time
    
    threshold = 0.9 * best["value"]
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best["solution"],
        'max_value': best["value"],
        'total_weight': best["weight"],
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': nodes_visited,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_non_recursive_backtracking(num_items, max_weight, max_value, capacity_ratio, seed):
    """3. Non-Recursive Backtracking - TOÀN BỘ QUY TRÌNH"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    stack = [(0, 0, 0, [0] * n, 0)]  # (index, weight, value, solution, stage)
    
    best_value = 0
    best_solution = [0] * n
    nodes_visited = 0
    convergence_steps = []
    
    start_time = time.time()
    
    while stack:
        i, cur_w, cur_v, sol, stage = stack.pop()
        nodes_visited += 1
        convergence_steps.append(best_value)
        
        if i == n:
            if cur_v > best_value:
                best_value = cur_v
                best_solution = sol[:]
            continue
        
        if stage == 0:
            if cur_w + weights[i] <= capacity:
                sol_inc = sol[:]
                sol_inc[i] = 1
                stack.append((i, cur_w, cur_v, sol, 1))
                stack.append((i + 1, cur_w + weights[i], cur_v + values[i], sol_inc, 0))
            else:
                stack.append((i, cur_w, cur_v, sol, 1))
        else:
            sol_exc = sol[:]
            sol_exc[i] = 0
            stack.append((i + 1, cur_w, cur_v, sol_exc, 0))
    
    execution_time = time.time() - start_time
    
    total_weight = sum(weights[i] for i in range(n) if best_solution[i] == 1)
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': nodes_visited,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_weight_pruning(num_items, max_weight, max_value, capacity_ratio, seed):
    """4. Weight Pruning - TOÀN BỘ QUY TRÌNH"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(values)
    best_value = 0
    best_solution = [0] * n
    convergence_steps = []
    nodes_visited = 0
    
    start_time = time.time()
    
    def backtrack(i, cur_weight, cur_value, sol):
        nonlocal best_value, best_solution, nodes_visited
        nodes_visited += 1
        convergence_steps.append(best_value)
        
        if cur_weight > capacity:
            return
        
        if i == n:
            if cur_value > best_value:
                best_value = cur_value
                best_solution = sol[:]
            return
        
        sol[i] = 1
        backtrack(i + 1, cur_weight + weights[i], cur_value + values[i], sol)
        sol[i] = 0
        backtrack(i + 1, cur_weight, cur_value, sol)
    
    sol = [0] * n
    backtrack(0, 0, 0, sol)
    execution_time = time.time() - start_time
    
    total_weight = sum(weights[i] for i in range(n) if best_solution[i] == 1)
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': nodes_visited,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_upper_bound(num_items, max_weight, max_value, capacity_ratio, seed):
    """5. Upper Bound Pruning - TOÀN BỘ QUY TRÌNH"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(values)
    best_value = 0
    best_solution = [0] * n
    convergence_steps = []
    nodes_visited = 0
    
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    sorted_values, sorted_weights = zip(*items)
    
    start_time = time.time()
    
    def bound(i, cur_weight, cur_value):
        if cur_weight >= capacity:
            return 0
        rem_capacity = capacity - cur_weight
        upper = cur_value
        for j in range(i, n):
            if sorted_weights[j] <= rem_capacity:
                rem_capacity -= sorted_weights[j]
                upper += sorted_values[j]
            else:
                upper += sorted_values[j] * (rem_capacity / sorted_weights[j])
                break
        return upper
    
    def backtrack(i, cur_weight, cur_value, sol):
        nonlocal best_value, best_solution, nodes_visited
        nodes_visited += 1
        convergence_steps.append(best_value)
        
        if cur_weight > capacity:
            return
        
        if i == n:
            if cur_value > best_value:
                best_value = cur_value
                best_solution = sol[:]
            return
        
        upper = bound(i, cur_weight, cur_value)
        if upper < best_value:
            return
        
        sol[i] = 1
        backtrack(i + 1, cur_weight + sorted_weights[i], cur_value + sorted_values[i], sol)
        sol[i] = 0
        backtrack(i + 1, cur_weight, cur_value, sol)
    
    sol = [0] * n
    backtrack(0, 0, 0, sol)
    execution_time = time.time() - start_time
    
    total_weight = sum(sorted_weights[i] for i in range(n) if best_solution[i] == 1)
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': nodes_visited,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': list(sorted_values),
        'weights': list(sorted_weights)
    }

def run_ratio_pruning(num_items, max_weight, max_value, capacity_ratio, seed):
    """6. Ratio-Pruning - TOÀN BỘ QUY TRÌNH"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(values)
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    sorted_values, sorted_weights = zip(*items)
    
    best_value = 0
    best_solution = [0]*n
    convergence_steps = []
    nodes_visited = 0
    
    start_time = time.time()
    
    def backtrack(i, cur_w, cur_v, sol):
        nonlocal best_value, best_solution, nodes_visited
        nodes_visited += 1
        convergence_steps.append(best_value)
        
        if cur_w > capacity:
            return
        
        if cur_v > best_value:
            best_value = cur_v
            best_solution = sol[:]
        
        if i == n:
            return
        
        bound = cur_v
        rem_capacity = capacity - cur_w
        for j in range(i, n):
            if sorted_weights[j] <= rem_capacity:
                rem_capacity -= sorted_weights[j]
                bound += sorted_values[j]
            else:
                bound += sorted_values[j] * rem_capacity / sorted_weights[j]
                break
        
        if bound <= best_value:
            return
        
        sol[i] = 1
        backtrack(i+1, cur_w + sorted_weights[i], cur_v + sorted_values[i], sol)
        sol[i] = 0
        backtrack(i+1, cur_w, cur_v, sol)
    
    sol_init = [0]*n
    backtrack(0, 0, 0, sol_init)
    execution_time = time.time() - start_time
    
    total_weight = sum(sorted_weights[i] for i in range(n) if best_solution[i] == 1)
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': nodes_visited,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': list(sorted_values),
        'weights': list(sorted_weights)
    }

def run_branch_and_bound(num_items, max_weight, max_value, capacity_ratio, seed):
    """7. Branch & Bound - TOÀN BỘ QUY TRÌNH"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    values_sorted, weights_sorted = zip(*items)
    
    convergence_steps = []
    nodes_visited = 0
    
    start_time = time.time()
    
    class Node:
        def __init__(self, level, value, weight, bound, solution):
            self.level = level
            self.value = value
            self.weight = weight
            self.bound = bound
            self.solution = solution
        
        def __lt__(self, other):
            return self.bound > other.bound
    
    def bound(node, n, weights, values, capacity):
        if node.weight >= capacity:
            return 0
        result = node.value
        total_weight = node.weight
        j = node.level + 1
        
        while j < n and total_weight + weights[j] <= capacity:
            total_weight += weights[j]
            result += values[j]
            j += 1
        
        if j < n:
            result += (capacity - total_weight) * values[j] / weights[j]
        return result
    
    queue = []
    root = Node(level=-1, value=0, weight=0, bound=0, solution=[0]*n)
    root.bound = bound(root, n, weights_sorted, values_sorted, capacity)
    heapq.heappush(queue, root)
    
    best_value = 0
    best_solution = [0]*n
    
    while queue:
        node = heapq.heappop(queue)
        nodes_visited += 1
        convergence_steps.append(best_value)
        
        if node.bound <= best_value or node.level == n-1:
            continue
        
        next_level = node.level + 1
        
        if node.weight + weights_sorted[next_level] <= capacity:
            sol_inc = node.solution[:]
            sol_inc[next_level] = 1
            value_inc = node.value + values_sorted[next_level]
            weight_inc = node.weight + weights_sorted[next_level]
            child_inc = Node(next_level, value_inc, weight_inc, 0, sol_inc)
            child_inc.bound = bound(child_inc, n, weights_sorted, values_sorted, capacity)
            
            if value_inc > best_value:
                best_value = value_inc
                best_solution = sol_inc[:]
            
            if child_inc.bound > best_value:
                heapq.heappush(queue, child_inc)
        
        sol_exc = node.solution[:]
        sol_exc[next_level] = 0
        child_exc = Node(next_level, node.value, node.weight, 0, sol_exc)
        child_exc.bound = bound(child_exc, n, weights_sorted, values_sorted, capacity)
        if child_exc.bound > best_value:
            heapq.heappush(queue, child_exc)
    
    execution_time = time.time() - start_time
    
    total_weight = sum(weights_sorted[i] for i in range(n) if best_solution[i] == 1)
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': nodes_visited,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': list(values_sorted),
        'weights': list(weights_sorted)
    }

# =============================================================================
# PHẦN 3: 6 THUẬT TOÁN BCO
# =============================================================================

def run_bco_basic(num_items, max_weight, max_value, capacity_ratio, seed):
    random.seed(seed)
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    random.seed(seed)
    """8. BCO Basic - TOÀN BỘ QUY TRÌNH (ĐÃ SỬA - GIỐNG CODE 2)"""
    NUM_ITEMS = len(weights)
    ITEMS = [[values[i], weights[i]] for i in range(NUM_ITEMS)]
    convergence_steps = []
    evaluations = 0
    
    start_time = time.time()
    
    def knapsack_fitness(solution):
        nonlocal evaluations
        evaluations += 1
        total_value = sum(ITEMS[i][0] for i in range(NUM_ITEMS) if solution[i] == 1)
        total_weight = sum(ITEMS[i][1] for i in range(NUM_ITEMS) if solution[i] == 1)
        return total_value if total_weight <= capacity else 0
    
    def create_bee():
        return {'solution': [0] * NUM_ITEMS, 'fitness': 0, 'is_recruiter': False, 'recruiter_index': -1}
    
    def construct_solution(bee, nc, is_follower=False, recruiter_solution=None):
        if is_follower and recruiter_solution is not None:
            bee['solution'] = list(recruiter_solution)
        else:
            bee['solution'] = [0] * NUM_ITEMS
        
        available_indices = [i for i, val in enumerate(bee['solution']) if val == 0]
        random.shuffle(available_indices)
        
        for _ in range(min(nc, len(available_indices))):
            if not available_indices:
                break
            item_idx = available_indices.pop(0)
            bee['solution'][item_idx] = 1
            if knapsack_fitness(bee['solution']) == 0:
                bee['solution'][item_idx] = 0
        
        bee['fitness'] = knapsack_fitness(bee['solution'])
    
    B = 30
    NC = 10
    MaxIter = 50
    
    population = [create_bee() for _ in range(B)]
    best_solution_global = None
    best_fitness_global = 0
    
    # ===== KHỞI TẠO BAN ĐẦU =====
    for bee in population:
        construct_solution(bee, NC)
        if bee['fitness'] > best_fitness_global:
            best_fitness_global = bee['fitness']
            best_solution_global = list(bee['solution'])
    
    convergence_steps.append(best_fitness_global)
    
    # ===== VÒNG LẶP CHÍNH =====
    for iter_count in range(MaxIter):
        # BƯỚC 1: PHÂN VAI TRÒ (Role Assignment)
        population.sort(key=lambda bee: bee['fitness'], reverse=True)
        Recruiter_Count = B // 2
        recruiters = population[:Recruiter_Count]
        followers = population[Recruiter_Count:]
        
        for i in range(B):
            population[i]['is_recruiter'] = (i < Recruiter_Count)
            population[i]['recruiter_index'] = -1
        
        # BƯỚC 2: TUYỂN DỤNG (Recruitment)
        total_recruiter_fitness = sum([r['fitness'] for r in recruiters])
        
        if total_recruiter_fitness > 0:
            probabilities = [r['fitness'] / total_recruiter_fitness for r in recruiters]
            for follower in followers:
                chosen_recruiter = random.choices(recruiters, weights=probabilities, k=1)[0]
                recruiter_index = next(idx for idx, bee in enumerate(population) if bee is chosen_recruiter)
                follower['recruiter_index'] = recruiter_index
        
        # ⭐ BƯỚC 3: FORWARD PASS (Lượt đi) ⭐
        for bee in population:
            if bee['is_recruiter']:
                construct_solution(bee, NC, is_follower=False)
            else:
                if bee['recruiter_index'] != -1:
                    recruiter_sol = population[bee['recruiter_index']]['solution']
                    construct_solution(bee, NC, is_follower=True, recruiter_solution=recruiter_sol)
                else:
                    construct_solution(bee, NC, is_follower=False)
            
            if bee['fitness'] > best_fitness_global:
                best_fitness_global = bee['fitness']
                best_solution_global = list(bee['solution'])
        
        # ⭐ BƯỚC 4: BACKWARD PASS (Lượt về) - THÊM MỚI ⭐
        for bee in population:
            if bee['is_recruiter']:
                construct_solution(bee, NC, is_follower=False)
            else:
                if bee['recruiter_index'] != -1:
                    recruiter_sol = population[bee['recruiter_index']]['solution']
                    construct_solution(bee, NC, is_follower=True, recruiter_solution=recruiter_sol)
                else:
                    construct_solution(bee, NC, is_follower=False)
            
            if bee['fitness'] > best_fitness_global:
                best_fitness_global = bee['fitness']
                best_solution_global = list(bee['solution'])
        
        convergence_steps.append(best_fitness_global)
    
    execution_time = time.time() - start_time
    
    total_weight = sum(weights[i] for i in range(NUM_ITEMS) if best_solution_global[i] == 1)
    threshold = 0.9 * best_fitness_global
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution_global,
        'max_value': best_fitness_global,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': evaluations,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_bco_local_search(num_items, max_weight, max_value, capacity_ratio, seed):
    """9. BCO + Local Search - ĐÃ SỬA (CÓ BACKWARD PASS)"""
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    random.seed(seed)  # Set seed sau khi generate data

    NUM_ITEMS = len(weights)
    ITEMS = [[values[i], weights[i]] for i in range(NUM_ITEMS)]
    convergence_steps = []
    evaluations = 0
    
    start_time = time.time()
    
    def knapsack_fitness(solution):
        nonlocal evaluations
        evaluations += 1
        total_value = sum(ITEMS[i][0] for i in range(NUM_ITEMS) if solution[i] == 1)
        total_weight = sum(ITEMS[i][1] for i in range(NUM_ITEMS) if solution[i] == 1)
        return total_value if total_weight <= capacity else 0
    
    def local_search_refinement(solution, max_ls_steps=5):
        current_solution = list(solution)
        current_fitness = knapsack_fitness(current_solution)
        
        for _ in range(max_ls_steps):
            neighbor_type = random.choice([0, 1])
            new_solution = list(current_solution)
            
            if neighbor_type == 0:
                idx = random.randint(0, NUM_ITEMS - 1)
                new_solution[idx] = 1 - new_solution[idx]
            else:
                idx1, idx2 = random.sample(range(NUM_ITEMS), 2)
                new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
            
            new_fitness = knapsack_fitness(new_solution)
            
            if new_fitness > current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness
        
        return current_solution, current_fitness
    
    def create_bee():
        return {'solution': [0] * NUM_ITEMS, 'fitness': 0, 'is_recruiter': False, 'recruiter_index': -1}
    
    def construct_solution(bee, nc, is_follower=False, recruiter_solution=None):
        if is_follower and recruiter_solution is not None:
            bee['solution'] = list(recruiter_solution)
        else:
            bee['solution'] = [0] * NUM_ITEMS
        
        available_indices = [i for i, val in enumerate(bee['solution']) if val == 0]
        random.shuffle(available_indices)
        
        for _ in range(min(nc, len(available_indices))):
            if not available_indices:
                break
            item_idx = available_indices.pop(0)
            bee['solution'][item_idx] = 1
            if knapsack_fitness(bee['solution']) == 0:
                bee['solution'][item_idx] = 0
        
        bee['fitness'] = knapsack_fitness(bee['solution'])
    
    B = 30
    NC = 10
    MaxIter = 50
    ls_steps = 5
    
    population = [create_bee() for _ in range(B)]
    best_solution_global = None
    best_fitness_global = 0
    
    # Khởi tạo ban đầu
    for bee in population:
        construct_solution(bee, NC)
        if bee['fitness'] > best_fitness_global:
            best_fitness_global = bee['fitness']
            best_solution_global = list(bee['solution'])
    
    convergence_steps.append(best_fitness_global)
    
    # Vòng lặp chính
    for iter_count in range(MaxIter):
        # BƯỚC 1: PHÂN VAI TRÒ
        population.sort(key=lambda bee: bee['fitness'], reverse=True)
        Recruiter_Count = B // 2
        recruiters = population[:Recruiter_Count]
        followers = population[Recruiter_Count:]
        
        for i in range(B):
            population[i]['is_recruiter'] = (i < Recruiter_Count)
            population[i]['recruiter_index'] = -1
        
        # BƯỚC 2: TUYỂN DỤNG
        total_recruiter_fitness = sum([r['fitness'] for r in recruiters])
        
        if total_recruiter_fitness > 0:
            probabilities = [r['fitness'] / total_recruiter_fitness for r in recruiters]
            for follower in followers:
                chosen_recruiter = random.choices(recruiters, weights=probabilities, k=1)[0]
                recruiter_index = next(idx for idx, bee in enumerate(population) if bee is chosen_recruiter)
                follower['recruiter_index'] = recruiter_index
        
        # ⭐ BƯỚC 3: FORWARD PASS ⭐
        for bee in population:
            if bee['is_recruiter']:
                construct_solution(bee, NC, is_follower=False)
            else:
                if bee['recruiter_index'] != -1:
                    recruiter_sol = population[bee['recruiter_index']]['solution']
                    construct_solution(bee, NC, is_follower=True, recruiter_solution=recruiter_sol)
                else:
                    construct_solution(bee, NC, is_follower=False)
            
            if bee['fitness'] > best_fitness_global:
                best_fitness_global = bee['fitness']
                best_solution_global = list(bee['solution'])
        
        # ⭐ BƯỚC 4: BACKWARD PASS (THÊM MỚI) ⭐
        for bee in population:
            if bee['is_recruiter']:
                construct_solution(bee, NC, is_follower=False)
            else:
                if bee['recruiter_index'] != -1:
                    recruiter_sol = population[bee['recruiter_index']]['solution']
                    construct_solution(bee, NC, is_follower=True, recruiter_solution=recruiter_sol)
                else:
                    construct_solution(bee, NC, is_follower=False)
            
            if bee['fitness'] > best_fitness_global:
                best_fitness_global = bee['fitness']
                best_solution_global = list(bee['solution'])
        
        # ⭐ BƯỚC 5: LOCAL SEARCH ⭐
        for bee in population:
            refined_sol, refined_fit = local_search_refinement(bee['solution'], ls_steps)
            if refined_fit > bee['fitness']:
                bee['solution'] = refined_sol
                bee['fitness'] = refined_fit
            
            if bee['fitness'] > best_fitness_global:
                best_fitness_global = bee['fitness']
                best_solution_global = list(bee['solution'])
        
        convergence_steps.append(best_fitness_global)
    
    execution_time = time.time() - start_time
    
    total_weight = sum(weights[i] for i in range(NUM_ITEMS) if best_solution_global[i] == 1)
    threshold = 0.9 * best_fitness_global
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution_global,
        'max_value': best_fitness_global,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': evaluations,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_adaptive_bco(num_items, max_weight, max_value, capacity_ratio, seed):
    """10. Adaptive BCO - TOÀN BỘ QUY TRÌNH"""
    random.seed(seed)
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    num_bees = 20
    max_iterations = 100
    convergence_steps = []
    evaluations = 0
    
    start_time = time.time()
    
    def evaluate_fitness(solution):
        total_weight = sum(w * s for w, s in zip(weights, solution))
        total_value = sum(v * s for v, s in zip(values, solution))
        return total_value if total_weight <= capacity else 0
    
    def generate_neighbor(solution, alpha):
        new_solution = solution[:]
        flip_prob = 0.5 * (1 - alpha)
        for i in range(len(solution)):
            if random.random() < flip_prob:
                new_solution[i] = 1 - new_solution[i]
        return new_solution
    
    bees = [[random.choice([0, 1]) for _ in range(n)] for _ in range(num_bees)]
    best_solution = max(bees, key=lambda s: evaluate_fitness(s))
    best_value = evaluate_fitness(best_solution)
    
    
    for iteration in range(max_iterations):
        alpha = iteration / max_iterations
        new_bees = []
        
        for bee in bees:
            neighbor = generate_neighbor(bee, alpha)
            eval_old = evaluate_fitness(bee)
            eval_new = evaluate_fitness(neighbor)
            evaluations += 2
            new_bees.append(neighbor if eval_new > eval_old else bee)
        
        bees = new_bees
        current_best = max(bees, key=lambda s: evaluate_fitness(s))
        current_fit = evaluate_fitness(current_best)
        if current_fit > best_value:
            best_solution, best_value = current_best, current_fit
        
        convergence_steps.append(best_value)
    
    execution_time = time.time() - start_time
    
    total_weight = sum(w * s for w, s in zip(weights, best_solution))
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': evaluations,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_elite_bco(num_items, max_weight, max_value, capacity_ratio, seed):
    """11. Elite BCO - TOÀN BỘ QUY TRÌNH"""
    random.seed(seed)
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    num_bees = 20
    max_iterations = 100
    elite_ratio = 0.1
    elite_size = max(1, int(num_bees * elite_ratio))
    convergence_steps = []
    evaluations = 0
    
    start_time = time.time()
    
    def evaluate_fitness(solution):
        total_weight = sum(w * s for w, s in zip(weights, solution))
        total_value = sum(v * s for v, s in zip(values, solution))
        return total_value if total_weight <= capacity else 0
    
    def generate_neighbor(solution, flip_prob=0.2):
        new_solution = solution[:]
        for i in range(len(solution)):
            if random.random() < flip_prob:
                new_solution[i] = 1 - new_solution[i]
        return new_solution
    
    bees = [[random.choice([0, 1]) for _ in range(n)] for _ in range(num_bees)]
    best_solution = max(bees, key=lambda s: evaluate_fitness(s))
    best_value = evaluate_fitness(best_solution)
    
    
    for iteration in range(max_iterations):
        fitnesses = [evaluate_fitness(bee) for bee in bees]
        evaluations += len(bees)
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        elites = [bees[i] for i in sorted_indices[:elite_size]]
        
        new_bees = elites[:]
        sorted_bees = [bees[i] for i in sorted_indices]
        while len(new_bees) < num_bees:
            parent = random.choice(elites + sorted_bees[:num_bees // 2])
            new_bees.append(generate_neighbor(parent, flip_prob=0.3))
        bees = new_bees
        
        current_best = max(bees, key=lambda s: evaluate_fitness(s))
        current_fit = evaluate_fitness(current_best)
        if current_fit > best_value:
            best_solution, best_value = current_best, current_fit
        
        convergence_steps.append(best_value)
    
    execution_time = time.time() - start_time
    
    total_weight = sum(w * s for w, s in zip(weights, best_solution))
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': evaluations,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_greedy_bco(num_items, max_weight, max_value, capacity_ratio, seed):
    """12. Greedy BCO - TOÀN BỘ QUY TRÌNH"""
    random.seed(seed)
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    num_bees = 20
    max_iterations = 100
    greedy_ratio = 0.3
    convergence_steps = []
    evaluations = 0
    
    start_time = time.time()
    
    def evaluate_fitness(solution):
        total_weight = sum(w * s for w, s in zip(weights, solution))
        total_value = sum(v * s for v, s in zip(values, solution))
        return total_value if total_weight <= capacity else 0
    
    def greedy_initial_solution():
        ratio = sorted([(values[i] / weights[i], i) for i in range(n)], reverse=True)
        solution = [0] * n
        total_weight = 0
        
        for r, i in ratio:
            if total_weight + weights[i] <= capacity:
                solution[i] = 1
                total_weight += weights[i]
        return solution
    
    def generate_neighbor(solution, flip_prob=0.2):
        new_solution = solution[:]
        for i in range(len(solution)):
            if random.random() < flip_prob:
                new_solution[i] = 1 - new_solution[i]
        return new_solution
    
    num_greedy = int(num_bees * greedy_ratio)
    base_greedy = greedy_initial_solution()
    bees = []
    
    for _ in range(num_greedy):
        sol = base_greedy[:]
        for i in range(len(sol)):
            if random.random() < 0.1:
                sol[i] = 1 - sol[i]
        bees.append(sol)
    
    for _ in range(num_bees - num_greedy):
        bees.append([random.choice([0, 1]) for _ in range(n)])
    
    best_solution = max(bees, key=lambda s: evaluate_fitness(s))
    best_value = evaluate_fitness(best_solution)
    
    
    for iteration in range(max_iterations):
        new_bees = []
        
        for bee in bees:
            neighbor = generate_neighbor(bee, flip_prob=0.3)
            eval_old = evaluate_fitness(bee)
            eval_new = evaluate_fitness(neighbor)
            evaluations += 2
            new_bees.append(neighbor if eval_new > eval_old else bee)
        
        bees = new_bees
        current_best = max(bees, key=lambda s: evaluate_fitness(s))
        current_fit = evaluate_fitness(current_best)
        if current_fit > best_value:
            best_solution, best_value = current_best, current_fit
        
        convergence_steps.append(best_value)
    
    execution_time = time.time() - start_time
    
    total_weight = sum(w * s for w, s in zip(weights, best_solution))
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': evaluations,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

def run_bco_hybrid(num_items, max_weight, max_value, capacity_ratio, seed):
    """13. BCO Hybrid - TOÀN BỘ QUY TRÌNH"""
    random.seed(seed)
    values, weights, capacity = generate_knapsack_problem(num_items, max_weight, max_value, capacity_ratio, seed)
    
    n = len(weights)
    num_bees = 20
    max_iterations = 100
    elite_size = 5
    convergence_steps = []
    evaluations = 0
    
    start_time = time.time()
    
    def evaluate_fitness(solution):
        total_weight = sum(w * s for w, s in zip(weights, solution))
        total_value = sum(v * s for v, s in zip(values, solution))
        if total_weight > capacity:
            return 0
        return total_value
    
    def greedy_probability_init(weights, values):
        n = len(weights)
        ratios = [values[i] / weights[i] for i in range(n)]
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        
        prob = []
        for ratio in ratios:
            if max_ratio == min_ratio:
                prob.append(0.6)
            else:
                normalized = (ratio - min_ratio) / (max_ratio - min_ratio)
                prob.append(0.3 + 0.6 * normalized)
        return prob
    
    def local_search(solution, weights, values, capacity):
        n = len(solution)
        best_solution = solution[:]
        best_fitness = evaluate_fitness(best_solution)
        improved = True
        
        while improved:
            improved = False
            for i in range(n):
                neighbor = best_solution[:]
                neighbor[i] = 1 - neighbor[i]
                fitness = evaluate_fitness(neighbor)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = neighbor[:]
                    improved = True
                    break
        
        return best_solution, best_fitness
    
    prob = greedy_probability_init(weights, values)
    best_solution = None
    best_value = 0
    
    for iteration in range(max_iterations):
        learning_rate = 0.3 * (1 - iteration / max_iterations)
        bees_solutions = []
        bees_fitness = []
        
        for bee_id in range(num_bees):
            solution = []
            current_weight = 0
            
            for i in range(n):
                if random.random() < prob[i]:
                    if current_weight + weights[i] <= capacity:
                        solution.append(1)
                        current_weight += weights[i]
                    else:
                        solution.append(0)
                else:
                    solution.append(0)
            
            fitness = evaluate_fitness(solution)
            evaluations += 1
            bees_solutions.append(solution)
            bees_fitness.append(fitness)
            
            if fitness > best_value:
                best_value = fitness
                best_solution = solution[:]
        
        if iteration % 10 == 0 and best_solution:
            improved_solution, improved_fitness = local_search(best_solution, weights, values, capacity)
            if improved_fitness > best_value:
                best_value = improved_fitness
                best_solution = improved_solution
        
        convergence_steps.append(best_value)
        
        sorted_indices = sorted(range(len(bees_fitness)), key=lambda i: bees_fitness[i], reverse=True)
        elite_solutions = [bees_solutions[i] for i in sorted_indices[:elite_size]]
        prob_update = [0] * n
        
        for elite_sol in elite_solutions:
            for i in range(n):
                if elite_sol[i] == 1:
                    prob_update[i] += learning_rate / elite_size
                else:
                    prob_update[i] -= learning_rate / elite_size
        
        for i in range(n):
            if prob_update[i] > 0:
                prob[i] = prob[i] + prob_update[i] * (1 - prob[i])
            else:
                prob[i] = prob[i] + prob_update[i] * prob[i]
            prob[i] = max(0.05, min(0.95, prob[i]))
    
    execution_time = time.time() - start_time
    
    total_weight = sum(w * s for w, s in zip(weights, best_solution))
    threshold = 0.9 * best_value
    conv_step = next((i for i, v in enumerate(convergence_steps) if v >= threshold), len(convergence_steps))
    
    return {
        'solution': best_solution,
        'max_value': best_value,
        'total_weight': total_weight,
        'capacity': capacity,
        'execution_time': execution_time,
        'nodes_visited': evaluations,
        'convergence_steps': convergence_steps,
        'conv_step': conv_step,
        'values': values,
        'weights': weights
    }

# =============================================================================
# MAP THUẬT TOÁN
# =============================================================================
ALGORITHM_RUNNER_MAP = {
    "Brute Force": run_brute_force,
    "Simple Backtracking": run_simple_backtracking,
    "Non-Recursive Backtracking": run_non_recursive_backtracking,
    "Weight Pruning": run_weight_pruning,
    "Upper Bound Pruning": run_upper_bound,
    "Ratio-Pruning Backtracking": run_ratio_pruning,
    "Branch & Bound": run_branch_and_bound,
    "BCO Basic": run_bco_basic,
    "BCO + Local Search": run_bco_local_search,
    "Adaptive BCO": run_adaptive_bco,
    "Elite BCO": run_elite_bco,
    "Greedy BCO": run_greedy_bco,
    "BCO Hybrid": run_bco_hybrid
}
# =============================================================================
# HÀM CHẠY THỰC NGHIỆM ĐƠN LẺ
# =============================================================================

def run_full_experiment(algorithm_name, num_items, max_weight, max_value, capacity_ratio, seed):
    try:
        runner_func = ALGORITHM_RUNNER_MAP.get(algorithm_name)
        if runner_func is None:
            return "Thuật toán không được hỗ trợ!", None, None
        
        result = runner_func(num_items, max_weight, max_value, capacity_ratio, seed)
        
        result_text = f"""
═══════════════════════════════════════════════════════════════════
                  KẾT QUẢ THÍ NGHIỆM: {algorithm_name:^30}         
═══════════════════════════════════════════════════════════════════
 THÔNG TIN BÀI TOÁN:
   • Số lượng vật phẩm: {num_items}
   • Sức chứa ba lô: {result['capacity']}
   • Seed: {seed}

 KẾT QUẢ:
   • Giá trị tối đa: {result['max_value']}
   • Tổng trọng lượng: {result['total_weight']} / {result['capacity']}
   • Tỷ lệ sử dụng: {(result['total_weight']/result['capacity']*100):.1f}%
   • Số items chọn: {sum(result['solution'])} / {num_items}

 HIỆU NĂNG:
   • Thời gian thực thi: {result['execution_time']:.6f} giây
   • Nodes/Evaluations: {result['nodes_visited']:,}
   • Bước hội tụ (90%): {result['conv_step']}

NGHIỆM: {result['solution']}
"""
        
        fig = plt.figure(figsize=(14, 6))
        threshold = 0.9 * result['max_value']
        conv = result['convergence_steps']
        conv_step = result['conv_step']
        
        plt.plot(range(len(conv)), conv, label="Best Fitness", color="blue", linewidth=2)
        plt.axhline(y=threshold, color="red", linestyle="--", label=f"90% Optimal ({threshold:.1f})", linewidth=2)
        plt.scatter(conv_step, conv[conv_step], color="green", s=100, zorder=5, label=f"Convergence ({conv_step})")
        plt.title(f"Convergence Process - {algorithm_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Best Value", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        table_data = [["Item", "Value", "Weight", "Ratio", "Selected"]]
        for i in range(num_items):
            ratio = result['values'][i] / result['weights'][i]
            selected = "✓" if result['solution'][i] == 1 else "✗"
            table_data.append([f"Item {i+1}", result['values'][i], result['weights'][i], f"{ratio:.2f}", selected])
        
        return result_text, fig, table_data
    except Exception as e:
        return f"Lỗi: {str(e)}", None, None

# =============================================================================
# HÀM SO SÁNH THUẬT TOÁN - PHẦN MỚI 
# =============================================================================

def compare_algorithms(selected_algorithms, num_items, max_weight, max_value, capacity_ratio, seed):
    """So sánh nhiều thuật toán với 5 biểu đồ trực quan"""
    if not selected_algorithms:
        return " Vui lòng chọn ít nhất 1 thuật toán!", None, None, None, None, None
    
    results = {}
    progress_text = " Đang chạy thí nghiệm...\n\n"
    
    for algo_name in selected_algorithms:
        try:
            runner_func = ALGORITHM_RUNNER_MAP.get(algo_name)
            if runner_func:
                result = runner_func(num_items, max_weight, max_value, capacity_ratio, seed)
                results[algo_name] = result
                progress_text += f" {algo_name}: Hoàn thành (Value={result['max_value']}, Time={result['execution_time']:.4f}s)\n"
        except Exception as e:
            progress_text += f" {algo_name}: Lỗi - {str(e)}\n"
    
    if not results:
        return " Không có kết quả nào!", None, None, None, None, None
    
    algo_names = list(results.keys())
    exec_times = [results[name]['execution_time'] for name in algo_names]
    nodes = [results[name]['nodes_visited'] for name in algo_names]
    max_values = [results[name]['max_value'] for name in algo_names]
    conv_steps = [results[name]['conv_step'] for name in algo_names]
    
    # ===== BIỂU ĐỒ 1: THỜI GIAN THỰC THI =====
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(algo_names)))
    bars1 = ax1.barh(algo_names, exec_times, color=colors1, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel(' Thời gian (giây)', fontsize=13, fontweight='bold')
    ax1.set_title(' So sánh Thời gian Thực thi', fontsize=15, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, time in zip(bars1, exec_times):
        ax1.text(time, bar.get_y() + bar.get_height()/2, f' {time:.4f}s', 
                va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    # ===== BIỂU ĐỒ 2: NODES VISITED =====
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(algo_names)))
    bars2 = ax2.barh(algo_names, nodes, color=colors2, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel(' Số Nodes/Evaluations', fontsize=13, fontweight='bold')
    ax2.set_title(' So sánh Số lượng Nodes Visited (Log Scale)', fontsize=15, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, node in zip(bars2, nodes):
        ax2.text(node, bar.get_y() + bar.get_height()/2, f' {node:,}', 
                va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    # ===== BIỂU ĐỒ 3: GIÁ TRỊ TỐI ƯU =====
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    colors3 = plt.cm.coolwarm(np.linspace(0, 1, len(algo_names)))
    bars3 = ax3.bar(range(len(algo_names)), max_values, color=colors3, edgecolor='black', linewidth=2)
    ax3.set_xticks(range(len(algo_names)))
    ax3.set_xticklabels(algo_names, rotation=30, ha='right', fontsize=11)
    ax3.set_ylabel(' Giá trị', fontsize=13, fontweight='bold')
    ax3.set_title(' So sánh Giá trị Tối ưu', fontsize=15, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars3, max_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkred')
    plt.tight_layout()
    
    # ===== BIỂU ĐỒ 4: QUÁ TRÌNH HỘI TỤ =====
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    colors4 = plt.cm.tab10(np.linspace(0, 1, len(algo_names)))
    for i, name in enumerate(algo_names):
        conv_data = results[name]['convergence_steps']
        if len(conv_data) > 2000:
            step = len(conv_data) // 2000
            conv_data = conv_data[::step]
        ax4.plot(range(len(conv_data)), conv_data, label=name, linewidth=2.5, 
                color=colors4[i], alpha=0.85, marker='o', markersize=3, markevery=len(conv_data)//10)
    ax4.set_xlabel(' Bước lặp', fontsize=13, fontweight='bold')
    ax4.set_ylabel(' Giá trị tốt nhất', fontsize=13, fontweight='bold')
    ax4.set_title(' So sánh Quá trình Hội tụ', fontsize=15, fontweight='bold')
    ax4.legend(loc='best', fontsize=11, framealpha=0.9)
    ax4.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    
    # ===== BIỂU ĐỒ 5: RADAR CHART =====
    fig5 = plt.figure(figsize=(10, 10))
    ax5 = fig5.add_subplot(111, projection='polar')
    
    max_time = max(exec_times) if max(exec_times) > 0 else 1
    max_nodes_val = max(nodes) if max(nodes) > 0 else 1
    max_val_val = max(max_values) if max(max_values) > 0 else 1
    max_conv_val = max(conv_steps) if max(conv_steps) > 0 else 1
    
    categories = ['Thời gian\n(thấp tốt hơn)', 'Nodes\n(thấp tốt hơn)', 
                  'Giá trị\n(cao tốt hơn)', 'Hội tụ\n(thấp tốt hơn)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax5.set_theta_offset(np.pi / 2)
    ax5.set_theta_direction(-1)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1)
    
    colors_radar = plt.cm.Set3(np.linspace(0, 1, len(algo_names)))
    for i, name in enumerate(algo_names):
        values_norm = [
            1 - (results[name]['execution_time'] / max_time),
            1 - (results[name]['nodes_visited'] / max_nodes_val),
            results[name]['max_value'] / max_val_val,
            1 - (results[name]['conv_step'] / max_conv_val)
        ]
        values_norm += values_norm[:1]
        ax5.plot(angles, values_norm, 'o-', linewidth=2.5, label=name, color=colors_radar[i])
        ax5.fill(angles, values_norm, alpha=0.15, color=colors_radar[i])
    
    ax5.set_title(' Radar Chart - Hiệu năng Tổng quan', fontsize=15, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    
    # ===== BẢNG SO SÁNH =====
    table_data = [["Thuật toán", "Giá trị", "Thời gian (s)", "Nodes", "Hội tụ"]]
    for name in algo_names:
        table_data.append([
            name,
            results[name]['max_value'],
            f"{results[name]['execution_time']:.4f}",
            f"{results[name]['nodes_visited']:,}",
            results[name]['conv_step']
        ])
    
    progress_text += f"\n Hoàn thành so sánh {len(results)} thuật toán!"
    
    return progress_text, fig1, fig2, fig3, fig4, table_data

# =============================================================================
# GIAO DIỆN GRADIO
# =============================================================================

with gr.Blocks(title="Knapsack Solver - Comparison Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    #  KNAPSACK PROBLEM SOLVER - SO SÁNH THUẬT TOÁN
    Ứng dụng này cho phép bạn chạy các thuật toán giải bài toán ba lô (0/1 Knapsack Problem) và so sánh hiệu năng của chúng thông qua các biểu đồ trực quan và bảng dữ liệu chi tiết.
    - Chọn thuật toán và cấu hình tham số để chạy thí nghiệm đơn lẻ.
    - Hoặc chọn nhiều thuật toán để so sánh hiệu năng với các biểu đồ và bảng tổng hợp.
    """)
    
    with gr.Tabs():
        # ===== TAB 1: CHẠY ĐƠN LẺ =====
        with gr.Tab(" Thí nghiệm Đơn lẻ"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  Cấu hình")
                    num_items_single = gr.Slider(5, 25, value=15, step=1, label="Số items")
                    max_weight_single = gr.Slider(10, 50, value=20, step=1, label="Max weight/item")
                    max_value_single = gr.Slider(50, 200, value=100, step=10, label="Max value/item")
                    capacity_ratio_single = gr.Slider(0.3, 0.7, value=0.5, step=0.05, label="Capacity ratio")
                    seed_single = gr.Number(value=42, label="Seed")
                    algorithm_choice = gr.Dropdown(
                        choices=list(ALGORITHM_RUNNER_MAP.keys()),
                        value="Branch & Bound",
                        label="Chọn thuật toán"
                    )
                    run_btn = gr.Button(" Chạy", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    result_text_single = gr.Textbox(label="Kết quả", lines=20)
                    result_plot_single = gr.Plot(label="Biểu đồ hội tụ")
                    result_table_single = gr.Dataframe(label="Bảng dữ liệu")
            
            run_btn.click(
                fn=run_full_experiment,
                inputs=[algorithm_choice, num_items_single, max_weight_single, 
                       max_value_single, capacity_ratio_single, seed_single],
                outputs=[result_text_single, result_plot_single, result_table_single]
            )
        
       # ===== TAB 2: SO SÁNH THUẬT TOÁN  =====
        with gr.Tab("So sánh Thuật toán"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Cấu hình So sánh")
                    num_items_comp = gr.Slider(5, 20, value=12, step=1, label="Số items")
                    max_weight_comp = gr.Slider(10, 50, value=20, step=1, label="Max weight/item")
                    max_value_comp = gr.Slider(50, 200, value=100, step=10, label="Max value/item")
                    capacity_ratio_comp = gr.Slider(0.3, 0.7, value=0.5, step=0.05, label="Capacity ratio")
                    seed_comp = gr.Number(value=42, label="Seed")
                    
                    algorithms_select = gr.CheckboxGroup(
                        choices=list(ALGORITHM_RUNNER_MAP.keys()),
                        value=["Branch & Bound", "BCO Basic"],
                        label="Chọn thuật toán để so sánh"
                    )
                    
                    compare_btn = gr.Button("So sánh Thuật toán", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    progress_output = gr.Textbox(label="Tiến trình", lines=8)
            
            # Mỗi hàng chỉ có 1 biểu đồ để hiển thị rõ ràng hơn
            with gr.Row():
                plot1 = gr.Plot(label="Thời gian thực thi")
            with gr.Row():
                plot2 = gr.Plot(label="Nodes visited")

            with gr.Row():
                plot3 = gr.Plot(label="Giá trị tối ưu")
            with gr.Row():
                plot4 = gr.Plot(label="Quá trình hội tụ")
            with gr.Row():
                comparison_table = gr.Dataframe(label="Bảng So sánh Tổng hợp")
            
            compare_btn.click(
                fn=compare_algorithms,
                inputs=[algorithms_select, num_items_comp, max_weight_comp, 
                    max_value_comp, capacity_ratio_comp, seed_comp],
                outputs=[progress_output, plot1, plot2, plot3, plot4, comparison_table]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)