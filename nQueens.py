# Please read README.txt for info on usage
#
# Umar Tariq (umartariq)
# December 2024

import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
import sys
import time

# Backtracking Algorithm

def is_safe(board, row, col, n):
    # Check if placing a queen at (row, col) is safe
    for i in range(col):
        if board[i] == row:
            return False
        if abs(board[i] - row) == abs(i - col):
            return False
    return True

def solve_backtracking(board, col, n):
    # Recursive backtracking function to place queens
    if col >= n:
        return True
    for i in range(n):
        if is_safe(board, i, col, n):
            board[col] = i
            if solve_backtracking(board, col + 1, n):
                return True
            board[col] = -1
    return False

# Hill Climbing Algorithm

def get_conflicts(board, n):
    # Calculate number of conflicts in the current board state
    row_conflicts = np.zeros(n)
    diag1_conflicts = np.zeros(2 * n - 1)
    diag2_conflicts = np.zeros(2 * n - 1)

    for col in range(n):
        row = board[col]
        row_conflicts[row] += 1
        diag1_conflicts[row + col] += 1
        diag2_conflicts[row - col + n - 1] += 1

    conflicts = 0
    for col in range(n):
        row = board[col]
        conflicts += (row_conflicts[row] - 1)
        conflicts += (diag1_conflicts[row + col] - 1)
        conflicts += (diag2_conflicts[row - col + n - 1] - 1)
    return conflicts // 2

def hill_climbing(n, max_restarts=10):
    # Start with multiple restarts to avoid local minima
    for restart in range(max_restarts):
        board = np.random.permutation(n)
        current_conflicts = get_conflicts(board, n)
        iterations = 0
        while current_conflicts != 0:
            iterations += 1
            neighbor = board.copy()
            min_conflicts = current_conflicts
            for col in range(n):
                original_row = board[col]
                for row in range(n):
                    if row != original_row:
                        neighbor[col] = row
                        conflicts = get_conflicts(neighbor, n)
                        if conflicts < min_conflicts:
                            min_conflicts = conflicts
                            board[col] = row
                neighbor[col] = original_row
            if min_conflicts == current_conflicts:
                break
            current_conflicts = min_conflicts
        if current_conflicts == 0:
            return board, iterations
    return None, iterations

# A* Algorithm

class NQueensState:
    def __init__(self, board, cost, heuristic):
        self.board = board
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def heuristic_conflicts(board, n):
    return get_conflicts(board, n)

def a_star(n):
    initial_board = [-1] * n
    open_list = []
    heapq.heappush(open_list, NQueensState(initial_board, 0, n))
    nodes_expanded = 0

    while open_list:
        nodes_expanded += 1
        current_state = heapq.heappop(open_list)

        if current_state.cost == n:
            return current_state.board, nodes_expanded

        for row in range(n):
            if row not in current_state.board:
                new_board = current_state.board.copy()
                new_board[current_state.cost] = row
                h = heuristic_conflicts(new_board, n)
                new_state = NQueensState(new_board, current_state.cost + 1, h)
                heapq.heappush(open_list, new_state)

    return None, nodes_expanded

# Display Board

def display_board(board, title):
    n = len(board)
    visual_board = np.zeros((n, n), dtype=int)
    for col, row in enumerate(board):
        visual_board[row][col] = 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(visual_board, cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.xticks(range(n), range(1, n+1))
    plt.yticks(range(n), range(1, n+1))
    plt.show()

def solve_n_queens(n, method):
    start_time = time.time()

    if method == 'backtracking':
        board = [-1] * n
        success = solve_backtracking(board, 0, n)
        elapsed_time = time.time() - start_time
        return board if success else None, elapsed_time, None

    elif method == 'hill_climbing':
        result, iterations = hill_climbing(n)
        elapsed_time = time.time() - start_time
        return result, elapsed_time, iterations

    elif method == 'a_star':
        result, nodes_expanded = a_star(n)
        elapsed_time = time.time() - start_time
        return result, elapsed_time, nodes_expanded

    return None, None, None

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python n_queens_solver_fixed.py <board_size> <method>")
        print("Methods: backtracking, hill_climbing, a_star")
        sys.exit(1)

    n = int(sys.argv[1])
    method = sys.argv[2].lower()

    if method not in ['backtracking', 'hill_climbing', 'a_star']:
        print("Invalid method. Choose from: backtracking, hill_climbing, a_star")
        sys.exit(1)

    solution, elapsed_time, stat = solve_n_queens(n, method)

    if solution is not None:
        print(f"{method.capitalize()} Solution found in {elapsed_time:.4f} seconds.")
        if method == 'hill_climbing':
            print(f"Number of iterations: {stat}")
        elif method == 'a_star':
            print(f"Nodes expanded: {stat}")
        display_board(solution, f"{method.capitalize()} Solution")
    else:
        print(f"{method.capitalize()} failed to find a solution.")
