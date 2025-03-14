import numpy as np

# Initialize
N = 4  # Grid size (NxN)
gamma = 1.0  # Discount factor (no discounting)
theta = 1e-4  # Convergence threshold
actions = ['up', 'down', 'left', 'right']

# Rewards
rewards = -1 * np.ones((N, N))
rewards[-1, -1] = 0  # Terminal state reward

# Initialize value function V(s) = 0 for all states
V = np.zeros((N, N))

# Helper function: Given state (i, j) and action, return next state (ni, nj)
def next_state(i, j, action):
    if action == 'up':
        return max(i - 1, 0), j
    elif action == 'down':
        return min(i + 1, N - 1), j
    elif action == 'left':
        return i, max(j - 1, 0)
    elif action == 'right':
        return i, min(j + 1, N - 1)

# Value Iteration
iteration = 0
while True:
    delta = 0  # Track maximum change
    V_new = V.copy()
    
    # Loop over each state in the grid
    for i in range(N):
        for j in range(N):
            if (i, j) == (N - 1, N - 1):  # Terminal state, skip updating
                continue
            
            v = V[i, j]
            
            # For each action, calculate next state and get its value
            new_values = []
            for action in actions:
                ni, nj = next_state(i, j, action)
                reward = rewards[ni, nj]
                new_values.append(1/4 * (reward + gamma * V[ni, nj]))  # Equal probability for all moves
            
            # Bellman update: sum over all actions (average since equiprobable)
            V_new[i, j] = sum(new_values)
            
            # Update delta
            delta = max(delta, abs(v - V_new[i, j]))
    
    V = V_new
    iteration += 1
    
    # Check for convergence
    if delta < theta:
        break

# Print final value function
print(f"Converged after {iteration} iterations.\n")
print(np.round(V, decimals=8))
