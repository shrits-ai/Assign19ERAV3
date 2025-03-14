# 🔢 4x4 GridWorld Value Iteration using Bellman Updates

This project implements **Value Iteration** using the **Bellman equation** in a simple 4x4 GridWorld environment. The goal is to compute the **state-value function** `V(s)` for each state, representing the expected return when starting from that state and following an optimal policy.

---

## 🌍 Environment Description

- **Grid Size**: 4x4  
- **States**: 16 (labeled 0 to 15)  
- **Start State**: Top-left corner (state 0)  
- **Terminal State**: Bottom-right corner (state 15)  
- **Actions**: Up, Down, Left, Right  
- **Transition Model**: Equal probability (1/4) of moving in any direction  
- **Rewards**:
  - `-1` for each move  
  - `0` at the terminal state (no further rewards)  
- **Discount Factor (γ)**: 1.0 (no discounting)  
- **Convergence Threshold**: `1e-4`  

---

## 🧮 Bellman Update Logic

For each non-terminal state `s`, we update its value using the Bellman equation:

```
V(s) = Σ_a P(s'|s,a) * [ R(s') + γ * V(s') ]
```
Where:

a is the action (up, down, left, right)
P(s'|s,a) is the transition probability (0.25 for each action)
R(s') is the reward at the next state
γ is the discount factor (1.0 in this case)
The iteration stops when the maximum change in V(s) across all states is less than theta = 1e-4.


Where:
- `a` is the action (up, down, left, right)  
- `P(s'|s,a)` is the transition probability (0.25 for each action)  
- `R(s')` is the reward at the next state  
- `γ` is the discount factor (1.0 in this case)

The iteration stops when the maximum change in `V(s)` across all states is less than `theta = 1e-4`.

---

## 🚀 How to Run

### Prerequisites
- Python 3.x  
- No external libraries needed (uses NumPy, part of standard packages)

### Run the Code
```
python3 bellmanUpdate.py
```

### Output
```
Converged after 470 iterations.

[[-58.42367735 -56.42387125 -53.2813141  -50.71012579]
 [-56.42387125 -53.56699476 -48.71029394 -44.13926711]
 [-53.2813141  -48.71029394 -39.85391609 -28.99766609]
 [-50.71012579 -44.13926711 -28.99766609   0.        ]]
```
📝 References
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
Bellman, R. (1957). Dynamic Programming
