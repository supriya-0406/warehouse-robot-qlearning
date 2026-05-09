import numpy as np

# Environment
rewards = np.array([
    [-1, -1, -1, -1, 100],
    [-1, -100, -1, -100, -1],
    [-1, -1, -1, -1, -1],
    [-100, -1, -100, -1, -1],
    [-1, -1, -1, -1, -1]
])

# Q-table
q_table = np.zeros((5, 5))

# Parameters
learning_rate = 0.8
discount_factor = 0.9
episodes = 100

# Training
for episode in range(episodes):
    state = np.random.randint(0, 5)

    while state != 4:
        action = np.random.randint(0, 5)

        reward = rewards[state][action]

        q_table[state][action] = q_table[state][action] + learning_rate * (
            reward + discount_factor * np.max(q_table[action]) - q_table[state][action]
        )

        state = action

# Output
print("Q-Table:\n")
print(q_table)

# Best path
print("\nOptimal Path:")

state = 0
path = [state]

while state != 4:
    next_state = np.argmax(q_table[state])
    path.append(next_state)
    state = next_state

print(path)
