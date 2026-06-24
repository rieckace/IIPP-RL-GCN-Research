"""
GridWorld Q-Learning Implementation

This project implements a simple GridWorld environment and a Q-Learning
agent from scratch to understand the fundamentals of Reinforcement Learning.

Concepts Covered:
- States and Actions
- Reward Function
- Exploration vs Exploitation
- Q-Table
- Q-Learning
- Epsilon-Greedy Policy

Author: Rikesh Yadav
Internship: INEC Lab, Yuan Ze University
Date: June 2026
"""

# Create the Environment

import random
from collections import defaultdict

# GridWorld Environment
GRID_SIZE = 3
grid = [
    ["S", ".", "."],
    [".", "X", "."],
    [".", ".", "G"]
]

    
agent_pos = [0,0]
goal_pos = [2,2]

# print("\nAgent Position: ", agent_pos)
# print("Goal Position: ", goal_pos)

actions = ["RIGHT", "LEFT", "UP", "DOWN"]
print("\nActions available: ")
for action in actions:
    print(action)

# Q-learning settings
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05
num_episodes = 3000
max_steps_per_episode = 50

q_table = defaultdict(lambda: {action: 0.0 for action in actions})


def reset_agent():
    global agent_pos
    agent_pos = [0, 0]
    return tuple(agent_pos)


def get_state():
    return tuple(agent_pos)


def choose_action(state, exploration_rate):
    if random.random() < exploration_rate:
        return random.choice(actions)

    return max(q_table[state], key=q_table[state].get)
    
# Visualize the Agent
# print("\nAgent Current State: ")

for i in range(3):
    row = []
    for j in range(3):

        if [i, j] == agent_pos:
            row.append("A")

        else:
            row.append(grid[i][j])

    # print(row)

    
# Function Skeleton
def move_agent(action):
    global agent_pos
    old_position = agent_pos.copy()
    reward = -1
    done = False
    
    # Moving the agent based on action
    # RIGHT MOVEMENT
    if action == "RIGHT":

        if agent_pos[1] < GRID_SIZE - 1:
            agent_pos[1] += 1
            
        else:
            reward = -5
            
    # Moving left
    elif action =="LEFT":
        if agent_pos[1] > 0:
            agent_pos[1] -=1
        else:
            reward = -5
            
    # Moving UP
    elif action == "UP":
        if agent_pos[0] > 0:
            agent_pos[0] -= 1;
        else:
            reward = -5
            
    # MOving Down
    elif action == "DOWN":
        if agent_pos[0] < GRID_SIZE - 1:
            agent_pos[0] += 1
            
        else:
            reward = -5
            
            
    # Check Obstacle
    if agent_pos == [1,1]:
        reward = -10
        agent_pos = old_position
        
    #   Check Goal
    if agent_pos == goal_pos:
        reward = 10
        done = True
        
# Return Values
    return tuple(agent_pos), reward, done

def display_grid():

    for i in range(GRID_SIZE):

        row = []

        for j in range(GRID_SIZE):

            if [i, j] == agent_pos:
                row.append("A")

            else:
                row.append(grid[i][j])

        print(row)

    print()


def train_q_learning():
    global epsilon

    for episode in range(num_episodes):
        reset_agent()
        state = get_state()

        for _ in range(max_steps_per_episode):
            action = choose_action(state, epsilon)
            next_state, reward, done = move_agent(action)

            best_next_value = 0 if done else max(q_table[next_state].values())
            current_value = q_table[state][action]

            q_table[state][action] = current_value + alpha * (
                reward + gamma * best_next_value - current_value
            )

            state = next_state

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}: epsilon={epsilon:.3f}")


def get_best_action(state):
    return max(q_table[state], key=q_table[state].get)


def run_trained_agent():
    reset_agent()
    state = get_state()

    print("\nGreedy run after training:")
    display_grid()

    for _ in range(max_steps_per_episode):
        action = get_best_action(state)
        next_state, reward, done = move_agent(action)

        print("Action:", action)
        print("State:", next_state)
        print("Reward:", reward)
        display_grid()

        state = next_state

        if done:
            print("✅ Goal reached by the trained agent.")
            break

episode_actions = ["RIGHT", "DOWN","UP","RIGHT", "DOWN","DOWN"]
print("\nResult from the function:")
for action in episode_actions:
    state, reward, done = move_agent(action)
   
    print("Agent Position:", agent_pos)
    print("Action: ", action)
    print("State:", state)
    print("Reward:", reward)
    print("---------------\n")
    
    if done:
        print("\n✅✅ Goal Reached.")
        break


print("\nTraining Q-learning agent...")
train_q_learning()
run_trained_agent()