# Week 1 - Grid World Q-Learning

This folder contains a simple **Grid World** project for learning the basics of **Q-learning** in Reinforcement Learning.

## What this project does

The agent starts in a 3x3 grid and learns how to move from the **start position** to the **goal** while avoiding the **obstacle**.

### Grid setup

- **S** = Start position
- **G** = Goal position
- **X** = Obstacle
- **A** = Agent

The grid looks like this:

```text
S . .
. X .
. . G
```

## Main idea

The agent learns by trying actions and receiving rewards:

- `-1` for each normal move
- `-5` for trying to move outside the grid
- `-10` for hitting the obstacle
- `+10` for reaching the goal

The agent uses a **Q-table** to remember which action is better in each state.

## Files

- `gridWorld.py` - main Python file with the environment and Q-learning code

## How the agent learns

The program uses:

- **alpha** - learning rate
- **gamma** - discount factor
- **epsilon** - exploration rate

At first, the agent explores more. After training for many episodes, it starts choosing the best actions more often.

## How to run

Run the Python file from this folder:

```bash
python gridWorld.py
```

## What you will see

When you run the file, it will:

1. Show the available actions
2. Test a sample action sequence
3. Train the Q-learning agent
4. Run the trained agent using the learned policy

## Important note

This is a basic learning project. It is enough for a small Q-learning demo, but it can still be improved by:

- tracking rewards per episode
- plotting learning progress
- saving the Q-table
- making the grid larger
- adding more obstacles or goals

## Goal of the project

The goal is to make the agent learn on its own how to reach the goal by using trial and error instead of hard-coded movement rules.

## Author

Created as part of the IIPP Taiwan Internship RL Week 1 work.
