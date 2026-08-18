import os
import sys
import subprocess

print("===================================================================")
print("Week 12: Adaptive Evacuation GNN-DQN (Final Project Launcher)")
print("===================================================================")

print("\nThis script acts as the entry point for the final internship project.")
print("Because the final project is massive and contains its own UI, models,")
print("and training environments, it is housed in the root directory under:")
print("-> /Adaptive-Evacuation-GNN-DQN/")

# Path to the main project
main_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Adaptive-Evacuation-GNN-DQN'))
main_script = os.path.join(main_project_dir, 'main.py')

print(f"\nLooking for final project at:\n{main_project_dir}")

if not os.path.exists(main_script):
    print("\nError: Could not find main.py in the project directory.")
    sys.exit(1)

print("\nProject found! How would you like to run the final project?")
print("1. Run Training (GNN-DQN)")
print("2. Run Evaluation (Test Trained Model)")
print("3. Start WebSocket Backend (For React Dashboard)")
print("4. Exit")

# For the sake of this automated script, we will just print the commands
# rather than blocking with input(). In a real terminal, the student can use it.
print("\n[To execute manually, run the following commands from the root directory]")
print("Train:  python main.py --mode train-gnn")
print("Eval:   python main.py --mode evaluate")
print("Server: python visualization/server.py")
print("===================================================================")
