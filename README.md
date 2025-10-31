### **Deep Reinforcement Learning for Delayed Rewards (Experiment 1)**

This project explores Deep Reinforcement Learning (Deep RL) techniques for solving delayed reward problems in decision-making processes. It is implemented as a single, self-contained Jupyter notebook (Ex1.ipynb).

**🧠 Overview**

The notebook sets up a graph-based environment (an 8×8 grid world) where an agent performs random walks and receives rewards only after a delay. The main challenge is to efficiently assign credit to past actions that contributed to delayed rewards — known as the temporal credit assignment problem.

**⚙️ Key Features**

Implementation of an 8×8 grid environment using PyTorch Geometric and NetworkX.

Setup for model-free reinforcement learning with delayed episodic rewards.

Visualization of agent policies and trajectories.

Focus on reward redistribution to convert delayed feedback into more immediate learning signals.

**🧩 Technologies Used**

Python, PyTorch, PyTorch Geometric

NumPy, NetworkX, Matplotlib, Seaborn

**🚀 Future Work**

This experiment is part of a larger research direction exploring multiple delayed-reward settings and different neural architectures. Future notebooks will include deeper policy evaluation and gradient-based optimization.

**🗂️ File**

Ex1.ipynb – Complete, independent experiment notebook.
