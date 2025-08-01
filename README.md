# RL Car with Fuel & Pit Stops

## Overview

This project demonstrates a Reinforcement Learning (RL) agent controlling a car navigating an oval track with realistic features such as fuel consumption and mandatory pit stops. The agent learns using Deep Q-Learning (DQN) with curriculum learning, progressively tackling harder tracks.

The environment is built using **Pygame** for 2D rendering and physics simulation, while the RL agent is implemented with **PyTorch**. The project integrates perception (sensors), car dynamics, fuel management, and strategy (pit stops) to provide a rich RL challenge.

---

## Features

- Oval track environment with three difficulty stages (easy, medium, hard), implemented via masks and collision detection.  
- Car physics simulation including acceleration, turning, friction, velocity, and rotation.  
- Sensors simulate distance measurements at multiple angles for environment perception.  
- Fuel consumption during movement and mandatory pit stops to refuel.  
- Deep Q-Network (DQN) RL agent with experience replay and target network.  
- Curriculum learning to progressively increase track difficulty based on performance.  
- Model checkpointing and training logs in CSV format.  
- Real-time rendering with Pygame and live reward plotting with Matplotlib.  

---

## Project Structure

- **Car class:** Models the car's physics, sensors, fuel, and pit stop logic.  
- **F1Env class:** RL environment wrapper handling track creation, step logic, state representation, and rendering.  
- **Network class:** Neural network architecture for DQN agent.  
- **DQNAgent class:** Handles action selection, experience replay memory, learning/updating policy and target networks.  
- **train() function:** Training loop over multiple episodes, including event handling, training steps, logging, and model saving.  
- **Main execution block:** Initializes environment and agent, then starts training.  

---

## How It Works

### Environment

- The environment simulates a top-down 2D oval race track with an inner hole representing grass/off-track.  
- A blue rectangle defines the pit stop zone where the car can refuel if it stops there.  
- The track mask surface is used to detect off-track (black areas), on-track (white), and pit stop areas (blue).  

### Car Dynamics & Sensors

- The car has realistic movement: acceleration, turning (scaled with speed), friction, and velocity capping.  
- Five sensors cast rays at -60°, -30°, 0°, +30°, +60° relative to the car’s heading to measure normalized distance to track edges.  
- Sensor data, velocity, angle, fuel level, and pit stop count form the RL state input.  

### Fuel & Pit Stop Logic

- Fuel depletes proportional to car speed; if fuel runs out, the car stops and episode ends.  
- To refuel, the car must enter the pit stop zone with near-zero velocity and cooldown must allow it.  
- Pit stops when done correctly provide positive reward and reset fuel; unnecessary pit stops are penalized.  

### Reinforcement Learning Agent

- Uses Deep Q-Learning with a feedforward neural network (2 hidden layers, 64 neurons each).  
- Experience replay memory stores transitions for batch training.  
- Epsilon-greedy policy for balancing exploration and exploitation with decaying epsilon.  
- Target network updated periodically to stabilize learning.  
- Rewards shaped to encourage staying on track, speed, proper pit stops, and penalize crashes or running out of fuel.  

### Curriculum Learning

- The agent starts on the easiest track configuration.  
- Upon successfully completing episodes with required pit stops, the difficulty increases to narrower tracks.  

---

## Getting Started

*Instructions to run the project and setup dependencies would go here.*

---

## Training

*Explain the training procedure and parameters.*

---

## Visualizing Results

*Explain how to visualize training progress.*

---

## Future Improvements

*List possible extensions and improvements.*

---

## Dependencies

- Python 3.8+  
- pygame  
- torch (PyTorch)  
- matplotlib  

---
