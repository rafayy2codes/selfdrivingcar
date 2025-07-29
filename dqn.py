import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Network
class Network(nn.Module):
    def __init__(self, input_size: int, nb_action: int):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, nb_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# DQN Agent
class DQNAgent:
    def __init__(self, input_size: int, nb_action: int, gamma: float = 0.9, lr=1e-3):
        self.gamma = gamma
        self.input_size = input_size
        self.nb_action = nb_action
        self.model = Network(input_size, nb_action).to(DEVICE)
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = 64

        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000

    def select_action(self, state: torch.Tensor) -> int:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  torch.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() > epsilon.item():
            with torch.no_grad():
                q_values = self.model(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.nb_action)

        return action

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float).to(DEVICE)
        next_states = torch.cat(next_states).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)

        q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.model(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()

# === Live plotting setup ===
plt.ion()
fig, ax = plt.subplots()
episode_rewards = []

def update_plot(new_reward):
    episode_rewards.append(new_reward)
    ax.clear()
    ax.plot(episode_rewards, label="Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.pause(0.001)

# === Example training loop ===
def train(agent, env, num_episodes=500):
    for episode in range(num_episodes):
        state = env.reset()
        # Convert state to tensor, add batch dimension
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)

        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(DEVICE)

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        print(f"Episode {episode} Reward: {total_reward}")
        update_plot(total_reward)

    # Save final plot
    plt.ioff()
    plt.savefig("training_rewards.png")
    plt.show()

# === Main ===
if __name__ == "__main__":
    # Initialize your environment here
    # For example: env = YourCustomEnv()
    # Replace with your actual environment

    input_size = 5      # e.g. number of state inputs
    nb_action = 3       # number of actions

    agent = DQNAgent(input_size, nb_action)
    
    # train(agent, env)  # Uncomment this line and provide your env

    print("Setup done! Run train(agent, env) with your environment to start training.")
