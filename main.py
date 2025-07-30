import pygame
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import sys

# === Pygame Setup ===
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RL F1 Car")

# === Create oval track mask ===
def create_oval_track(surface, outer_rect, inner_rect):
    surface.fill((0, 0, 0))  # off track black background
    pygame.draw.ellipse(surface, (255, 255, 255), outer_rect)  # track white
    pygame.draw.ellipse(surface, (0, 0, 0), inner_rect)        # inside hole black (off track)

track_mask = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
outer = pygame.Rect(100, 100, 600, 400)  # Outer ellipse bounds
inner = pygame.Rect(200, 175, 400, 250)  # Inner ellipse bounds (track hole)
create_oval_track(track_mask, outer, inner)

# === Car class with sensors ===
class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0.0
        self.length = 60
        self.width = 30
        self.max_velocity = 8
        self.acceleration = 0.3
        self.rotation_speed = 5
        self.image = pygame.image.load('f1_car.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (self.length, self.width))
        self.rect = self.image.get_rect(center=(self.x, self.y))
        # Increased sensors: front wide spread for better sensing
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensors = [1.0 for _ in self.sensor_angles]

    def update(self, action, track_mask):
        # Actions: 0 = accelerate, 1 = turn right, 2 = turn left, 3 = no action (slow down)
        if action == 0:
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
        elif action == 1:
            self.angle += self.rotation_speed
        elif action == 2:
            self.angle -= self.rotation_speed
        elif action == 3:
            self.velocity = max(self.velocity - self.acceleration, 0)

        rad = math.radians(self.angle)
        self.x += self.velocity * math.cos(rad)
        self.y += self.velocity * math.sin(rad)
        self.rect.center = (self.x, self.y)
        self.update_sensors(track_mask)

        # Friction slows down velocity gradually
        self.velocity = max(self.velocity * 0.95, 0)

    def update_sensors(self, track_mask):
        max_distance = 150
        self.sensors = []
        for sensor_angle in self.sensor_angles:
            angle = math.radians(self.angle + sensor_angle)
            distance = max_distance
            for dist in range(max_distance):
                check_x = int(self.x + dist * math.cos(angle))
                check_y = int(self.y + dist * math.sin(angle))
                if (check_x < 0 or check_y < 0 or
                    check_x >= track_mask.get_width() or
                    check_y >= track_mask.get_height()):
                    distance = dist
                    break
                color = track_mask.get_at((check_x, check_y))
                if color == pygame.Color(0, 0, 0, 255):  # off track = black pixel
                    distance = dist
                    break
            self.sensors.append(distance / max_distance)

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        new_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, new_rect.topleft)
        # Draw sensor rays
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            length = self.sensors[i] * 150
            end_x = self.x + length * math.cos(angle)
            end_y = self.y + length * math.sin(angle)
            pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (end_x, end_y), 2)

# === RL Environment wrapper ===
class F1Env:
    def __init__(self):
        # Start position on track white area (top-middle on oval track)
        self.car_start_x = SCREEN_WIDTH // 2
        self.car_start_y = 150
        self.reset()

    def reset(self):
        self.car = Car(self.car_start_x, self.car_start_y, angle=0)
        return self.get_state()

    def get_state(self):
        norm_velocity = self.car.velocity / self.car.max_velocity
        norm_angle = (self.car.angle % 360) / 180 - 1
        state = self.car.sensors + [norm_velocity, norm_angle]
        return torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)

    def step(self, action):
        self.car.update(action, track_mask)
        state = self.get_state()
        center_color = track_mask.get_at((int(self.car.x), int(self.car.y)))
        done = center_color == pygame.Color(0, 0, 0, 255)
        # Reward = velocity + average sensor distance (encourage speed & staying centered)
        avg_sensor = sum(self.car.sensors) / len(self.car.sensors)
        reward = self.car.velocity / self.car.max_velocity + avg_sensor
        if done:
            reward = -10
        return state, reward, done, {}

    def render(self):
        screen.fill((50, 50, 50))
        screen.blit(track_mask, (0, 0))
        self.car.draw(screen)
        pygame.display.flip()

# === DQN Network and Agent ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, nb_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, input_size, nb_action, gamma=0.9, lr=1e-3):
        self.gamma = gamma
        self.input_size = input_size
        self.nb_action = nb_action
        self.model = Network(input_size, nb_action).to(DEVICE)
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = 64
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 3000

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.model(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.nb_action)
        return action

    def store_transition(self, state, action, reward, next_state, done):
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

# === Training loop + plotting ===
# === Training loop + plotting + saving ===
import os
import csv
from datetime import datetime

plt.ion()
fig, ax = plt.subplots()
episode_rewards = []

def update_plot(new_reward):
    episode_rewards.append(new_reward)
    ax.clear()
    ax.plot(episode_rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards")
    plt.pause(0.001)

def train(agent, env, episodes=300):
    model_folder = "models"
    os.makedirs(model_folder, exist_ok=True)

    log_path = os.path.join(model_folder, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "ModelFile"])

    best_total_reward = float('-inf')

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            env.render()

        print(f"Episode {episode} reward: {total_reward:.2f}")
        update_plot(total_reward)

        # Save model
        model_name = f"model_{episode}.pth"
        model_path = os.path.join(model_folder, model_name)
        torch.save(agent.model.state_dict(), model_path)

        # Save log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, model_name])

        # Save best model
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            torch.save(agent.model.state_dict(), os.path.join(model_folder, "best_model.pth"))
            print(f"✅ Best model updated at episode {episode} with reward {total_reward:.2f}")

    # Final plot
    plt.ioff()
    plt.savefig(os.path.join(model_folder, "rewards_plot.png"))
    plt.show()
    print(f"\n🎉 Training complete! Models & logs saved in '{model_folder}/'")

# === Main ===
if __name__ == "__main__":
    input_size = 7  # 5 sensors + velocity + angle
    nb_action = 4   # accelerate, turn right, turn left, no action
    env = F1Env()
    agent = DQNAgent(input_size, nb_action)
    train(agent, env, episodes=500)