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
import os
import csv

# === Pygame Setup ===
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RL F1 Car with Fuel & Pit Stops")

# === Create oval track mask with pit stop zone ===
def create_oval_track(surface, outer_rect, inner_rect, pit_rect):
    surface.fill((0, 0, 0))  # off track black background
    pygame.draw.ellipse(surface, (255, 255, 255), outer_rect)  # track white
    pygame.draw.ellipse(surface, (0, 0, 0), inner_rect)        # inside hole black (off track)
    # Draw pit stop zone in blue on track mask (to detect pit area)
    pygame.draw.rect(surface, (0, 0, 255), pit_rect)

# === Curriculum learning: Define track difficulties ===
curriculum_tracks = [
    { # Easy track
        "outer": pygame.Rect(100, 100, 600, 400),
        "inner": pygame.Rect(250, 200, 300, 200),
        "pit": pygame.Rect(150, 450, 100, 50),
    },
    { # Medium difficulty track: narrower track
        "outer": pygame.Rect(120, 120, 560, 360),
        "inner": pygame.Rect(280, 230, 240, 140),
        "pit": pygame.Rect(160, 440, 80, 40),
    },
    { # Hard difficulty track: even narrower and more challenging
        "outer": pygame.Rect(140, 140, 520, 320),
        "inner": pygame.Rect(310, 260, 180, 80),
        "pit": pygame.Rect(170, 430, 70, 35),
    },
]

# === Car class with sensors, fuel and pit stop logic ===
class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle

        # Realistic movement using 2D vector
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration_value = 0.15
        self.max_speed = 8
        self.rotation_speed = 5  # degrees per update
        self.friction = 0.02  # drag factor

        # Car size
        self.length = 60
        self.width = 30

        # Fuel and pit stop logic
        self.fuel = 1.0  # fuel level (0 to 1)
        self.fuel_consumption_rate = 0.00015
        self.pit_stops_made = 0
        self.in_pit_stop = False
        self.pit_stop_cooldown = 0

        # Car image
        self.image = pygame.image.load('f1_car.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (self.length, self.width))
        self.rect = self.image.get_rect(center=(self.x, self.y))

        # Sensor angles and distances
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensors = [1.0 for _ in self.sensor_angles]

    def update(self, action, track_mask):
        # === Fuel Consumption ===
        if self.velocity.length() > 0:
            self.fuel -= self.velocity.length() * self.fuel_consumption_rate
            if self.fuel < 0:
                self.fuel = 0
                self.velocity = pygame.Vector2(0, 0)

        # === Direction Vector ===
        rad = math.radians(self.angle)
        direction = pygame.Vector2(math.cos(rad), math.sin(rad))

        # === Handle Actions ===
        if action == 0 and self.fuel > 0:  # Accelerate
            self.velocity += direction * self.acceleration_value
        elif action == 3:  # Brake
            self.velocity *= 0.90
        elif action == 1:  # Turn right
            turn_scale = max(0.3, 1.0 - self.velocity.length() / self.max_speed)
            self.angle += self.rotation_speed * turn_scale
        elif action == 2:  # Turn left
            turn_scale = max(0.3, 1.0 - self.velocity.length() / self.max_speed)
            self.angle -= self.rotation_speed * turn_scale

        # === Clamp speed ===
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        # === Apply Friction ===
        self.velocity *= (1 - self.friction)

        # === Move Car ===
        self.x += self.velocity.x
        self.y += self.velocity.y
        self.rect.center = (self.x, self.y)

        # === Sensor Update ===
        self.update_sensors(track_mask)

        # === Pit Cooldown ===
        if self.pit_stop_cooldown > 0:
            self.pit_stop_cooldown -= 1

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
                if color == pygame.Color(0, 0, 0, 255):  # off track
                    distance = dist
                    break
            self.sensors.append(distance / max_distance)

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        new_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, new_rect.topleft)

        # Draw sensors
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            length = self.sensors[i] * 150
            end_x = self.x + length * math.cos(angle)
            end_y = self.y + length * math.sin(angle)
            pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (end_x, end_y), 2)

# === RL Environment wrapper with fuel and pit stop logic + curriculum difficulty ===
class F1Env:
    def __init__(self):
        self.car_start_x = SCREEN_WIDTH // 2
        self.car_start_y = 150
        self.curriculum_stage = 0  # Start at easiest track
        self.max_curriculum_stage = len(curriculum_tracks) - 1
        self.reset()

    def create_track_mask(self):
        track = curriculum_tracks[self.curriculum_stage]
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        create_oval_track(surface, track["outer"], track["inner"], track["pit"])
        return surface, track["pit"]

    def reset(self):
        self.track_mask, self.pit_zone = self.create_track_mask()
        self.car = Car(self.car_start_x, self.car_start_y, angle=0)
        self.steps = 0
        self.min_pit_stops = 2
        return self.get_state()

    def get_env_state(self):
        return {
            "car_x": self.car.x,
            "car_y": self.car.y,
            "car_angle": self.car.angle,
            "car_velocity": self.car.velocity,
            "car_fuel": self.car.fuel,
            "car_pit_stops_made": self.car.pit_stops_made,
            "curriculum_stage": self.curriculum_stage,
            "steps": self.steps
        }

    def set_env_state(self, state_dict):
        self.car.x = state_dict["car_x"]
        self.car.y = state_dict["car_y"]
        self.car.angle = state_dict["car_angle"]
        self.car.velocity = state_dict["car_velocity"]
        self.car.fuel = state_dict["car_fuel"]
        self.car.pit_stops_made = state_dict["car_pit_stops_made"]
        self.curriculum_stage = state_dict["curriculum_stage"]
        self.steps = state_dict["steps"]
        self.car.rect.center = (self.car.x, self.car.y)

    def get_state(self):
        norm_velocity = self.car.velocity.length() / self.car.max_speed
        norm_angle = (self.car.angle % 360) / 180 - 1
        norm_fuel = self.car.fuel  # already between 0 and 1
        norm_pit_stops = self.car.pit_stops_made / 5  # normalize max 5 pit stops for input
        state = self.car.sensors + [norm_velocity, norm_angle, norm_fuel, norm_pit_stops]
        return torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)

    def step(self, action):
        self.car.update(action, self.track_mask)
        self.steps += 1
        state = self.get_state()

        # Detect off track / crash
        center_color = self.track_mask.get_at((int(self.car.x), int(self.car.y)))
        done = False
        crash = False
        if center_color == pygame.Color(0, 0, 0, 255):
            done = True
            crash = True
            reward = -50  # bigger penalty for crashing
            return state, reward, done, {"crash": crash}

        # Base reward: velocity + avg sensor distance (encourage speed and staying on track)
        avg_sensor = sum(self.car.sensors) / len(self.car.sensors)
        reward = self.car.velocity.length() / self.car.max_speed + avg_sensor

        # Fuel penalty if running out of fuel
        if self.car.fuel <= 0:
            done = True
            reward = -20  # harsh penalty for running out of fuel
            return state, reward, done, {"crash": False}

        # Check if car is inside pit stop zone (approximate blue color)
        color = self.track_mask.get_at((int(self.car.x), int(self.car.y)))
        in_pit_zone = (color.b > 200 and color.r < 50 and color.g < 50)

        # Pit stop occurs if in pit zone AND velocity near zero AND cooldown is over
        if in_pit_zone and self.car.velocity.length() < 1.0 and self.car.pit_stop_cooldown == 0:
            self.car.pit_stop_cooldown = 50  # cooldown period (steps) to avoid counting multiple stops at once

            # Penalize if pit stop when fuel is still high (>0.3)
            if self.car.fuel > 0.3:
                reward -= 5  # penalty for unnecessary pit stop
            else:
                reward += 10
                self.car.fuel = 1.0
                self.car.pit_stops_made += 1

        # End episode if max steps reached
        if self.steps > 2000:
            done = True
            if self.car.pit_stops_made >= self.min_pit_stops:
                reward += 20
                if self.curriculum_stage < self.max_curriculum_stage:
                    print(f"🔼 Increasing difficulty to stage {self.curriculum_stage + 1}")
                    self.curriculum_stage += 1
            else:
                reward -= 10  # penalty for missing required pit stops

        return state, reward, done, {"crash": crash}

    def render(self):
        screen.fill((50, 50, 50))
        screen.blit(self.track_mask, (0, 0))

        pygame.draw.rect(screen, (0, 0, 255), self.pit_zone, 2)

        font = pygame.font.SysFont(None, 24)
        pit_label = font.render("Pit Stop (Refuel)", True, (0, 0, 255))
        label_x = self.pit_zone.x + self.pit_zone.width // 2 - pit_label.get_width() // 2
        label_y = self.pit_zone.y - 25
        screen.blit(pit_label, (label_x, label_y))

        self.car.draw(screen)

        fuel_bar_length = 100
        fuel_bar_height = 10
        fuel_x = 10
        fuel_y = 10
        pygame.draw.rect(screen, (100, 100, 100), (fuel_x, fuel_y, fuel_bar_length, fuel_bar_height))
        pygame.draw.rect(screen, (0, 255, 0), (fuel_x, fuel_y, fuel_bar_length * self.car.fuel, fuel_bar_height))
        fuel_label = font.render("Fuel", True, (255, 255, 255))
        screen.blit(fuel_label, (fuel_x, fuel_y - 20))

        pit_count_label = font.render(f"Pit Stops: {self.car.pit_stops_made}", True, (255, 255, 255))
        screen.blit(pit_count_label, (10, 40))

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
        self.policy_net = Network(input_size, nb_action).to(DEVICE)
        self.target_net = Network(input_size, nb_action).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.batch_size = 64
        self.steps_done = 0
        self.epsilon_start = 0.5
        self.epsilon_end = 0.01
        self.epsilon_decay = 3000
        self.update_target_steps = 1000
        self.learn_steps = 0

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
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

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# === Training loop + plotting ===
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
            writer.writerow(["Episode", "TotalReward", "ModelFile", "CurriculumStage"])

    best_total_reward = float('-inf')
    best_model_path = os.path.join(model_folder, "best_model.pth")

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
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            env.render()

        print(f"Episode {episode} reward: {total_reward:.2f} (Stage {env.curriculum_stage})")

        # Save episode model
        model_name = f"model_{episode}.pth"
        model_path = os.path.join(model_folder, model_name)
        torch.save(agent.policy_net.state_dict(), model_path)

        # Save log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, model_name, env.curriculum_stage])

        # Save best model
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            torch.save(agent.policy_net.state_dict(), best_model_path)
            print(f"✅ Best model updated at episode {episode} with reward {total_reward:.2f}")

        # === Revert to best model on crash ===
        if info.get("crash", False):
            if os.path.exists(best_model_path):
                agent.policy_net.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
                agent.policy_net.eval()
                print(f"🔄 Crash detected. Reverted to best model from {best_model_path}")

# === Main ===
if __name__ == "__main__":
    input_size = 9  # 5 sensors + velocity + angle + fuel + pit stops count
    nb_action = 4   # accelerate, turn right, turn left, no action
    env = F1Env()
    agent = DQNAgent(input_size, nb_action)
    train(agent, env, episodes=500)