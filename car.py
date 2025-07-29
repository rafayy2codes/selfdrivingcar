import pygame
import math
import numpy as np

class Car:
    def __init__(self, x: float, y: float, angle: float = 0.0):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0.0
        self.length = 60  # Approximate length of F1 car sprite
        self.width = 30
        self.max_velocity = 10
        self.acceleration = 0.2
        self.rotation_speed = 5  # degrees per update

        # Load your F1 car image with transparent background
        self.image = pygame.image.load('f1_car.png')
        self.image = pygame.transform.scale(self.image, (self.length, self.width))
        self.rect = self.image.get_rect(center=(self.x, self.y))

        # Sensor angles relative to car (front, left front, right front)
        self.sensor_angles = [0, 30, -30]
        self.sensors = [0.0 for _ in self.sensor_angles]

    def update(self, action: int, track_mask: pygame.Surface):
        """
        action: 0 = accelerate straight
                1 = rotate right
                2 = rotate left
        """
        if action == 0:
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
        elif action == 1:
            self.angle += self.rotation_speed
        elif action == 2:
            self.angle -= self.rotation_speed

        # Move the car
        rad = math.radians(self.angle)
        self.x += self.velocity * math.cos(rad)
        self.y += self.velocity * math.sin(rad)

        # Update rect for collision/drawing
        self.rect.center = (self.x, self.y)

        # Update sensor readings
        self.update_sensors(track_mask)

    def update_sensors(self, track_mask: pygame.Surface):
        """Cast rays at sensor angles and measure distance to track edges."""
        max_distance = 150
        self.sensors.clear()

        for sensor_angle in self.sensor_angles:
            angle = math.radians(self.angle + sensor_angle)
            for dist in range(max_distance):
                check_x = int(self.x + dist * math.cos(angle))
                check_y = int(self.y + dist * math.sin(angle))

                # Check if out of bounds or off track (black pixel means off track)
                if check_x < 0 or check_y < 0 or check_x >= track_mask.get_width() or check_y >= track_mask.get_height():
                    self.sensors.append(dist / max_distance)
                    break

                # Get pixel color at sensor point (track_mask is white on track, black off track)
                color = track_mask.get_at((check_x, check_y))
                if color == pygame.Color(0, 0, 0):  # off track detected
                    self.sensors.append(dist / max_distance)
                    break
            else:
                self.sensors.append(1.0)  # max distance clear

    def draw(self, screen: pygame.Surface):
        """Draw the rotated car and sensors."""
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        new_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, new_rect.topleft)

        # Draw sensors for debug
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            sensor_length = self.sensors[i] * 150
            end_x = self.x + sensor_length * math.cos(angle)
            end_y = self.y + sensor_length * math.sin(angle)
            pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (end_x, end_y), 2)
