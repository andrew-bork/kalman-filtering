import numpy as np
import math

class RoverState:
    def __init__(self):

        self.x = np.array([
            0.0, 0.0, # Position
            0.0, # Heading
            0.0, # Velocity
            # 0.0, # Heading Rate
            0.0, # 0.0 # Acceleration
            0, # 1 / Turn Radius
        ])

        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.heading_rate = 0.0
        self.velocity = 0.0
        self.acceleration = np.array([0.0, 0.0])
        
        self.control = np.array([0.0, 0.0])

        
    
    def set_control(self, radius, velocity):
        self.control[0] = radius
        self.control[1] = velocity

    def update(self, dt):
        alpha = (self.x[3] + 0.5 * dt * self.x[4]) * dt * self.x[5]
        # print("b", alpha)

        self.new_x = np.zeros_like(self.x)
        self.new_x[0] = self.x[0] + (self.x[3] + 0.5 * dt * self.x[4]) * math.sin(self.x[2]) * dt
        self.new_x[1] = self.x[1] + (self.x[3] + 0.5 * dt * self.x[4]) * math.cos(self.x[2]) * dt

        self.new_x[2] = normalize_angle(self.x[2] + alpha)
        self.new_x[3] = self.x[3] + self.x[4] * dt
        # self.new_x[4] = max(min((self.control[1] - self.x[3]) * 100, 0.05), -0.05)
        self.new_x[4] = self.control[1] - self.x[3]

        self.new_x[5] = self.x[5] + max(min(1 / self.control[0] - self.x[5], 1), -1) * dt

        self.x = self.new_x

        # self.position += self.control[1] * np.array([math.sin(self.heading + alpha), math.cos(self.heading + alpha)]) * dt
        # self.new_x[0, 0] = self.x[0,0] + self.control[1] * math.sin(self.x[2,0] + alpha) * dt
        # self.new_x[1, 0] = self.x[1,0] + self.control[1] * math.cos(self.x[2,0] + alpha) * dt
        # self.new_x[2, 0] = self.x[2,0] + alpha
        # self.position += (self.velocity * dt + 0.5 * self.acceleration[0] * dt * dt) * np.array([math.sin(self.heading), math.cos(self.heading)])
        # self.acceleration[0] = (self.control[1] - self.velocity) / dt
        # self.velocity = self.control[1]
        # self.acceleration[1] = self.control[1] / (self.control[0] ** 2)
        # self.heading_rate = self.velocity * self.control[0]
        # self.heading += alpha
        # self.heading = normalize_angle(self.heading)


def normalize_angle(s):
    s %= math.pi * 2
    if(s > math.pi):
        s -= math.pi * 2
    return s