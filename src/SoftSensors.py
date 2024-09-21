from RoverState import RoverState
import numpy as np
import math


class SoftCompass:
    def __init__(self, s, rover:RoverState):
        self.s = s
        self.rover = rover
    
    def read(self):
        ang = self.rover.x[2] + np.random.normal(0, self.s)
        ang %= math.pi * 2
        if(ang > math.pi):
            ang -= math.pi * 2
        return ang

class SoftGPS:
    def __init__(self, s, rover:RoverState):
        self.s = s
        self.rover = rover
    
    def read(self):
        return np.array([self.rover.x[0], self.rover.x[1]]) + np.random.normal(0, self.s, size=2)


class SoftAccelerometer:
    def __init__(self, s, rover:RoverState):
        self.s = s
        self.rover = rover
    
    def read(self):
        a_c = self.rover.x[5] * self.rover.x[3]
        return np.array([self.rover.x[4], a_c]) + np.random.normal(0, self.s, size=2)

class SoftGyro:
    def __init__(self, s, rover:RoverState):
        self.s = s
        self.rover = rover
    
    def read(self):
        return self.rover.x[3] * self.rover.x[5] + np.random.normal(0, self.s)

class SoftSpeedSensor:
    def __init__(self, s, rover:RoverState):
        self.s = s
        self.rover = rover
    
    def read(self):
        return self.rover.velocity + np.random.normal(0, self.s)
