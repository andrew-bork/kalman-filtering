
import numpy as np
import math


def normalize_angle(s):
    s %= math.pi * 2
    if(s > math.pi):
        s -= math.pi * 2
    return s

class AdaptiveRoverEKF:
    def __init__(self, dt, P, R, V):

        self.x = np.array([
            0.0, 0.0, 0.0, 0.0
            ## x, y, head, v, at
        ])

        self.dt = dt
        self.P = P
        self.K = np.zeros((4, 3))
        
        # self.N = N
        self.R = R
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            # [0.0, 0.0, 0.0, 0.0, 0.0],
            # [0.0, 0.0, 0.0],
        ])

        self.H_gps = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ])

        self.V = V
        # self.H_acc = np.array([
        #     [1.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.0, 0.0],
        # ])
        
        # self.acceleration_ekf = AccelerationRoverEKF(dt)
        # self.velocity_ekf = VelocityRoverEKF(dt)
        # self.stationary_ekf = StationaryRoverEKF(dt)

        # self.model = self.acceleration_ekf
        self.model_i = 0

        self.models = [ AccelerationRoverEKF(dt), VelocityRoverEKF(dt), StationaryRoverEKF(dt) ]

    
    def current_model(self):
        return self.models[self.model_i]



    def predict(self, u):
        # print(self.x, u)
        # if(0.05 < abs(u[0])):
        #     self.model_i = 0
        model = self.current_model()

        Fx = model.Fx(self.x, u)
        Fu = model.Fu(self.x, u)


        self.x = model.f(self.x, u)
        # print(Fu.shape, Fu.T.shape, self.V.shape)
        # print(Fx)
        self.P = Fx @ self.P @ Fx.T + model.N + Fu @ self.V @ Fu.T

    def h(self, x):
        sensor = np.zeros(4, dtype=np.float32)

        sensor[0] = x[0]
        sensor[1] = x[1]
        sensor[2] = x[2]
        sensor[3] = x[3]

        return sensor
    
    def update(self, z):
        model = self.current_model()

        y = z - self.h(self.x)
        y[2] = normalize_angle(y[2])

        if(abs(self.x[3]) < 0.005):
            self.model_i = 2
        else:
            self.model_i = 0


        # if(abs(y[3]) < 0.001 and abs(self.x[3]) < 0.005 and abs(self.P[3, 3]) < 0.2 and abs(self.x[4]) < 0.001):
        #     self.model_i = 2
        # elif(abs(y[3]) < 0.001 and abs(self.x[4]) < 0.001):
        #     self.model_i = 1
        # else:
        #     self.model_i = 0
        # if(0.001 < abs(y[3]) or 0.1 < abs(self.x[4])):
        #     self.model_i = 0
        # elif(abs(y[3]) < 0.0005 and abs(self.x[4]) < 0.05 and abs(self.x))
        # if(abs(y[3]) < 0.001 and abs(self.x[3]) < 0.01 and abs(self.P[3, 3]) < 0.1):
        #     self.model_i = 1
        # if(0.01 < abs(y[3])):
        #     self.model_i = 0
        # print(self.h(self.x))
        self.S = self.H @ self.P @ self.H.T + self.R
        # print(self.S)
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        # print(self.K.shape)
        # print(y.shape)
        # print(self.x.shape)
        # print(self.K.shape)

        self.x = self.x + self.K @ y
        # print(self.x)
        self.P = (np.eye(4) - self.K @ self.H) @ (self.P)


class AccelerationRoverEKF:
    def __init__(self, dt):
        self.x = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0
            ## x, y, head, v, at
        ])
        self.dt = dt

        self.P = np.diag([ 10.0, 10.0, 100.0, 100.0, 100.0 ])
        self.K = np.zeros((5, 6))
        self.N = np.diag([ 0.01**2, 0.01**2, 0.01**2, 0.1**2])
        self.R = np.diag([
            5**2, 5**2, 0.005**2, 0.1**2
        ])
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            # [0.0, 0.0, 0.0],
        ])

    def Fx(self, x, u):
        V = x[3]
        # R = u[1]
        omega = u[1]
        a = u[0]

        dt = self.dt

        out = np.eye(4, dtype=np.float32)
        out[0, 2] = dt * (V + 0.5 * dt * a) * math.cos(x[2] + omega * dt)
        out[1, 2] = -dt * (V + 0.5 * dt * a) * math.sin(x[2] + omega * dt)
        out[0, 3] = dt * math.sin(x[2] + omega * dt)
        out[1, 3] = dt * math.cos(x[2] + omega * dt)
        # out[2, 3] = dt / R
        # out[0, 4] = 0.5 * dt * dt * math.sin(x[2])
        # out[1, 4] = 0.5 * dt * dt * math.cos(x[2])
        # out[2, 4] = 0.5 * dt * dt / R
        # out[3, 4] = dt 
        # out[2, 0] = dt / R
        # out[2, 1] = -dt * V / (R ** 2)

        return out


    def Fu(self, x, u):
        V = x[3]
        # R = u[1]
        a = u[0]
        omega = u[1]
        dt = self.dt

        out = np.zeros((4, 2), dtype=np.float32)
        out[0, 0] = 0.5 * dt * dt * math.sin(x[2] + omega)
        out[1, 0] = 0.5 * dt * dt * math.cos(x[2] + omega)
        out[2, 0] = 0
        out[3, 0] = dt
        out[0, 1] = (0.5 * dt * a + V) * dt * dt * math.cos(x[2] + omega)
        out[1, 1] = -(0.5 * dt * a + V) * dt * dt * math.sin(x[2] + omega)
        out[2, 1] = dt

        return out


    def f(self, x, u):
        V = x[3]
        omega = u[1]
        # R = u[2]
        dt = self.dt
        a = u[0]
        # print(V,R,dt)

        x_new = np.zeros_like(x)
        # x_new[0] = 123

        v_new = V + 0.5 * a * dt
        
        x_new[0] = x[0] + v_new * dt * math.sin(x[2] + omega * dt)
        x_new[1] = x[1] + v_new * dt * math.cos(x[2] + omega * dt)
        x_new[2] = x[2] + omega * dt
        x_new[3] = x[3] + a * dt
        # x_new[4] = a


        return x_new

    def h(self, x):
        sensor = np.zeros(5, dtype=np.float32)

        sensor[0] = x[0]
        sensor[1] = x[1]
        sensor[2] = x[2]
        sensor[3] = x[4]
        sensor[4] = x[3]

        return sensor

    def predict(self, u):
        # print(self.x, u)
        Fx = self.Fx(self.x, u)
        Fu = self.Fu(self.x, u)
        self.x = self.f(self.x, u)
        # print(Fx)
        self.P = Fx @ self.P @ Fx.T + self.N

    
    def update(self, z):
        y = z - self.h(self.x)
        y[2] = normalize_angle(y[2])
        # print(self.h(self.x))
        self.S = self.H @ self.P @ self.H.T + self.R
        # print(self.S)
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        # print(self.K.shape)
        # print(y.shape)

        self.x = self.x + self.K @ y
        # print(self.x)
        self.P = (np.eye(5) - self.K @ self.H) @ (self.P)




class StationaryRoverEKF:
    def __init__(self, dt):
        self.x = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0
            ## x, y, head, v, at
        ])
        self.dt = dt

        self.P = np.diag([ 10.0, 10.0, 100.0, 100.0, 100.0 ])
        self.K = np.zeros((5, 6))
        self.N = np.diag([ 0.01**2, 0.01**2, 0.01**2, 10**2])
        self.R = np.diag([
            5**2, 5**2, 0.005**2, 0.1**2
        ])
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            # [0.0, 0.0, 0.0],
        ])

    def Fx(self, x, u):
        V = x[3]
        # R = u[1]
        omega = u[1]
        a = u[0]

        dt = self.dt

        out = np.eye(4, dtype=np.float32)
        # out[0, 2] = dt * (V + 0.5 * dt * a) * math.cos(x[2] + omega * dt)
        # out[1, 2] = -dt * (V + 0.5 * dt * a) * math.sin(x[2] + omega * dt)
        # out[0, 3] = dt * math.sin(x[2] + omega * dt)
        # out[1, 3] = dt * math.cos(x[2] + omega * dt)
        out[3, 3] = 0
        # out[2, 3] = dt / R
        # out[0, 4] = 0.5 * dt * dt * math.sin(x[2])
        # out[1, 4] = 0.5 * dt * dt * math.cos(x[2])
        # out[2, 4] = 0.5 * dt * dt / R
        # out[3, 4] = dt 
        # out[2, 0] = dt / R
        # out[2, 1] = -dt * V / (R ** 2)

        return out


    def Fu(self, x, u):
        V = x[3]
        # R = u[1]
        a = u[0]
        omega = u[1]
        dt = self.dt

        out = np.zeros((4, 2), dtype=np.float32)
        # out[0, 0] = 0.5 * dt * dt * math.sin(x[2] + omega)
        # out[1, 0] = 0.5 * dt * dt * math.cos(x[2] + omega)
        # out[2, 0] = 0
        # out[3, 0] = dt
        # out[0, 1] = (0.5 * dt * a + V) * dt * dt * math.cos(x[2] + omega)
        # out[1, 1] = -(0.5 * dt * a + V) * dt * dt * math.sin(x[2] + omega)
        # out[2, 1] = dt

        return out


    def f(self, x, u):
        V = x[3]
        omega = u[1]
        # R = u[2]
        dt = self.dt
        a = u[0]
        # print(V,R,dt)

        x_new = np.zeros_like(x)
        # x_new[0] = 123

        v_new = V + 0.5 * a * dt
        
        x_new[0] = x[0] # + v_new * dt * math.sin(x[2] + omega * dt)
        x_new[1] = x[1] # + v_new * dt * math.cos(x[2] + omega * dt)
        x_new[2] = x[2] # + omega * dt
        x_new[3] = 0
        # x_new[4] = a


        return x_new

    def h(self, x):
        sensor = np.zeros(5, dtype=np.float32)

        sensor[0] = x[0]
        sensor[1] = x[1]
        sensor[2] = x[2]
        sensor[3] = x[4]
        sensor[4] = x[3]

        return sensor

    def predict(self, u):
        # print(self.x, u)
        Fx = self.Fx(self.x, u)
        Fu = self.Fu(self.x, u)
        self.x = self.f(self.x, u)
        # print(Fx)
        self.P = Fx @ self.P @ Fx.T + self.N

    
    def update(self, z):
        y = z - self.h(self.x)
        y[2] = normalize_angle(y[2])
        # print(self.h(self.x))
        self.S = self.H @ self.P @ self.H.T + self.R
        # print(self.S)
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        # print(self.K.shape)
        # print(y.shape)

        self.x = self.x + self.K @ y
        # print(self.x)
        self.P = (np.eye(5) - self.K @ self.H) @ (self.P)


class VelocityRoverEKF:

    def __init__(self, dt):
        self.x = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0
            ## x, y, head, v, at
        ])
        self.dt = dt

        self.P = np.diag([ 100.0, 100.0, 100.0, 0.0, 0.0 ])
        self.K = np.zeros((5, 6))
        self.N = np.diag([ 0.0001**2, 0.0001**2, 0.0001**2, 0.01**2, 2**2 ])
        self.R = np.diag([
            5**2, 5**2, 0.005**2, 0.1**2
        ])
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            # [0.0, 0.0, 0.0],
        ])

    def Fx(self, x, u):
        V = u[0]
        R = u[1]
        a = x[4]

        dt = self.dt

        out = np.eye(5, dtype=np.float32)
        # out[0, 2] = dt * (V + 0.5 * dt * a) * math.cos(x[2])
        # out[1, 2] = -dt * (V + 0.5 * dt * a) * math.sin(x[2])
        # out[0, 3] = dt * math.sin(x[2])
        # out[1, 3] = dt * math.cos(x[2])
        # out[2, 3] = dt / R
        # out[0, 4] = 0.5 * dt * dt * math.sin(x[2])
        # out[1, 4] = 0.5 * dt * dt * math.cos(x[2])
        # out[2, 4] = 0.5 * dt * dt / R
        # out[3, 4] = dt 
        out[3, 3] = 0
        out[4, 4] = 0
        # out[2, 0] = dt / R
        # out[2, 1] = -dt * V / (R ** 2)

        return out


    def Fu(self, x, u):
        V = u[0]
        R = u[1]
        dt = self.dt

        out = np.zeros((5, 2), dtype=np.float32)
        # out[0, 0] = dt * math.sin(x[2])
        # out[1, 0] = dt * math.cos(x[2])
        # out[2, 0] = dt / R
        # out[2, 1] = -dt * V / (R ** 2)
        # out[3, 0] = 1
 
        return out


    def f(self, x, u):
        V = u[0]
        R = u[1]
        dt = self.dt
        a = x[4]
        # print(V,R,dt)

        x_new = np.zeros_like(self.x)
        # x_new[0] = 123

        v_new = V + 0.5 * a * dt
        
        # x_new[0] = x[0] + v_new * dt * math.sin(x[2])
        # x_new[1] = x[1] + v_new * dt * math.cos(x[2])
        # x_new[2] = x[2] + v_new * dt / R
        # x_new[3] = V + a * dt
        # x_new[4] = (u[0] - x[3])
        x_new[0] = x[0] # + v_new * dt * math.sin(x[2])
        x_new[1] = x[1] # + v_new * dt * math.cos(x[2])
        x_new[2] = x[2] # + v_new * dt / R
        x_new[3] = x[3] + x[4] * dt
        x_new[4] = x[4]


        return x_new

    def h(self, x):
        sensor = np.zeros(4, dtype=np.float32)

        sensor[0] = x[0]
        sensor[1] = x[1]
        sensor[2] = x[2]
        sensor[3] = x[4]

        return sensor

    def predict(self, u):
        # print(self.x, u)
        Fx = self.Fx(self.x, u)
        Fu = self.Fu(self.x, u)
        self.x = self.f(self.x, u)
        # print(Fx)
        self.P = Fx @ self.P @ Fx.T + self.N

    
    def update(self, z):
        y = z - self.h(self.x)
        y[2] = normalize_angle(y[2])
        # print(self.h(self.x))
        self.S = self.H @ self.P @ self.H.T + self.R
        # print(self.S)
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        # print(self.K.shape)
        # print(y.shape)

        self.x = self.x + self.K @ y
        # print(self.x)
        self.P = (np.eye(5) - self.K @ self.H) @ (self.P)