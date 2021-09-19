import math
import numpy as np
import matplotlib.pyplot as plt


class GBM:

    def __init__(self, asset_price, drift, vol, dt, T):
        self.current_price = asset_price
        self.drift = drift
        self.vol = vol
        self.dt = dt
        self.T = T
        self.prices = []

    def time_step(self):
        dW = np.random.normal(0, math.sqrt(self.dt))
        dS = self.drift*self.dt*self.current_price + self.vol*self.current_price*dW
        self.current_price += dS
        self.prices.append(self.current_price)

    def simulate_path(self):
        while(self.T - self.dt > 0):
            self.time_step()
            self.T -= self.dt

    
    
    
asset_price = 100
drift = 0.15
vol = 0.2
dt = 1/365
T = 1
num_sims = 100
paths = []

for i in range(num_sims):
    gbm = GBM(asset_price, drift, vol, dt, T)
    gbm.simulate_path()
    paths.append(gbm.prices)


for path in paths:
    plt.plot(path)
plt.show()
