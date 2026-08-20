import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    x = np.linspace(0, 20, 1000)
    w_low = 0.1
    w_high =10.0
    tau = 2.0
    y = w_low+(w_high-w_low)/(1+(np.exp(-(x-1)/tau)))
    plt.plot(x, y)
    plt.show()
