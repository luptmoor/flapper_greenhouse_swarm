from matplotlib import pyplot as plt
import numpy as np
import random

# Counter function
def counter(xlist):
    return [item**2 / tmax**2 for item in xlist]

# Simulation settings
tmax = 600
dt = 0.01
t = np.arange(0, tmax, dt)


while True:
    x = []

    # Random simulation
    val = 0
    for i in range(int(tmax//dt+1)):
        x.append(val)
        val += dt

        if random.uniform(0, 1) < 0.00005:
            val = 0


    y = counter(x)

    avg = np.mean(y)
    avglist = avg * np.ones(len(t))
  
    # Fitness
    print(1-avg/ 0.3333250000459273)

    plt.plot(t, y)
    plt.plot(t, avglist)
    plt.xlim([-5, 605])
    plt.ylim(-0.01, 1.01)
    plt.grid()
    plt.show()

