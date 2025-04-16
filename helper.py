
import numpy as np

def fn(heading):
    heading = heading / 57.3
    return ((heading + np.pi) % (2*np.pi) - np.pi) * 57.3


while True:
    heading = float(input('Enter heading [deg]:'))
    print(f"Clipped heading: {fn(heading)}")