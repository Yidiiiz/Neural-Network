import numpy as np
import matplotlib as mpl

inputs = [1, 2, 3]
weights = [.2, .8, -.5]
bias = 2

output = (inputs[0]*weights[0]
        + inputs[1]*weights[1]
        + inputs[2]*weights[2] + bias)

print(output)