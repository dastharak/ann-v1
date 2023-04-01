import ActivationFunction as af
import numpy as np


# Example usage
af = af.ActivationFunction('sigmoid')
activation_fn = af.get_function()
derivative_fn = af.get_derivative()
x = np.array([1, 2, 3, 4])
y = activation_fn(x)
dy = derivative_fn(y)

print(f'x:{x},y:{y},dy:{dy}')
