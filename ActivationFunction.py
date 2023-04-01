import numpy as np

class ActivationFunction:
    def __init__(self, function_name):
        self.function_name = function_name

    def get_function(self):
        if self.function_name == 'sigmoid':
            return self.sigmoid
        elif self.function_name == 'relu':
            return self.relu
        elif self.function_name == 'tanh':
            return self.tanh
        elif self.function_name == 'identity':
            return self.identity
        else:
            raise ValueError("Unknown activation function")

    def get_derivative(self):
        if self.function_name == 'sigmoid':
            return self.sigmoid_derivative
        elif self.function_name == 'relu':
            return self.relu_derivative
        elif self.function_name == 'tanh':
            return self.tanh_derivative
        elif self.function_name == 'identity':
            return self.identity_derivative
        else:
            raise ValueError("Unknown activation function derivative")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #return x * (1 - x)
        #return (np.exp(-x) / (1 + np.exp(-x))**2)
        X = 1 / (1 + np.exp(-x))
        return X*(1-X)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.square(x)

    def identity(self, x):
        return x

    def identity_derivative(self, x):
        return 1




