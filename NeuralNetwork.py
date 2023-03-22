import numpy as np

class NeuralNetwork:
    def __init__(self, layers, neurons_per_layer, initial_weights=None, activation_function='sigmoid'):
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.weights = initial_weights if initial_weights is not None else self._init_weights()
        self.activation_function = self._get_activation_function(activation_function)
        self.activation_function_deriv = self._get_activation_function_deriv(activation_function)

    def _init_weights(self):
        weights = []
        for i in range(1, self.layers): #no of layers
            w = np.random.randn(self.neurons_per_layer[i], self.neurons_per_layer[i-1])
            weights.append(w)
        return weights

    def _get_activation_function(self, name):
        if name == 'sigmoid':
            return lambda x: expit(x)
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError('Unsupported activation function')

    def _get_activation_function_deriv(self, name):
        if name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif name == 'relu':
            return lambda x: np.where(x > 0, 1, 0)
        elif name == 'tanh':
            return lambda x: 1 - np.power(x, 2)
        else:
            raise ValueError('Unsupported activation function')

    def forward(self, input):
        self.a = [input]
        self.z = []
        for i, w in enumerate(self.weights):
            z = np.dot(w, self.a[-1])
            self.z.append(z)
            a = self.activation_function(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, input, target, learning_rate):
        self.forward(input)
        target = np.array(target)
        error = target - self.a[-1]
        delta = error * self.activation_function_deriv(self.a[-1])
        for i in reversed(range(len(self.weights))):
            a = self.a[i]
            z = self.z[i]
            w = self.weights[i]
            delta = np.dot(w.T, delta) * self.activation_function_deriv(a)
            w += learning_rate * np.outer(delta, self.a[i-1])
            self.weights[i] = w