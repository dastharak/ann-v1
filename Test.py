import numpy as np
from NeuralNetwork import NeuralNetwork

# input dataset
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]]) #8x3

# output dataset
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]]) #8x1

class XORNeuralNetwork:
    def __init__(self):
        self.nn = NeuralNetwork(layers=3, neurons_per_layer=[3, 4, 1], activation_function='sigmoid')

    def train(self, X, y, learning_rate=0.1, epochs=10):
        for epoch in range(epochs):
            print(f'epoch:{epoch}')
            for i in range(len(X)):# train each input row
                print(f'Training input {i}: X[i]:{X[i]},y[i]:{y[i]}')
                self.nn.backward2(X[i], y[i], learning_rate)
                print('--------------')

    def predict(self, X):
        return self.nn.forward(X)
#Test code
xor_nn = XORNeuralNetwork()
xor_nn.train(X, y)

for i in range(len(X)):
    print(f'Input: {X[i]}, Output: {xor_nn.predict(X[i])[-1]}')#-1 is the last layer

'''
Input: [0 0 0], Output: [0.0408291]
Input: [0 0 1], Output: [0.96523052]
Input: [0 1 0], Output: [0.96222515]
Input: [0 1 1], Output: [0.04501081]
Input: [1 0 0], Output: [0.96304054]
Input: [1 0 1], Output: [0.04477256]
Input: [1 1 0], Output: [0.04521181]
Input: [1 1 1], Output: [0.9605212]
'''