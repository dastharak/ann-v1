import numpy as np
from Neuron import Neuron
from Layer import Layer
from NeuralNetwork import NeuralNetwork


def test():
    # create a network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    layer_sizes = [3, 2, 1]#first number is the input layer second number is the output or first hidden layer
    activation_functions = [lambda x: (x), lambda x: 1/(1+np.exp(-x)), lambda x: 1/(1+np.exp(-x))]#, lambda x: x]
    layer_weights = [[],[[0.1,0.2,0.1],[0.1,0.2,0.1]],[[0.3,0.1]]] #list of matrices for each layer of the size of layer before, except input layer
    input = [1, 0, 1]
    target = [0.1]
    np.random.seed(42)
    nn = NeuralNetwork(layer_sizes, activation_functions,layer_weights,len(input),layer_sizes[-1])

    # set the inputs to the network
    inputs = np.array(input)
    nn.set_inputs(inputs)

    #infor of network
    nn.printNetworkSetup()

    # calculate the output of the network
    output = nn.output()


    #infor of network
    nn.printNetworkSetup()

    nn.backpropagate(target,0.1)

    # print the output
    print(output)

test()
