import numpy as np
from Neuron import Neuron
from Layer import Layer
from NeuralNetwork import NeuralNetwork


def test():
    # create a network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    layer_sizes = [3, 2]
    activation_functions = [lambda x: (x), lambda x: 1/(1+np.exp(-x))]#, lambda x: x]
    weights = [1,2,3,4,5]
    input = [1, 0, 1]
    nn = NeuralNetwork(layer_sizes, activation_functions,weights,len(input),2)

    # set the inputs to the network
    inputs = np.array(input)
    nn.set_inputs(inputs)

    # calculate the output of the network
    output = nn.output()

    #architecture of network
    nn.printNetworkSetup()

    # print the output
    print(output)

test()