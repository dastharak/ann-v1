import numpy as np
from Neuron import Neuron
from Layer import Layer
from NeuralNetwork import NeuralNetwork


def test():
    # create a network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    layer_sizes = [3, 2, 2]#first number is the input layer second number is the output or first hidden layer
    activation_functions = [lambda x: (x), lambda x: 1/(1+np.exp(-x)), lambda x: 1/(1+np.exp(-x))]#, lambda x: x]
    #ctivation_function_derivatives = [lambda x: (1), lambda x: np.exp(-x)/(1+np.exp(-x))**2, lambda x: np.exp(-x)/(1+np.exp(-x))**2]#, lambda x: x]
    layer_weights = [[],[[0.1,0.1,-.1],[-.1,0.1,0.1]],[[-.1,0.1],[0.1,-.1]]] #list of weight-lists for each layer, of size of layer before, except for input layer
    input = [1, 0, 0]
    target = [1,0]
    np.random.seed(42)
    nn = NeuralNetwork(layer_sizes, activation_functions,layer_weights,len(input),layer_sizes[-1])

    # set the inputs to the network
    inputs = np.array(input)
    nn.set_inputs(inputs)

    #infor of network
    strBefore = nn.getNetworkSetup()
    out1 = nn.output()
    for i in range(1,2000):

        # calculate the output of the network
        output = nn.output()


        print(f'net:{output}')

        nn.backpropagate(target,0.1)

    strAfter = nn.getNetworkSetup()
    # print the output
    out2 = nn.output()

    print(f'NetBefore:{strBefore}')
    print(f'     Out1:{out1}')
    print(f' NetAfter:{strAfter}')
    print(f'     Out2:{out2}')
    print(f'   Target:{target}')


test()
