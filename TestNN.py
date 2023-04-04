import numpy as np
from Neuron import Neuron
from Layer import Layer
from NeuralNetwork import NeuralNetwork
from ActivationFunction import ActivationFunction
import time


def test():
    # create a network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    layer_sizes = [2, 3, 1]#first number is the input layer second number is the output or first hidden layer
    #activation_functions = [lambda x: (x), ActivationFunction('sigmoid'), lambda x: 1/(1+np.exp(-x))]#, lambda x: x]
    activation_functions = [ActivationFunction('identity'), ActivationFunction('sigmoid'), ActivationFunction('sigmoid')]#, lambda x: x]
    #ctivation_function_derivatives = [lambda x: (1), lambda x: np.exp(-x)/(1+np.exp(-x))**2, lambda x: np.exp(-x)/(1+np.exp(-x))**2]#, lambda x: x]
    layer_weights = [[],[[0.1,-.1],[-.1,0.1],[0.1,0.1]],[[-.1,0.05,-.1]]] #list of weight-lists for each layer, of size of layer before, except for input layer
    
    learning_rate = 0.2
    epochs = 150000
    #XOR
    input = np.array([[1, 0],[1, 1],[0, 1],[0, 0]])
    target = np.array([1,     0,     1,     0])
    
    #OR
    #input = np.array([[1, 0],[1, 1],[0, 1],[0, 0]])
    #target = np.array([1,     1,     1,     0])

    #AND
    #input = np.array([[1, 0],[1, 1],[0, 1],[0, 0]])
    #target = np.array([0,     1,     0,     0])

    np.random.seed(42)
    nn = NeuralNetwork(layer_sizes, activation_functions,layer_weights)#,len(input),layer_sizes[-1])

    # set the inputs to the network
    #inputs = np.array(input)
    nn.set_inputs(input[0])

    output = np.array([0,     0,     0,     0],dtype=float)
    #infor of network
    strBefore = nn.getNetworkSetup()
    # record start time
    start_time = time.time()

    # calculate the output of the network
    out1 = nn.output()
    nn.backpropagate(target[0],learning_rate)
    for i in range(1,epochs):  #):# 
        k = i%4
        nn.set_inputs(input[k])
        # calculate the output of the network
        #output[k] = nn.output()
        #print(f'output[k]{type(output[k])},nn.output().item(){(type(nn.output()[0]))}')
        output[k] = nn.output()[0] # Convert ndarray to float

        #output[k] = nn.output()

        nn.backpropagate(target[k],learning_rate)

        err = nn.get_error()
        if(i%(epochs/10) == 0 ):
            print(f'net:{output[k]},error:{err},target:{target[k]}')
    
    # record end time
    end_time = time.time()

    # calculate elapsed time
    elapsed_time = end_time - start_time

    strAfter = nn.getNetworkSetup()
    # print the output
    out2 = nn.output()
    print(f"Elapsed time:{elapsed_time} seconds")
    print(f'  Net Before:{strBefore}')
    print(f'        Out1:{out1}')
    print(f'   Net After:{strAfter}')
    print(f'        Out2:{output}')
    print(f'   Target(s):{target}')


test()
