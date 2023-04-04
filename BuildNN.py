import NeuralNetwork
import logging
import numpy as np
from Neuron import Neuron
from Layer import Layer
from NeuralNetwork import NeuralNetwork
from ActivationFunction import ActivationFunction as AF

modulename = __name__
logger = logging.getLogger(modulename)

'''
    layer_sizes = [2, 3, 1]#first number is the input layer second number is the output or first hidden layer
    layer_weights = [[],[[0.1,-.1],[-.1,0.1],[0.1,0.1]],[[-.1,0.05,-.1]]] #list of weight-lists for each layer, of size of layer before, except for input layer
    activation_functions = [AF('identity'), AF('sigmoid'), AF('sigmoid')]
'''
def build(layer_sizes=[2,3,1],layer_weights=[[],[[0.1,-.1],[-.1,0.1],[0.1,0.1]],[[-.1,0.05,-.1]]],activation_functions=[AF('identity'), AF('sigmoid'), AF('sigmoid')]):#,learning_rate=0.1,epochs=10000):

    np.random.seed(42)

    nn = NeuralNetwork(layer_sizes, activation_functions,layer_weights)


    return nn

