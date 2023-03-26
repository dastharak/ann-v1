import NeuralNetwork as nn
import logging

modulename = __name__
logger = logging.getLogger(modulename)

def build(num_of_layers,inputs_array,output_array):

    logger.debug(f'num_of_layers{num_of_layers}:inputs_array{inputs_array.shape}')
    ann = nn.NeuralNetwork(num_of_layers, inputs_array.shape[0], initial_weights=None, activation_function='sigmoid')
