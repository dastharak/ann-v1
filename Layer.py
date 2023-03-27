import numpy as np
from Neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function=lambda x: x,weights=None,input_size=1,output_size=1):
        self.num_neurons = num_neurons
        self.neurons = []
        #self.layer_id = None
        for i in range(num_neurons):
            neuron = Neuron()
            neuron.set_layer(self)
            neuron.set_id(i)
            if (weights!=None):
                neuron.initialize_weights(num_inputs)#TODO set the user given weights to the network
            else:    
                neuron.initialize_weights(num_inputs)
            neuron.set_activation_function(activation_function)
            self.neurons.append(neuron)

    def set_inputs(self, inputs):
        for neuron in self.neurons:
            neuron.set_inputs(inputs)

    def get_num_neurons(self):
        return self.num_neurons

    def get_neurons(self):
        return self.neurons

    def output(self):
        return np.array([neuron.output() for neuron in self.neurons])

    def get_layer_id(self):
        return self.layer_id

    def set_layer_id(self,id):
        self.layer_id = id
