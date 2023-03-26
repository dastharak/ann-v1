import numpy as np

class Neuron:
    def __init__(self, inputs=None, weights=None, bias=0, activation_function=lambda x: x):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._weights = None
        else:
            self._weights = np.array(weights)
    
    def set_inputs(self, inputs):
        self.inputs = inputs
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_bias(self, bias):
        self.bias = bias
    
    def set_activation_function(self, activation_function):
        self.activation_function = activation_function
    
    def output(self):
        #print(f'self.inputs, self.weights, self.bias{self.inputs, self.weights, self.bias}')
        if(self.layer.isInputLayer()):
            print("inputlayer")
            weighted_sum = self.inputs[self.id]
        else:
            print("Notinputlayer")
            weighted_sum = np.dot(self.inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)
    
    def initialize_weights(self, num_inputs):
        if num_inputs != None:
            self.weights = np.random.rand(num_inputs)#fix the number
        else:#None comes for input layer, inputs for this layer is same as the layer size
            input_width = self.layer.get_num_neurons()
            #self.weights = np.ones(input_width,dtype=int)
    
    def set_layer(self, layer):#which layer this neurone belongs to?
        self.layer = layer

    def set_id(self,id):
        self.id = id

    def __str__(self):
        return f"(inputs={self.inputs}, weights={self.weights}, bias={self.bias}, activation_function={self.activation_function.__name__})"
