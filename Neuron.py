import numpy as np

class Neuron:
    def __init__(self, inputs=None, weights=None, bias=0, activation_function=lambda x: x):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.net = None
        self.activation_function = activation_function
        self.outputs = None
    
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
    
    def get_inputs(self):
        return self.inputs
    
    def set_weights(self, weights):
        self.weights = weights
    
    def get_weights(self):
        return self.weights
    
    def set_bias(self, bias):
        self.bias = bias
    
    def get_net(self):
        return self.net

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function
    
    def derivative(self):
        x = self.net
        return np.exp(-x) / (1 + np.exp(-x))**2

    def output(self):
        #print(f'self.inputs, self.weights, self.bias{self.inputs, self.weights[0], self.bias}')
        #print(f"Layer id:{self.layer.get_layer_id()} Neuron id:{self.get_id()}")
        if(self.layer.get_id() == 0):#print("The input layer")
            #Input layer weights are 1x1 vector - (this is redundent)
            self.net = np.dot(self.inputs[self.id].T,self.weights[0]) + self.bias
        else:#print("Not the input layer")
            self.net = np.dot(self.inputs, self.weights) + self.bias
        self.outputs = self.activation_function(self.net)
        return self.outputs
    
    def initialize_weights(self, num_inputs):
        if num_inputs != None:
            self.weights = np.random.rand(num_inputs)#fix the number
        else:#None comes for input layer, inputs for this layer is same as the layer size
            self.weights = np.ones(1,dtype=int)
    
    def set_layer(self, layer):#which layer this neurone belongs to?
        self.layer = layer

    def set_id(self,id):
        self.id = id

    def get_id(self):
        return self.id

    def __str__(self):
        return f"({self.id}:inputs={self.inputs}, weights={self.weights}, bias={self.bias},outputs={self.outputs}, activation_function={self.activation_function.__name__})"
