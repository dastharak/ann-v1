import numpy as np
import logging
from Layer import Layer

modulename = __name__
logger = logging.getLogger(modulename)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions,weights,input_size,output_size):
        self.layers = []
        num_layers = len(layer_sizes)
        for i in range(num_layers):
            if i == 0:
                layer = Layer(layer_sizes[i], None, activation_functions[i],weights[i],input_size,output_size)
            else:
                layer = Layer(layer_sizes[i], layer_sizes[i-1], activation_functions[i],weights[i],input_size,output_size)
            layer.set_layer_id(i)
            self.layers.append(layer)

    def set_inputs(self, inputs):
        self.layers[0].set_inputs(inputs)

    def output(self):
        output = self.layers[0].output()#the input layer
        for i in range(1, len(self.layers)):
            self.layers[i].set_inputs(output)
            output = self.layers[i].output()
        return output

    def printNetworkSetup(self):
        print(f'No of Layers:{len(self.layers)}')
        for i in range(0,len(self.layers)):
            li = self.layers[i]
            n = li.get_num_neurons()
            print(f'Layer {i}|Neurons:{n}')
            nsli = li.get_neurons()
            for nur in nsli:
                print(f'         |Neuron:{str(nur)}')
        print(f'---------------------------')

    '''
        def _get_activation_function(self, name):
            if name == 'sigmoid':
                return lambda x: 1 / (1 + np.exp(-x))
            elif name == 'relu':
                return lambda x: np.maximum(0, x)
            elif name == 'tanh':
                return lambda x: np.tanh(x)
            else:
                raise ValueError('Unsupported activation function')

        def _get_activation_function_deriv(self, name):
            if name == 'sigmoid':
                return lambda x: x * (1 - x)
            elif name == 'relu':
                return lambda x: np.where(x > 0, 1, 0)
            elif name == 'tanh':
                return lambda x: 1 - np.power(x, 2)
            else:
                raise ValueError('Unsupported activation function')

    class NeuralNetwork:
        def __init__(self, layers, neurons_per_layer, initial_weights=None, activation_function='sigmoid'):
            self.layers = layers
            self.neurons_per_layer = neurons_per_layer
            #A list of weight matrices, where self.weights[i] corresponds to the weights connecting layer i to layer i+1.
            self.weights = initial_weights if initial_weights is not None else self._init_weights()
            self.neurons = self._init_neurons(activation_function)
            #Following 3 are defined for commenting only (not necessary to be here)
            #A list that contains the activations of each layer in the network, starting with the input layer. self.a[0] corresponds to the input layer activations, self.a[1] corresponds to the first hidden layer activations, self.a[2] corresponds to the second hidden layer activations, and so on. During the forward pass, the activations are computed sequentially and added to this list.
            self.activation = None 
            #A list that contains the linear combinations of inputs and weights, also known as the "logits", for each layer in the network, starting with the first hidden layer. self.net[0] corresponds to the logits for the first hidden layer, self.net[1]
            self.net = None

        def _init_weights(self):
            weights = []
            for i in range(0, self.layers-1): #no of layers
                #i = neurones in the current layer, i+1 neurones in the next layer
                w = np.random.randn(self.neurons_per_layer[i], self.neurons_per_layer[i+1])
                weights.append(w)
                print(f'{i}->{i+1}:weight:{w.shape}')
            print(f'wights list:{len(weights)}')
            return weights

        def _init_neurons(self, activation_function):
            neurons = []
            for i in range(self.neurons_per_layer[0]):
                neuron = Neuron(activation_function)
                neurons.append(neuron)
            return neurons

        def forward(self, input):
            print(f'>forward()')
            input = np.array(input)#X[i]
            input = input.T if len(input) != self.neurons_per_layer[0] else input
            self.activation = []
            self.activation.append(input)
            #print(f'self.activation start:{len(self.activation)}{self.activation}')
            self.net = []
            for i, w in enumerate(self.weights):#Loop through each weight matrix and its corresponding index.
                print(f' weight matrix:{i}->{i+1},w:{


        def backward2(self, input, y_true, learning_rate):
            #https://towardsdatascience.com/how-to-create-a-simple-neural-network-model-in-python-70697967738f
            print(f'<backward()')

    '''