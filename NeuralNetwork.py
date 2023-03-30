import numpy as np
import logging
from Layer import Layer

modulename = __name__
logger = logging.getLogger(modulename)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions,weights,input_size,output_size):
        self.validateConfiguration(layer_sizes, activation_functions,weights,input_size,output_size)
        self.layers = []
        num_layers = len(layer_sizes)
        for i in range(num_layers):
            if i == 0:#input layer,no previous layer, weights are fixed to [1]
                layer = Layer(layer_sizes[i], None, activation_functions[i],None,input_size,output_size)
            else:
                layer = Layer(layer_sizes[i], layer_sizes[i-1], activation_functions[i],weights[i],input_size,output_size)
            layer.set_id(i)
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

    def validateConfiguration(self,layer_sizes, activation_functions,weights,input_size,output_size):
        if(len(layer_sizes)!=len(activation_functions)):
            print(f'activation functions are not defined for each layer')
            raise ValueError
        if(weights!=None):
            for i in range(1,len(layer_sizes)):
                if len(weights[i]) != layer_sizes[i]:
                    print(f'weights matrix and neurons count in layer {i} do not match')
                    raise ValueError
            for i in range(1,len(layer_sizes)):#check the weights and previous layer neuron counts
                for j in range (0,len(weights[i])):
                    if len(weights[i][j]) != layer_sizes[i-1]:
                        print(f'weights vector size of layer {i} nueron {j} and neurons count in layer {i-1} do not match')
                        raise ValueError                    

    def backpropagate(self, targets, learning_rate):
        # calculate errors and deltas for output layer
        print(f'# calculate errors and deltas for output layer')
        output_layer = self.layers[-1]
        print(f'output layer id:{output_layer.get_id()}')
        print(f'targets:{targets}-output_layer.output():{output_layer.output()}')
        output_error = targets - output_layer.output()#d-f(net)
        print(f'output_error:{output_error}')
        print(f'output_layer.get_neurons():{len(output_layer.get_neurons())}')
        output_delta = output_error * (output_layer.get_neurons()[0]).derivative()
        print(f'output_delta:{output_delta}')

        # calculate errors and deltas for all hidden layers in reverse order
        hidden_layers = reversed(self.layers[1:-1])
        for layer in hidden_layers:
            layer_neurons = layer.get_neurons()#list of neurones
            layer_weights = [n.get_weights() for n in layer_neurons]
            layer_deltas = []

            # calculate delta for each neuron in this layer
            for i, neuron in enumerate(layer_neurons):
                print(f'i:{i},output_delta:{output_delta},layer_weights[i]:{layer_weights[i]}')
                #error = np.dot(output_delta, layer_weights[i])
                #delta = error * neuron.derivative()
                #layer_deltas.append(delta)

            continue
            # update weights for each neuron in this layer
            for i, neuron in enumerate(layer_neurons):
                neuron_inputs = neuron.get_inputs()
                neuron_weights = neuron.get_weights()
                new_weights = neuron_weights + (learning_rate * neuron_inputs * layer_deltas[i])
                neuron.set_weights(new_weights)

            output_delta = np.array(layer_deltas)

        # update weights for neurons in the input layer
        print(f'# We dont update weights for neurons in the input layer')
        print(f'# Input layer in this network does only a pass through!')
        if(False):
            input_layer = self.layers[0]#get input layer
            input_neurons = input_layer.get_neurons()
            input_weights = [n.get_weights() for n in input_neurons]
            input_outputs = input_layer.output()
            input_deltas = []

        # calculate delta for each neuron in the input layer
        #print(f'#calculate delta for each neuron in the input layer')
        if(False):
            for i, neuron in enumerate(input_neurons):
                print(f'neuron:{i},output_delta:{output_delta},input_weights[i]:{input_weights[i]}')
                #error = np.dot(output_delta, input_weights[i])
                #delta = error * neuron.derivative()
                #input_deltas.append(delta)
                continue

            # update weights for each neuron in the input layer
            for i, neuron in enumerate(input_neurons):
                neuron_inputs = neuron.get_inputs()
                neuron_weights = neuron.get_weights()
                new_weights = neuron_weights + (learning_rate * neuron_inputs * input_deltas[i])
                neuron.set_weights(new_weights)


        #def backward2(self, input, y_true, learning_rate):
            #https://towardsdatascience.com/how-to-create-a-simple-neural-network-model-in-python-70697967738f
            #print(f'<backward()')

    
