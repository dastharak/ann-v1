import numpy as np
import logging
from Layer import Layer
from Neuron import Neuron

modulename = __name__
logger = logging.getLogger(modulename)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions,weights,input_size,output_size):
        self.validateConfiguration(layer_sizes, activation_functions,weights,input_size,output_size)
        self.layers = []
        self.error = None
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

    def set_error(self,err):
        self.error = err

    def get_error(self):
        return self.error

    def getNetworkSetup(self):
        strConfig = "\n"
        strConfig += (f'-----------------------------------------------------------')
        strConfig += '\n'

        for i in range(0,len(self.layers)):
            li = self.layers[i]
            n = li.get_num_neurons()
            strConfig += (f'Layer {i}  |Neurons:{n}')
            strConfig += '\n'
            nsli = li.get_neurons()
            for nur in nsli:
                strConfig += (f'         |Neuron:{str(nur)}')
                strConfig += '\n'

        # Determine the maximum length of each column
        #col_width = [max(len(str(x)) for x in col) for col in zip(*data)]

        # Output the data with auto-adjusted spacing
        #for row in data:
            #print('  '.join('{:<{}}'.format(str(row[i]), col_width[i]) for i in range(len(row))))

        strConfig += (f'-----------------------------------------------------------')
        strConfig += '\n'
        return strConfig

    def validateConfiguration(self,layer_sizes, activation_functions,weights,input_size,output_size):
        if(len(layer_sizes)!=len(activation_functions)):
            print(f'activation functions are not matching for number of layers')
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
        #print(f'# calculate errors and deltas for output layer')
        output_layer = self.layers[-1]
        #print(f'output layer id:{output_layer.get_id()}')
        #print(f'targets:{targets}-output_layer.output():{output_layer.output()}')
        output_error = targets - output_layer.output()#d-f(net)
        self.error = output_error
        #print(f'output_error:{output_error}')
        #print(f'output_layer.get_neurons():{len(output_layer.get_neurons())}')
        output_delta = []
        for i,output_layer_neuron in enumerate(output_layer.get_neurons()):#for each neuron in output layer
            #delta = output_error[i] * output_layer_neuron.activation_function.get_derivative(output_layer_neuron.net)#d-f(net)*f'(net)
            act_fun_diff = output_layer_neuron.activation_function.get_derivative()
            delta = output_error[i] * act_fun_diff(output_layer_neuron.net)#d-f(net)*f'(net)
            #print(f'output delta:{delta},{type(delta)},{delta.shape}')
            output_layer_neuron.set_deltaw(delta)
            output_delta.append(delta)
        
        #print(f'output_delta:{output_delta}')
        output_layer.set_deltas(output_delta)

        # calculate errors and deltas for all hidden layers in reverse order
        hidden_layers = reversed(self.layers[1:-1])#go backwards in {hidden layers}: Ln-1,{Ln-2,...L2,L1},L0
        for layer in hidden_layers:
            hlid = layer.get_id()
            #print(f'hidden layer id:{hlid}')
            next_layer = self.layers[hlid+1] #next layer
            #print(f'next layer id:{next_layer.get_id()}')
            layer_neurons = layer.get_neurons()#list of neurones in this layer
            #[print(f'layer_neurons:{n}') for n in layer_neurons]
            #calculate delta for each neuron
            for j,hidden_layer_neuron in enumerate(layer_neurons):
                hidden_layer_neuron_delta_sum = 0
                for next_layer_neuron in next_layer.get_neurons():
                    next_layer_neuron_weights = next_layer_neuron.get_weights()
                    #print(f'next_layer_neuron_weights:{next_layer_neuron_weights,type(next_layer_neuron_weights)}')
                    deltaNurone = next_layer_neuron.get_deltaw()
                    #print(f'deltaNurone:{deltaNurone,type(deltaNurone)}')
                    
                    #print(f'delta{type(delta),deltaNurone}')
                    #print(f'{next_layer_neuron_weights[j]}*{deltaNurone}')
                    sum = next_layer_neuron_weights[j]*deltaNurone
                    #print(f'sum1:{sum}')
                    #sum = hidden_layer_neuron.activation_function.get_derivative(hidden_layer_neuron.net)*sum #f'(net)*[Sum(w.delta)]
                    act_fun_diff = hidden_layer_neuron.activation_function.get_derivative()
                    sum = act_fun_diff(hidden_layer_neuron.net)*sum #f'(net)*[Sum(w.delta)]
                    #print(f'delta{hlid}{j}:{sum}')
                    hidden_layer_neuron_delta_sum+=sum

                hidden_layer_neuron.set_deltaw(hidden_layer_neuron_delta_sum)



        # calculate new weights for all layers in reverse order
        updatable_layers = reversed(self.layers[1:])#go backwards in {hidden layers}: {Ln-1,Ln-2,...L2,L1},L0
        for layer in updatable_layers:
            layer_id = layer.get_id()
            #print(f'layer_id:{layer_id}')#3,2,1
            # update weights for each neuron in this layer
            layer_neurons = layer.get_neurons()#list of neurones in this layer
            for i, neuron in enumerate(layer_neurons):
                neuron_inputs = neuron.get_inputs()
                neuron_weights = neuron.get_weights()
                new_weights = neuron_weights + (learning_rate * neuron_inputs * neuron.get_deltaw())
                neuron.set_weights(new_weights)

        # update weights for neurons in the input layer
        #print(f'# We dont update weights for neurons in the input layer')
        #print(f'# Input layer in this network does only a pass through!')
        


        #def backward2(self, input, y_true, learning_rate):
            #https://towardsdatascience.com/how-to-create-a-simple-neural-network-model-in-python-70697967738f
            #print(f'<backward()')

    

    '''
    bacth processing

    In batch training, the neural network processes a mini-batch of input examples simultaneously and calculates the forward pass and backward pass for all examples in the mini-batch.

    In your example, you have a 1000x3 matrix of inputs and a 1000x1 vector of outputs. You can split the input matrix and output vector into mini-batches, each with a certain number of input-output pairs. For example, you can use a mini-batch size of 32, which means each mini-batch will contain 32 input-output pairs.

    When training the neural network with a mini-batch size of 32, the first step is to randomly select 32 input-output pairs from the dataset. These input-output pairs are then fed into the neural network simultaneously. The neural network calculates the forward pass for all 32 input examples and computes the loss by comparing the network's output to the true output for all 32 examples.

    Once the loss is computed, the backpropagation algorithm is used to calculate the gradients of the loss with respect to the network's weights. The gradients are then used to update the weights of the network using an optimization algorithm such as stochastic gradient descent.

    After the weights are updated, a new mini-batch of 32 input-output pairs is randomly selected from the dataset, and the process repeats. This continues until all examples in the dataset have been used to update the weights of the network multiple times.

    In summary, during batch training, the network processes a mini-batch of input-output pairs simultaneously, computes the forward pass and backpropagation for all examples in the mini-batch, calculates the gradients of the loss with respect to the network's weights, and updates the weights of the network. The process repeats for multiple epochs until the network converges to a good solution.
    '''
