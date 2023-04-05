
import Setup as setup
import IO as io
import BuildNN as build
import TrainNN as train
import NeuralNetwork as nn

def run_script():

    logger,args = setup.setup()

    inoutputs_array = args.inoutputdata
    num_of_layers = args.numoflayers
    activation_functions = args.activationfunctions
    weights_4_each_layer = args.weights4eachlayer
    learning_rate = args.learningrate
    epochs = args.epochs

    neuralnet = build.build(num_of_layers,weights_4_each_layer,activation_functions)
    if(learning_rate is None and epochs is None):
        nn = train.train(neuralnet,inoutputs_array)
    elif(learning_rate is None):
        nn = train.train(neuralnet,inoutputs_array,epochs=epochs)
    elif(epochs is None):
        nn = train.train(neuralnet,inoutputs_array,learning_rate=learning_rate)
    if(learning_rate is not None and epochs is not None):
        nn = train.train(neuralnet,inoutputs_array,learning_rate,epochs)
        


    return


if __name__ == '__main__':

    result = run_script()


