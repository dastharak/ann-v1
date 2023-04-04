import numpy as np

import argparse
import logging
from ActivationFunction import ActivationFunction as AF

modulename = __name__
logger = logging.getLogger(modulename)

def setup():
    parser = argparse.ArgumentParser(description="Artificial Neural Network v1.0")
    parser.add_argument("-lgl","--loglevel", type=str, default='warn', \
        choices=['debug', 'info', 'warn', 'error'], help="Set log level")
    
    parser.add_argument("-iopd","--inoutputdata", type=str, help="Input and output data as command or CSV file")
    parser.add_argument("-nofl","--numoflayers", type=str, help="Number of neurons in each layer")
    parser.add_argument("-actfun","--activationfunctions", type=str, help="Activation function of each layer")
    parser.add_argument("-w4el","--weights4eachlayer", type=str, help="Weights for each layer")
    parser.add_argument("-lr","--learningrate", type=float, help="Learning rate between 0 and 1")
    parser.add_argument("-ep","--epochs", type=int, help="Number of epochs to train whole number > 0")
    
    #parser.add_argument("-resf","--resultfile", type=str, default='Result.txt', help="file")
    #parser.add_argument("-y","--yyyy", type=int, default=None, choices=[1], help="")
    #parser.add_argument("-x","--xxx",nargs='+', type=str, default=None, help="help")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.loglevel=='debug' else logging.INFO if args.loglevel=='info' else logging.WARN if args.loglevel=='warn' else logging.ERROR)
    logger = logging.getLogger(parser.prog)
    print(f"--loglevel:{logger.getEffectiveLevel()}:{args.loglevel}")
    print('args',args)
    #tar = args.target#default value or user input
    args.inoutputdata = getInOutputData(args.inoutputdata)
    logger.debug(f"Arg 1:{args.inoutputdata}")
    args.numoflayers = getNumberOfLayer(args.numoflayers)
    logger.debug(f"Arg 2:{args.numoflayers}")
    args.activationfunctions = getActivationFunctions(args.activationfunctions)
    logger.debug(f"Arg 3:{args.activationfunctions}")
    args.weights4eachlayer = getWeights(args.weights4eachlayer)
    logger.debug(f"Arg 4:{args.weights4eachlayer}")
    args.learningrate = None if args.learningrate is None else float(args.learningrate) if (0 < float(args.learningrate) and 1>float(args.learningrate)) else 'Input Error'
    logger.debug(f"Arg 5:{args.learningrate}")
    logger.debug(f"Arg 6:{args.epochs}")

    #logging.debug("This is a debug line") if debug else None
    'This will print the debug line only if debug is set to True.'

    'Its also worth noting that, you can set the logging level to only print certain levels of messages, like debug, info, warning, error, and critical.'
    return logger,args

    
def getInOutputData(arg1):
    if(arg1.startswith('-csv')):
        #process csv file
        csvFileName = arg1[5:]
        # load CSV file using numpy.loadtxt()
        data = np.loadtxt(csvFileName,dtype=float,skiprows=1, delimiter=',')
    elif(arg1.startswith('-cmd')):
        #process command input
        inoutputData = arg1[5:]
        data = np.array(inoutputData)
    logger.info(f'input:{data}')
    return data

def getNumberOfLayer(arg2):
    layers = []
    arg2 = arg2.strip('[]').split(',')
    for numofneurons in arg2:
        layers.append(int(numofneurons))
    logger.info(f'layers:{layers}')
    return layers

def getActivationFunctions(arg3):
    act_funcs = []
    arg3 = arg3.strip('[]').split(',')
    for afuncname in arg3:
        act_funcs.append(AF(afuncname))
    logger.info(f'actf:{act_funcs}')
    return act_funcs

def getWeights(arg4):
    layer_weights = eval(arg4)
    layer_weights = str_to_float(layer_weights)

    print(layer_weights)
    return layer_weights

def str_to_float(lst):
    if isinstance(lst, list):
        return [str_to_float(i) for i in lst]
    else:
        return float(lst)
