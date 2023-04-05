import NeuralNetwork as nn
import numpy as np
import time
import logging
import math as math

modulename = __name__
logger = logging.getLogger(modulename)

def train(nn,inoutputs_array=[],learning_rate=0.1,epochs=1000):
    logger.info(f'inoutputs_array:{inoutputs_array}')
    logger.info(f'learning_rate:{learning_rate}')
    logger.info(f'epochs:{epochs}')
    # set the inputs to the network
    X_0 = inoutputs_array[0,0:-1] #first input
    print(f'X_0:{X_0}')
    nn.set_inputs(X_0)
    y = inoutputs_array[:,-1]
    print(f'y:{y}')
    output = np.zeros_like(y,dtype=float)

    #infor of network
    strBefore = nn.getNetworkSetup()
    # record start time
    start_time = time.time()

    # calculate the output of the network
    out1 = nn.output()
    output_size = len(y)
    err_init = sum(abs(out1-y))/output_size
    logger.info(f'err_init:{err_init}')
    nn.backpropagate(y[0],learning_rate)
    print(f'|{convert_frac_to_symbols(err_init,err_init)}%:ep.{0}')
    for i in range(1,epochs):  #):# 
        k = i%4
        nn.set_inputs(inoutputs_array[k,0:-1])
        # calculate the output of the network
        output[k] = nn.output()[0] # Convert ndarray to float

        #train backwards
        nn.backpropagate(y[k],learning_rate)

        #err = nn.get_error()
        if(i%(epochs/100) == 0 ):
            err_avg = ((sum(abs(output-y))/output_size)) #*100)/err_init
            print(f'|{convert_frac_to_symbols(err_avg,err_init)}%:ep.{i}')
            logger.info(f'err_avg:{err_avg}')

    
    #print(f'ee:{err_avg}')
    # record end time
    end_time = time.time()

    # calculate elapsed time
    elapsed_time = end_time - start_time

    strAfter = nn.getNetworkSetup()
    # print some info
    print(f'Elapsed time:{elapsed_time} seconds')
    print(f'  Net Before:{strBefore}')
    print(f'   Net After:{strAfter}')
    print(f'   Error S,L:{err_init},{err_avg}')
    print(f'      Target:{y}')
    print(f'      Output:{output}')

    return nn

#some visual information to show the learning process
def convert_frac_to_symbols(value,init_err):
    symbols = ''
    abs_value = abs(value)
    abs_value = (abs_value/init_err)*100
    int_value = int(abs_value)
    for s in range(0,int_value):
        symbols += '='
    return symbols+str(int_value)
