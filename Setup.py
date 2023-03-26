import numpy as np
import pandas as pd

from time import time

#import pickle 
import argparse
from datetime import datetime
import os
import io
import logging

modulename = __name__
logger = logging.getLogger(modulename)

def setup():
    parser = argparse.ArgumentParser(description="Artificial Neural Network v1.0")
    parser.add_argument("-lgl","--loglevel", type=str, default='warn', \
        choices=['debug', 'info', 'warn', 'error'], help="Set log level")
    
    parser.add_argument("-iodf","--inoutdatafile", type=str, help="Input and output data as a CSV file")
    parser.add_argument("-nnel","--numnelayers", type=int, help="Number of neurones in each layer")
    
    #parser.add_argument("-fdf","--fulldatafile", type=str, default=fdf, help="CSV file with both training and test data")
    #parser.add_argument("-sff","--selectfeaturesfile", type=str, default=sf, help="CSV file with selected features from the data file")
    #parser.add_argument("-resf","--resultfile", type=str, default='Result.txt', help="Writes the result to a file")
    #parser.add_argument("-remsin", "--removesingles", action="store_true", default=False, help="If a given label has a single sample such can be removed by inclluding this flag")
    #parser.add_argument("-grupsev","--groupseverity", type=int, default=None, choices=[1], help="If the target is severity, its categorization can be grouped as severe and non-severe by setting this to 1")
    #parser.add_argument("-grupwho","--groupwhoscore", type=str, default=None, help="If the target is whoscore, its categorization to groups can be set as i.e.: -grupwho 1-2:1, 3-5:2, 6-8:3, 9-10:4")
    #parser.add_argument("-shmap","--showMap",nargs='+', type=str, default=None, help="Encoding of the features given with this param will be printed to log, ie: -shmap 'Land of birth' ")
    #parser.add_argument("grouping", help="a string representation of the grouping")#required argument example


    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.loglevel=='debug' else logging.INFO if args.loglevel=='info' else logging.WARN if args.loglevel=='warn' else logging.ERROR)
    logger = logging.getLogger(parser.prog)
    print(f"--loglevel:{logger.getEffectiveLevel()}:{args.loglevel}")
    #tar = args.target#default value or user input

    logger.debug(f"Arg 1:{args.inoutdatafile}")
    logger.debug(f"Arg 2:{args.numnelayers}")
    #logger.debug("Arg 2:"+ args.fulldatafile+'')
    #logger.debug("Argument 3:"+ args.selectfeaturesfile+'')
    #logger.debug("Argument 4:"+ str(args.removesingles)+'')
    #grouping 
    #logger.debug("Argument 5:"+ str(args.groupseverity)+'')
    #args.groupwhoscore = parse_grouping(args.groupwhoscore) if args.groupwhoscore is not None else args.groupwhoscore

    #logger.debug("Argument 6:"+ str(args.groupwhoscore)+'')
    #logger.debug("Argument 7:"+ str(args.showMap)+'')
    #logger.debug("Argument 8:"+ str(args.resultfile)+'')

    #logging.debug("This is a debug line") if debug else None
    'This will print the debug line only if debug is set to True.'

    'Its also worth noting that, you can set the logging level to only print certain levels of messages, like debug, info, warning, error, and critical.'
    return logger,args

    