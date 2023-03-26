
import Setup as setup 
import time
import IO as io
import BuildNN as build

def run_script():

    logger,args = setup.setup()
    print('args',args)

    inputs_array,output_array = io.readFile(args.inoutdatafile)

    build.build(args.numoflayers,inputs_array,output_array)
    #time.sleep(1)

    return


if __name__ == '__main__':

    result = run_script()


