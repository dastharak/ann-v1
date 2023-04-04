import csv
import numpy as np

def readFile(fileName):
    with open(fileName,'r') as csvfile:

        csvreader = csv.reader(csvfile)

        inputs = []
        output = []

        for row in csvreader:
            if row[0] in ('Input','input'):
                continue

            input_row = [int(i) for i in row[:-1]]

            output_row = int(row[-1])

            inputs.append(input_row)
            output.append(output_row)

        inputs_array = np.array(inputs)
        output_array = np.array(output)
        
        print(f'inputs_array{inputs_array} : output_array{output_array}')

    return inputs_array,output_array    
