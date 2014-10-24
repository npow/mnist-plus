import numpy as np
import csv
from scipy import misc


# Save all training inputs as pngs
with open('train_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for idx, train_input in enumerate(reader): 
        train_input_no_id = []
        for dimension in train_input[1:]:
            train_input_no_id.append(float(dimension))
        train_input_np = np.asarray(train_input_no_id)
        train_input_np = np.reshape(train_input_np, (48,48))
        misc.imsave('train_images/' + str(idx+1) + '.png', train_input_np)
        

# Save all test inputs to as pngs
with open('test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for idx, test_input in enumerate(reader): 
        test_input_no_id = []
        for dimension in test_input[1:]:
            test_input_no_id.append(float(dimension))
        test_input_np = np.asarray(test_input_no_id)
        test_input_np = np.reshape(test_input_np, (48,48))
        misc.imsave('test_images/' + str(idx+1) + '.png', test_input_np)


