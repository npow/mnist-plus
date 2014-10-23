import numpy as np
import csv

train_inputs = []
with open('../data/train_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
        for dimension in train_input[1:]:
            train_input_no_id.append(float(dimension))
        train_inputs.append(np.asarray(train_input_no_id)) # Load each sample as a numpy array, which is appened to the python list
train_inputs_np = np.asarray(train_inputs)
np.save('../blobs/X_train.npy', train_inputs_np)

train_outputs = []
with open('../data/train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        train_output_no_id = int(train_output[1])
        train_outputs.append(train_output_no_id)

train_outputs_np = np.asarray(train_outputs)
np.save('../blobs/Y_train.npy', train_outputs_np)

test_inputs = []
with open('../data/test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
        for dimension in train_input[1:]:
            train_input_no_id.append(float(dimension))
        test_inputs.append(np.asarray(train_input_no_id)) # Load each sample as a numpy array, which is appened to the python list

test_inputs_np = np.asarray(test_inputs)
np.save('../blobs/X_test.npy', test_inputs_np)
