import numpy as np
import csv
import random

# Load all test inputs to a python list
test_inputs = []
with open('../data/test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for test_input in reader: 
        test_input_no_id = []
        for pixel in test_input[1:]: # Start at index 1 to skip the Id
            test_input_no_id.append(float(pixel))
        test_inputs.append(test_input_no_id) 

# Write a random output for every test_input
test_output_file = open('../data/test_output_random.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',') 
writer.writerow(['Id', 'Prediction']) # write header
for idx, test_input in enumerate(test_inputs):
    random_int = random.randint(0,9)
    row = [idx+1, random_int]
    writer.writerow(row)
test_output_file.close()

