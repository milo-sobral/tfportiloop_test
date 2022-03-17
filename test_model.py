import numpy as np
from pycoral.utils import edgetpu
import argparse
import time
import json

PATH_JSON_DATASET = './dataset.json'

def run_inference(interpreter, input, comp_time, random=False, convert=False):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if convert:
        input_scale, input_zero_point = input_details[0]["quantization"]
        input = np.asarray(input) / input_scale + input_zero_point
    
    input = input.as_type(input_details["dtype"])

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    if random:
        input = np.array(np.random.random_sample(input_shape), dtype=np.int8)

    interpreter.set_tensor(input_details[0]['index'], input)
    if comp_time:
            start_time = time.time()
            
    interpreter.invoke()

    if comp_time:
        end_time = time.time()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    if comp_time:
        print(f"Received output {output} in {end_time - start_time} seconds")
    else:
        print(f"Received output {output}")

    if convert:
        output_scale, output_zero_point = output_details[0]["quantization"]
        output = float((output - output_zero_point)) / output_scale
    
    return output



def test_model(filename, comp_time, test_acc):
    # Load the TFLite model and allocate tensors.
    interpreter = edgetpu.make_interpreter(filename)
    interpreter.allocate_tensors()

    
    if test_acc:
        with open(PATH_JSON_DATASET, 'r') as f:
            dataset = json.load(f)  
        inputs = dataset['tests']
        results = dataset['results']
        for input, expected in zip(inputs, results):

            res = run_inference(interpreter, input, comp_time, convert=True)
            print(f"Got {res}, expected {expected}")
            
    else:
        run_inference(interpreter, None, comp_time, random=True)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--time', default=False, action='store_true')
    parser.add_argument('--test_acc', default=False, action='store_true')
    args = parser.parse_args()
    test_model(args.filename, args.time, args.test_acc)