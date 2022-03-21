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
    
    input = input.astype(input_details[0]["dtype"])

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

    if convert:
        output_scale, output_zero_point = output_details[0]["quantization"]
        output = float(output - output_zero_point) * output_scale
    
    if comp_time:
        print(f"Received output {output} in {end_time - start_time} seconds")
    else:
        print(f"Received output {output}")

    return output


def test_model(filename, comp_time, test_acc, compute_acc):
    # Load the TFLite model and allocate tensors.
    interpreter = edgetpu.make_interpreter(filename)
    interpreter.allocate_tensors()

    
    if test_acc:
        with open(PATH_JSON_DATASET, 'r') as f:
            dataset = json.load(f)  
        inputs = dataset['tests']
        results = dataset['results']
        accs = []
        precs = []
        for inputs, expected in zip(inputs, results):
            res = [run_inference(interpreter, np.expand_dims(np.asarray(input), 1), comp_time, convert=True) for input in inputs]
            res = np.asarray().reshape(-1)
            print(f"Got {res}, expected {expected}")
            expected = (np.asarray(expected).reshape(-1) > 0.5)
            res = (res > 0.5)
            accs.append((expected == res).sum() / res.shape[0])
            TP = ((expected == 1) & (res == 1)).sum()
            FP = ((expected == 1) & (res == 0)).sum()
            precs.append(TP / (TP+FP))

        print(f"Accuracy over test dataset: {sum(accs) / len(accs)}")            
        print(f"Precision over test dataset: {sum(precs) / len(precs)}")   
    else:
        run_inference(interpreter, None, comp_time, random=True)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--time', default=False, action='store_true')
    parser.add_argument('--test_acc', default=False, action='store_true')
    parser.add_argument('--compute_acc', default=False, action='store_true')

    args = parser.parse_args()
    test_model(args.filename, args.time, args.test_acc, args.compute_acc)