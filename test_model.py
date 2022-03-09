import numpy as np
from pycoral.utils import edgetpu
import argparse
import time


def test_model(filename, time):
    # Load the TFLite model and allocate tensors.
    interpreter = edgetpu.make_interpreter(filename)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()

    interpreter.invoke()

    end_time = time.time()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"Received output {output_data} in {end_time - start_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--time', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    test_model(args.filename, args.time)