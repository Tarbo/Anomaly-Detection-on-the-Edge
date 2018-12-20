import os
import zmq
import json
import time
import numpy as np
from .. import data_processor as dp

path = 'config.json'
if os.path.isfile(path):
    config = json.load(open(path, mode='r'))


def ventilator(exp_name):
    """Task distributor for distributing the task of RNN classifier prediction
    Args:
        mode_signal: determines if the task is run locally or in the cloud 
    """
    sender_addr = config['prim-ventilator-push-addr']
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    # bind the socket
    sender.bind(sender_addr)
    scenario = 'clean'
    start_index = 0
    train_eval = False
    data = dp.data_generator(exp_name, scenario, start_index, train_eval)
    encoder_data, target_data = next(data)
    duration = []
    _ = input('Please hit any button if the predictor workers are ready')
    print('Dispatching tasks to workers')
    for idx, item in enumerate(encoder_data):
        while idx < 200:
            t0 = time.time()
            err_msg = send_array(sender, item)
            if err_msg != None:
                print(f'>>> Error message: {err_msg} <<<')
            t1 = time.time()
            duration.append(t1 - t0)
    time.sleep(1)
    return np.mean(duration)


def send_array(socket, arr, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(arr.dtype),
        shape=arr.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(arr, flags=flags, copy=copy, track=track)


if __name__ == "__main__":
    result = ventilator('fifo')
    print(f'>>> Average durations primary ventilator: {result} <<<')
