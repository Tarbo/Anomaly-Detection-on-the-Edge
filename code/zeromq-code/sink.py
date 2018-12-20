import time
import os
import zmq
import json
import numpy as np
path = 'config.json'
if os.path.isfile(path):
    config = json.load(open(path, mode='r'))


def sink():
    """This serves as the sink for displaying accuracy results"""
    recv_addr = config['sink-pull-addr']
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(recv_addr)
    print(f'>>> Receiver booted up. Awaiting reception <<<')
    for task_nbr in np.arange(200):
        accuracy = receiver.recv()
    print(f'>>> Reception Completed. Shutting down <<<')
    return


if __name__ == "__main__":
    sink()
