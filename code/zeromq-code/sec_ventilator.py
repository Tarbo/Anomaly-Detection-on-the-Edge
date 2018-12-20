import os
import zmq
import time
import numpy as np
import json
path = 'config.json'
if os.path.isfile(path):
    config = json.load(open(path, mode='r'))


def ventilator(exp_name):
    """Collects workers' results and forwards to detector workers"""
    send_addr = config['sec-ventilator-push-addr']
    recv_addr = config['sec-ventilator-pull-addr']
    mode = config['mode']
    context = zmq.Context()
    # socket to receive messages from
    receiver = context.socket(zmq.PULL)
    receiver.bind(recv_addr)
    # socket to forward messages to
    sender = context.socket(zmq.PUSH)
    sender.bind(send_addr)
    duration = []
    _NUM_STREAMS = 200
    count = 0
    print(f'>>> Secondary ventillator is ready for business <<<')
    while count < _NUM_STREAMS:
        t0 = time.time()
        data = recv_array(receiver)
        err_msg = send_array(sender, data)
        if err_msg != None:
            print(f'>>> Error message: {err_msg}')
        t1 = time.time()
        duration.append(t1 - t0)
        count += 1
    return np.mean(duration)


def recv_array(socket, flags=0, copy=True, track=False):
    """receive a numpy array"""
    md = socket.recv_json(
        flags=flags)  # receives the metadata for decoding the numpy array
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    data = np.frombuffer(buf, dtype=md['dtype'])
    return data.reshape(md['shape'])


def send_array(socket, arr, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(arr.dtype),
        shape=arr.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(arr, flags, copy=copy, track=track)


if __name__ == "__main__":
    result = ventilator('fifo')
    print(f'>>> Average durations primary ventilator: {result} <<<')
