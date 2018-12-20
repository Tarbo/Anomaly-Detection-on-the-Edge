import os
import zmq
import time
import numpy as np
import json
from multiprocessing import Pool
path = 'config.json'
if os.path.isfile(path):
    config = json.load(open(path, mode='r'))


def workers(exp_name):
    """Collects workloads from ventilator and does prediction of sequences"""
    from .. import model
    recv_addr = config['prim-ventilator-push-addr']
    send_addr = config['sec-ventilator-pull-addr']
    mode = config['mode']
    context = zmq.Context()
    # socket to receive messages from
    receiver = context.socket(zmq.PULL)
    receiver.connect(recv_addr)
    # socket to forward messages to
    sender = context.socket(zmq.PUSH)
    sender.connect(send_addr)
    duration = []
    count = 0
    print(f'>>> Starting the predictor worker of process: {os.getpid()} <<<')
    while count < 40:
        t0 = time.time()
        data = recv_array(receiver)
        if mode == 'attention':
            pred_seq = model.inference_attention(exp_name, data)
        else:
            pred_seq = model.inference_no_attention(exp_name, data)
        pred_seq = np.array(pred_seq)
        err_msg = send_array(sender, pred_seq)
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


if __name__ == '__main__':
    with Pool(4) as pool:
        exp_name = config['exp-name']
        print(
            f'>>> Average durations for predictor workers: {pool.map(workers,[exp_name,exp_name,exp_name,exp_name])} <<<')
