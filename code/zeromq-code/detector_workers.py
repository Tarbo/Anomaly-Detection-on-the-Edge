import os
import zmq
import time
import numpy as np
import json
import joblib as jb
import multiprocessing as mp
path = 'config.json'
if os.path.isfile(path):
    config = json.load(open(path, mode='r'))


def worker(pred_size):
    """Detector using the KNN classifier"""
    kernel = jb.load('kernel-5.joblib')
    classifier = jb.load('y5-detector.joblib')
    # generate the true value for simulation
    recv_addr = config['sec-ventilator-push-addr']
    send_addr = config['sink-pull-addr']
    context = zmq.Context()
    # receiver socket
    receiver = context.socket(zmq.PULL)
    receiver.connect(recv_addr)
    # sender socket
    sender = context.socket(zmq.PUSH)
    sender.connect(send_addr)
    dummy_target = np.random.randint(1, 314, size=(pred_size))
    duration = []
    iteration_per_worker = 40
    count = 0
    print(f'>>> Starting the detector worker of process: {os.getpid()} <<<')
    while count < iteration_per_worker:
        t0 = time.time()
        data = recv_array(receiver)
        assert data.shape[0] == pred_size
        pdf = kernel.pdf(np.mean(dummy_target != data))
        pdf_reshaped = np.random.random(
            size=len(pdf) * 2).reshape(len(pdf), 2)
        predicted = classifier.predict(pdf_reshaped)
        dummy_class = np.random.randint(2)
        percentage_accuracy = np.mean(predicted == dummy_class)
        sender.send(percentage_accuracy)
        duration.append(time.time() - t0)
        count += 1
    return duration


def recv_array(socket, flags=0, copy=True, track=False):
    """receive a numpy array"""
    md = socket.recv_json(
        flags=flags)  # receives the metadata for decoding the numpy array
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    data = np.frombuffer(buf, dtype=md['dtype'])
    return data.reshape(md['shape'])


if __name__ == "__main__":
    with mp.Pool(4) as pool:
        print(
            f'>>> Average durations detector workers: {pool.map(worker,[5,5,5,5])} <<<')
