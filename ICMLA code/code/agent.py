"""
@author: Okwudili Ezeme
@date: 2019-June-27
This will run on each of the devices to be connected to the access point
"""
import os
import zmq
import time
import logging
import socket
import pickle
import h5py
import threading
import numpy as np
from itertools import cycle
from helper import load_data
#from models import rnn_model
from queue import Queue  # explore use of priority too
from pyre import Pyre, PyreEvent
from numpy import random, mean
#from multiprocessing import Process as p

logger = logging.getLogger(__name__)

random.seed(150)


class Agent:
    """
    A class object that represents each app in the network"""

    def __init__(self, name, ctx, group_name, cpu_clock_rate, experiment_name):
        self.lock = threading.Lock()
        self.cpu_clock_rate = cpu_clock_rate
        self.cpu_load = random.random()
        self.group_name = group_name
        self.routing_table = None
        self.name = name + str(os.getpid())
        self.tasks = Queue(-1)
        self.results = Queue(-1)
        self.exp_name = experiment_name
        self.task_duration_no_context = random.random()
        # compute duration using cpu load, etc
        self.task_duration_with_context = random.random()
        #self.weights = 'rnn-model-attention-weights.h5'
        #self.model = rnn_model()
        # self.model._make_predict_function()
        # self.model.load_weights(self.weights)
        self.agent = Pyre(
            name=self.name, ctx=ctx or zmq.Context.instance())
        try:
            self.agent.join(group_name)
            self.agent.start()
        except Exception as err:
            logger.error(f'>>> Cant start node: {err}', exc_info=True)

    def routing_table_setter(self, table):
        self.lock.acquire()
        try:
            # create an ascending round robin routing principle
            self.routing_table = cycle(
                sorted(table.items(), key=lambda x: x[1]))
        finally:
            self.lock.release()

    def add_task(self):
        """populates the task queue with new data for inference"""
        logger.debug(f'>>> {threading.current_thread().name} started')
        self.data = cycle(load_data(self.exp_name, 0))
        count = 0
        while count < 100:
            task_dict = dict.fromkeys(
                ['input', 'target', 'task-type', 'task-uuid', 'task-owner-name', 'result', 'duration'], 0)
            try:
                input_data, target_data = next(self.data)
                task_dict['input'] = input_data
                task_dict['target'] = target_data
                task_dict['task-type'] = 1
                task_dict['task-uuid'] = self.agent.uuid()
                task_dict['task-owner-name'] = self.agent.name()
                task_dict['duration'] = time.time()
                self.tasks.put(task_dict)
                count += 1
            except Exception as err:
                logger.error(f'>>> Exception type: {err}', exc_info=True)
                self.agent.leave(self.group_name)
                self.agent.stop()
            # Vary the frequency of input tasks
            time.sleep(random.randint(1, 8))

    def vary_cpu_load(self):
        logger.debug(
            f'>>> {threading.current_thread().name} thread started')
        while True:
            try:
                self.lock.acquire()
                self.cpu_load = random.random()
                self.lock.release()
                self.compute_duration_with_context()
            except Exception as err:
                logger.error(f'>>> Exception: {err}', exc_info=True)
            time.sleep(random.randint(10, 40))

    def compute_duration_with_context(self):
        try:
            self.lock.acquire()
            cpu_load = self.cpu_load
            task_duration_no_context = self.task_duration_no_context
            self.task_duration_with_context = (
                1 / task_duration_no_context) / (cpu_load * self.cpu_clock_rate)
            self.lock.release()
        except Exception as identifier:
            logger.error(f'>>> Exception: {identifier}')

    def compute_local(self, task):
        """argument is task"""
        try:
            task = task
            task_data = task['input']
            target = task['target']
            uuid = task['task-uuid']
            #predictions = self.model.predict(task_data, verbose=0)
            #predictions = predictions.flatten()
            # flatten the target
            average = mean(task_data.flatten())
            # window = 5
            # errors = self.regression_error(predictions, target, window)
            # mu, variance = np.mean(errors), np.var(errors)
            # probabilities = self.chebyshev_probability(mu, variance, errors)
            task['task-type'] = task['task-type'] + 1
            if uuid == self.agent.uuid():  # put results in our queue if its our uuid
                self.results.put(average)
                self.lock.acquire()
                self.task_duration_no_context = time.time() - task['duration']
                self.lock.release()
                self.compute_duration_with_context()
            else:
                task['result'] = average
                data_byte = pickle.dumps(task, -1)
                self.agent.whisper(uuid, data_byte)
                logger.error(
                    f'>>> Results sent back to task owner peer: {task["task-owner-name"]}')
        except Exception as identifier:
            logger.error(f'>>> Exception type: {identifier}', exc_info=True)
            self.agent.leave(self.group_name)
            self.agent.stop()  # clean up if there are issues.

    def check_results(self):
        logger.error(f'>>> {threading.current_thread().name} thread started')
        while True:
            try:
                if not self.results.empty():
                    result = self.results.get()
                    if result <= 0.25:
                        logger.error(
                            f'>>> Critical anomaly detected: {result}')
                    elif result > 0.25 and result < 0.5:
                        logger.error(
                            f'>>> Severe anomaly detected: {result}')
                    elif result > 0.5 and result < 0.75:
                        logger.error(
                            f'>>> Serious anomaly detected: {result}')
                    else:
                        logger.error(f'>>> Mild anomaly detected: {result}')
            except Exception as err:
                logger.error(f'>>> Exception: {err}', exc_info=True)
                self.agent.leave(self.group_name)
                self.agent.stop()

    def outbox(self, task, peer_uuid):
        try:
            task = pickle.dumps(task, -1)
            self.agent.whisper(peer_uuid, task)
        except Exception as identifier:
            logger.error(f'>>> Exception: {identifier}',exc_info=True)
            self.agent.leave(self.group_name)
            self.agent.stop()

    def num_of_peers(self, table):
        seen = []
        for peer in table:
            if peer[0] in seen:
                return len(seen)
            else:
                seen.append(peer[0])

    def handle_task(self):
        # decide if to compute locally or offload
        logger.error(f'>>> {threading.current_thread().name} thread started')
        while True:
            try:
                if not self.tasks.empty():
                    task = self.tasks.get()
                    self.lock.acquire()
                    local_duration = self.task_duration_with_context
                    table = self.routing_table
                    if table:
                        peer = next(table)  # peer = (uuid, latency)
                        if peer[1] < local_duration:
                            self.outbox(task, peer[0])
                            logger.debug(f'>>> Task offloaded')
                        else:
                            num_of_peers = self.num_of_peers(table)
                            peer = self.search_table(
                                table, num_of_peers, local_duration)
                            if peer:
                                self.lock.release()
                                self.outbox(task, peer[0])
                                logger.debug(f'>>> Task offloaded')
                            else:
                                self.compute_local(task)
                                logger.debug(f'>>> Task computed locally')
                    else:
                        self.compute_local(task)
                        logger.debug(f'>>> Task computed locally')
            except Exception as identifier:
                logger.error(
                    f'>>> Exception type : {identifier}', exc_info=True)
                self.agent.leave(self.group_name)
                self.agent.stop()  # stop if there are issues
            time.sleep(random.randint(0, 3))

    def search_table(self, table, num_of_peers, local_dur):
        for id in range(num_of_peers):
            peer = next(table)
            if peer[1] < local_dur:
                return peer
            else:
                return None

    def inbox(self):
        logger.error(f'>>> {threading.current_thread().name} thread started')
        try:
            events = self.agent.events()  # works like charm
            while True:
                if events:
                    event = next(events)
                    logger.error(f'>>> MSG TYPE: {event.type}')
                    logger.error(f'>>> Sender Agent Name: {event.peer_name}')
                    if event.type == 'WHISPER':
                        msg = pickle.loads(event.msg[0])
                        if msg['task-type'] == 2:
                            result = msg['result']
                            self.results.put(result)
                        elif msg['task-type'] == 1:  # peer sent us a task to execute
                            self.tasks.put(msg)
                    elif event.type == 'SHOUT':  # message from the Access Point AP
                        msg = pickle.loads(event.msg[0])
                        if msg['msg-type'] == 'REQUEST':
                            msg['uuid'] = self.agent.uuid()
                            self.lock.acquire()
                            msg['processing-time'] = self.task_duration_with_context
                            self.lock.release()
                            msg_b = pickle.dumps(msg, -1)
                            self.agent.whisper(event.peer_uuid, msg_b)
                        elif msg['msg-type'] == 'UPDATE':
                            table = msg['table']
                            own_uuid = self.agent.uuid()
                            if own_uuid in table.keys():
                                # remove our own UUID to avoid offloading to ourselves
                                del table[own_uuid]
                            self.routing_table_setter(table)
        except Exception as identifier:
            logger.error(f'>>> Exception type: {identifier}', exc_info=True)
            self.agent.leave(self.group_name)
            self.agent.stop()  # leave the cluster if you have issues

    # compute the chebyshev probability
    def chebyshev_probability(self, average, varianse, error_val):
        probability = []
        for val in error_val:
            if val - average >= 1:
                prob = varianse / ((val - average)**2)
                probability.append(prob)
        return probability

    def regression_error(self, outcome, truth, window):
        n_data = len(truth)
        count = 0
        errors = []
        while count + window <= n_data:
            error = [abs(y_pred - y_truth) for y_pred, y_truth in zip(
                outcome[count:count + window], truth[count:count + window])]
            errors.append(np.mean(error))
            count += window
        return errors

    def run(self):
        # start the threads here
        t1 = threading.Thread(target=self.add_task, name='add task')
        t2 = threading.Thread(target=self.vary_cpu_load, name='vary cpu load')
        t3 = threading.Thread(target=self.check_results, name='check results')
        t4 = threading.Thread(target=self.handle_task, name='handle task')
        t5 = threading.Thread(target=self.inbox, name='inbox')
        threads = [t1, t2, t3, t4, t5]
        try:
            for thread in threads:
                thread.start()
        except Exception as err:
            logger.error(f'>>> Exception: {err}', exc_info=True)
            self.agent.leave(self.group_name)
            self.agent.stop()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger('__main__').setLevel(logging.DEBUG)
    # declare the agent class argiments
    group_name = 'Ezeme'
    ctx = zmq.Context()
    agent_name = 'Chimdalu'
    cpu_clock_rate = 50000
    exp_name = 'delay'
    dalu = Agent(agent_name, ctx, group_name, cpu_clock_rate, exp_name)
    try:
        dalu.run()
    except (KeyboardInterrupt):
        raise
