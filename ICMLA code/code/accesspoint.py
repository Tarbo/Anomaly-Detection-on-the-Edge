"""
@author: Okwudili Ezeme
@date: 2019-June-27
This runs on the access point device and helps in load management and optimization
"""
import zmq
import pyre
import pickle
import time
import logging
from numpy import random
from itertools import cycle
import threading

logger = logging.getLogger(__name__)

random.seed(10)


class AccessPoint:
    def __init__(self, ctx, group_name):
        self.AP = pyre.Pyre('AP')
        self.AP.join(group_name)
        self.AP.start()
        self.group_name = group_name
        # used for authenticating replies
        self.sequence_num = cycle([1, 2, 4, 5, 6, 7, 8, 9, 10])
        self.lock = threading.Lock()
        self.active_seq = None

    def send_request(self):
        logger.debug(f'>>> {threading.current_thread().name} thread started')
        while True:
            self.lock.acquire()
            try:
                self.active_seq = next(self.sequence_num)
                msg = {'msg-type': 'REQUEST',
                       'uuid': None,
                       'table': None,
                       'processing-time': None,
                       'request-time': time.time(),
                       'sequence-ID': self.active_seq
                       }
                msg_b = pickle.dumps(msg, -1)
                self.AP.shout(self.group_name, msg_b)
                logger.debug(f'>>> AP sent msg to the group')
            finally:
                self.lock.release()
            time.sleep(5)

    def update(self):
        logger.debug(f'>>> {threading.current_thread().name} thread started')
        table_list = []
        events = self.AP.events()
        while True:
            if events:
                try:
                    event = next(events)
                    if event.type == 'WHISPER':
                        msg = pickle.loads(event.msg[0])
                        logger.debug(f'>>> {event.peer_name} replied')
                        self.lock.acquire()
                        if msg['sequence-ID'] == self.active_seq:
                            uuid = msg['uuid']
                            # add the processing and transmission time
                            duration = msg['processing-time'] + \
                                2 * (time.time() - msg['request-time']
                                     )
                            table_list.append((uuid, duration))
                            self.lock.release()
                        else:
                            table_dict = dict(table_list)
                            msg_update = {'msg-type': 'UPDATE',
                                          'table': table_dict}
                            msg_update_b = pickle.dumps(msg_update, -1)
                            self.AP.shout(self.group_name, msg_update_b)
                            table_list = []
                            uuid = msg['uuid']
                            # add the processing and transmission time
                            duration = msg['processing-time'] + \
                                2 * (time.time() - msg['request-time']
                                     )
                            table_list.append((uuid, duration))
                except Exception as identifier:
                    logger.debug(f'>>> Exception: {identifier}')
                    self.AP.leave(self.group_name)
                    self.AP.stop()  # clean up if there are issues

    def run(self):
        t1 = threading.Thread(target=self.send_request, name='send request')
        t2 = threading.Thread(target=self.update, name='update')
        threads = [t1, t2]
        try:
            for thread in threads:
                thread.start()
        except Exception as err:
            logger.debug(f'>>> Exception: {err}')
            self.AP.leave(self.group_name)
            self.AP.stop()  # clean up after errors


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger('__main__').setLevel(logging.DEBUG)

    ctx = zmq.Context()
    group_name = 'Ezeme'
    AP = AccessPoint(ctx, group_name)
    AP.run()
