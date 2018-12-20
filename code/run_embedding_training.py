from multiprocessing import Pool
from itertools import zip_longest
import data_processor as dp
import os


def train_embedding(exp_name):
    print(f'PID:{os.getpid()}')
    data = 'data'
    dp.train_embedding_weight(data, exp_name)


if __name__ == '__main__':
    exp_names = ['fifo', 'sporadic', 'hilrf', 'full']
    with Pool(processes=len(exp_names)) as pool:
        pool.map(train_embedding,  exp_names)
    print('End of the Pool processes')
