import os
import h5py
import json
import numpy as np
import pandas as pd
#from gensim.models.word2vec import Word2Vec

path = 'config.json'
if os.path.isfile(path):
    config = json.load(open(path, mode='r'))


def load_data(exp_name, start_index=0):
    """Load data from the database"""
    #from keras.utils import to_categorical

    xy_db = config['exp-name'][exp_name]
    # retrieve  number of samples for training
    input_name = config['input-name']
    target_name = config['target-name']
    batch_size = config['batch-size']
    with h5py.File(xy_db, 'r') as db:
        total_num_items = db[input_name].shape[0]
        current_index = start_index
        while (current_index + batch_size < total_num_items):
            input_data = db[input_name][current_index:current_index + batch_size]
            # target_data = to_categorical(
            #     db[target_name][index:], num_classes=feature_dim)
            target_data = db[target_name][current_index:current_index + batch_size]
            current_index += batch_size
            yield (input_data, target_data)
