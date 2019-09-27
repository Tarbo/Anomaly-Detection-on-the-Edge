import os
import numpy as np
import csv
import pandas as pd
import h5py
import json
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import RobustScaler


def vectorize_data():
    """A function to store the data in the format we want"""
    # read in the data
    data_path = '../dataset/'
    path = 'config.json'
    if os.path.isfile(path):
        config = json.load(open(path, mode='r'))
    #feature_width = config['feature-width'] + 2
    x_window = config['x-window']
    y_window = config['y-window']
    assert x_window >= 1, "Input timesteps X window must be >= 1"
    assert y_window >= 1, "Prediction steps Y window must be >= 1"
    input_var = ['normal2.csv', 'delay2.csv', 'random2.csv']
    #target_var = ['normal1.csv', 'delay1.csv', 'random1.csv']
    #db_path = config["data-path"]
    db_name = ['normal.h5', 'delay.h5', 'random.h5']
    #feature_width = config["feature-width"]
    for idx, item in enumerate(input_var):
        in_var = os.path.join(data_path, item)
        # read the system calls column
        input_df = pd.read_csv(in_var, dtype='float64', header=0, usecols=[0])
        input_list = input_df.values.tolist()
        # read the target list
        # output_df = pd.read_csv(in_var, header=0, dtype=np.int8, usecols=[1])
        #target_list = input_list
        # check the dimensions
        num_rows = np.array(input_list).shape[0]
        #num_cols = np.array(input_list).shape[1]
        input_data = []
        target_data = []
        index = 0
        while (index + x_window + y_window) <= num_rows:
            input_window_data = input_list[index:(index + x_window)]
            target_window_data = input_list[(
                index + x_window): (index + x_window + y_window)]
            # start and end of sequences are 0 and 315 respectively
            # input_window_data = np.array(
            #     input_window_data).flatten().tolist()
            # target_window_data = np.array(
            #     target_window_data).flatten().tolist()
            input_data.append(input_window_data)
            target_data.append(target_window_data)
            index += y_window
        print(f'>>> Done vectorizing: {item}. Now Scaling the input <<<')
        # input_data = StandardScaler().fit_transform(np.array(input_data))
        # print('>>> Done Scaling the input data <<<')
        input_data = np.array(input_data)
        target_data = np.array(target_data)
        print(
            f'>>> sample input:{input_data[0:5]}\n>>> Sample target: {target_data[0:5]}')
        n_features = config['num-features']
        input_data = np.reshape(
            input_data, (input_data.shape[0], input_data.shape[1], n_features))
        target_data = np.reshape(
            target_data, (target_data.shape[0], target_data.shape[1]))
        print(f'>>> Done reshaping: {item}. About to store in database')
        print(
            f'>>> Input shape: {input_data.shape} <<<\n>>> Target data shape:{target_data.shape} <<<')
        with h5py.File(db_name[idx], 'w') as db:
            # initialize the hdfs file with first chunk
            encoder_input = db.create_dataset(
                "input", shape=input_data.shape, dtype=np.int32)
            encoder_input[:] = input_data
            decoder_target = db.create_dataset(
                "target", shape=target_data.shape, dtype=np.int32)
            decoder_target[:] = target_data
            print(
                f'>>> {item} data stored successfully in: {db_name[idx]} <<<')
            print(
                f'>>> stored sample: {input_data[0]}<<<\n>>>{target_data[0]} <<<')
    return


if __name__ == '__main__':
    vectorize_data()
