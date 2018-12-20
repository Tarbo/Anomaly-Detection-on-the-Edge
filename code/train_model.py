# Import the necessary library
import os
import time
import json
from multiprocessing import Pool, freeze_support
import numpy as np
import h5py
# %matplotlib inline


def fit_model(exp_name):
    """This function fits the model using a fit_generator"""
    import data_processor as dp
    from model import load_model
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    earlystopping = EarlyStopping(monitor='acc', patience=30)
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2,
                                  patience=5, min_lr=0.001)
    # from keras.callbacks import History
    scenario = 'train'
    config = dp.load_config()
    # modify to either `model-weight-a` or `model-weight`
    filepath = config['model-weight-a'][exp_name]
    start_index = 0

    with h5py.File(config['xy-train'][exp_name], 'r') as db:
        nrows = db[config['encoder-name-db']].shape[0]
        # ncols = db[config['encoder-name-db']].shape[1]
    # retrieve number of features for training
    ntrain = int(config['train-test-split'] * nrows)
    steps_per_epoch = (ntrain / config['model']
                       ['epochs']) // config['batch-size']
    print(f'>>> {exp_name} has {nrows} data rows. Training on {ntrain} rows with {steps_per_epoch} steps-per-epoch <<<')

    epochs = config['model']['epochs']
    print(f'>>> {exp_name}...training on PID:{os.getpid()} <<<')
    model = load_model(exp_name)
    data_gen_train = dp.data_generator(
        exp_name, scenario, start_index=start_index, train_eval=True)
    #history = History()
    model.fit_generator(data_gen_train, steps_per_epoch,
                        epochs=epochs, verbose=2, callbacks=[reduce_lr, earlystopping])
    # for epoch in range(epochs):
    #     data_gen_train = dp.data_generator(
    #         exp_name, scenario, start_index=start_index)
    #     for step in range(steps_per_epoch):
    #         x_data, y_data = next(data_gen_train)
    #         model.fit(
    #             x_data, y_data, batch_size=config['model']['batch-size'], verbose=2, epochs=1, shuffle=False)
    # hist_file = config['history'][exp_name]['']
    # with open(hist_file, 'w') as jsonfile:
    #     json.dump(history.history, jsonfile, indent=4)
    # print('>...Training history saved in {} file'.format(hist_file))
    model.save_weights(filepath)
    # model.save(config['model']['model-weight'][folder_name])
    print(f'>>> Trained model: {exp_name}\tSaved in: {filepath} <<<')
    del model
    return


def process_raw_traces(exp_name):
    import data_processor as dp
    scenario = 'anomalous'
    dp.save_to_database(exp_name, scenario)


if __name__ == '__main__':
    freeze_support()
    exp_name = ['sporadic', 'fifo', 'full', 'hilrf']

    with Pool(processes=4) as pool:
        pool.map(process_raw_traces, exp_name)
