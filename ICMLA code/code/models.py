'''
Sequence to sequence anomaly prediction model for `anomaly detection`
'''
from __future__ import print_function
import os
import numpy as np
import json
import h5py
from keras.callbacks import EarlyStopping
from helper import load_data
from keras.optimizers import Adam
from keras import regularizers
kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
from keras.utils import plot_model
path = 'config.json'
if os.path.isfile(path):
    config = json.load(open(path, mode='r'))
batch_size = config['batch-size']  # Batch size for training.
epochs = config['epochs']  # Number of epochs to train for.
# Latent dimensionality of the encoding space.
input_dim = config['input-num-units']
x_window = config['x-window']
y_window = config['y-window']
# optimizer = config["model-optimizer"]
optimizer = Adam(lr=0.0001)
# load the weight file
train_test_split = config["train-test-split"]
batch_size = config['batch-size']
file_name = config['exp-name']['normal']
input_name = config['input-name']


def fit_model():
    """Fit the models using this convenient function"""
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')
    with h5py.File(file_name, 'r') as db:
        nrows = db[input_name].shape[0]
    # retrieve  number of samples for training
    ntrain = int(train_test_split * nrows)
    inputs, target = load_data('normal', ntrain)
    model = rnn_model()
    model.fit(x=inputs, y=target, epochs=epochs, verbose=2,
              shuffle=True, batch_size=batch_size, callbacks=[earlystopping],
              validation_split=0.1)
    weight_file = config['model-weights']
    model.save_weights(weight_file)
    print(
        f'>>> model saved in: {weight_file} <<<')
    return model


def rnn_model():
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Add,Concatenate
    from AttentionLayerWithContext import AttentionLayerWithContext as Attention
    input_1 = Input(shape=(x_window, y_window), name='input_1')
    input_lstm_1 = LSTM(input_dim, return_sequences=False, recurrent_dropout=0,
                        activation='relu', kernel_regularizer=kernel_regularizer,
                        go_backwards=True, name='input_lstm_1')
    lstm_1_output = input_lstm_1(input_1)
    input_lstm_2 = LSTM(input_dim, return_sequences=False, recurrent_dropout=0,
                        activation='relu', kernel_regularizer=kernel_regularizer, name='input_lstm_2')
    lstm_2_output = input_lstm_2(input_1)
    dense_input = Concatenate(axis=-1)([lstm_1_output,lstm_2_output])
    #attention_input = Add()([lstm_1_output, lstm_2_output])
    #state_h = Add()([state_h_1, state_h_2])
    #attention = Attention(name='attention_layer')
    #attention_output = attention(lstm_2_output)
    dense = Dense(y_window, kernel_regularizer=kernel_regularizer,
                  name='dense_output_1')
    dense_output_1 = dense(dense_input)
    model = Model(input_1, dense_output_1)
    # plot the model
    # plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['accuracy'])
    return model
