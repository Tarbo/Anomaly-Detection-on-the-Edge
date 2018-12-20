import json
import warnings
import numpy as np
from keras.layers import Dense, Input, LSTM, Bidirectional, Embedding
from keras.models import Model
# from keras import backend as K
from AttentionLayerWithContext import AttentionLayerWithContext
from keras.utils import plot_model
from data_processor import load_config


#vocab = json.load(open(configs['embed_model']['vocab_file'], mode='r'))
warnings.filterwarnings("ignore")  # Hide messy Numpy warning


def model_no_attention(exp_name):
    """This function constructs the model"""
    # Lets load the json configuration file
    configs = load_config()
    vocabs = json.load(
        open(configs['vocabs'][exp_name], 'r'))
    num_decoder_tokens = len(vocabs.keys()) + 2
    input_len = configs['x-window']  # from the dataset
    num_of_units = configs['num-units']
    # weights = np.load(
    #     open(configs['embed-weights'][exp_name], 'rb'))
    encoder_inputs = Input(
        shape=(None, num_decoder_tokens - 1), name='encoder_inputs')
    # encoder_embedding = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[
    #     weights], trainable=False, input_length=input_len, name='encoder_embedding')
    # decoder_embed_output = encoder_embedding(encoder_inputs)
    # encoder_output = Bidirectional(LSTM(units=num_of_units, return_sequences=True,
    #                               name='bidirectional_layer'), merge_mode='sum')(embed_output)
    encoder_output, state_h, state_c = LSTM(units=num_of_units, go_backwards=True,
                                            name='encoder_lstm_1', return_sequences=True, return_state=True)(encoder_inputs)
    #attention = AttentionLayerWithContext(name='attention_layer')
    #attention_output = attention(encoder_output)
    encoder_states = [state_h, state_c]
    # setup the decoder
    decoder_inputs = Input(
        shape=(None, num_decoder_tokens), name='decoder_input')
    # decoder_embedding = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[
    #     weights], trainable=False, input_length=input_len, name='decoder_embedding')
    #decoder_embed_output = decoder_embedding(decoder_inputs)
    decoder_lstm_2 = LSTM(num_of_units, return_state=True,
                          return_sequences=True, name='decoder_lstm_2')
    decoder_output_2, _, _ = decoder_lstm_2(
        decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens,
                          activation='softmax', name='dense_output')
    decoder_outputs = decoder_dense(decoder_output_2)
    # dense_output = Dense(
    #     weights.shape[0], activation='softmax', name='dense_main')(attention_output)
    model = model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # plot the model
    #plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(optimizer=configs['model']['optimizer_function'][1],
                  loss=configs['model']['loss_function'], metrics=configs['model']['metrics'])
    print(f'>>> {exp_name} model constructed <<<')
    return model


def inference_no_attention(exp_name, data):
    """Make predictions to test the model quality using attention layer"""
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense
#   import h5py
    #from keras.optimizers import SGD
    #from AttentionLayerWithContext import AttentionLayerWithContext as Attention
#    import data_processor as dp
    configs = load_config()
    vocabs = json.load(
        open(configs['vocabs'][exp_name], 'r'))
    num_decoder_tokens = len(vocabs.keys()) + 2
    max_id = max(vocabs.values())
    end_of_seq = max_id + 1
    num_of_units = configs['num-units']
    # if scenario == 'clean':
    #     with h5py.File(configs['xy-clean'][exp_name], 'r') as db:
    #         ntest = db[configs['encoder-name-db']].shape[0]
    #     # retrieve number of features for training
    #     start_index = 0
    #     data_gen_eval = dp.data_generator(
    #         exp_name, scenario, start_index=start_index, train_eval=False)
    #     steps_test = ntest // configs['batch-size']
    # else:
    #     with h5py.File(configs['xy-anomaly'][exp_name], 'r') as db:
    #         nrows = db[configs['encoder-name-db']].shape[0]
    #     neval = int(0.5 * nrows)
    #     ntest = nrows - neval
    #     data_gen_eval = dp.data_generator(
    #         exp_name, scenario, start_index=ntest, train_eval=False)
    #     steps_test = neval // configs['batch-size']
    encoder_inputs = Input(
        shape=(None, num_decoder_tokens - 1), name='encoder_inputs')
    # encoder_lstm_1 = LSTM(encoder_dim, return_sequences=True,
    #                       go_backwards=True, name='encoder_lstm_1')(encoder_inputs)
    encoder_lstm_2 = LSTM(units=num_of_units, go_backwards=True,
                          name='encoder_lstm_1', return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm_2(encoder_inputs)
    #attention = Attention(name='attention_layer')
    #attention_output = attention(encoder_outputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.load_weights(
        configs['model-weight'][exp_name], by_name=True)
    # Set up the decoder inference model
    decoder_state_input_h = Input(shape=(num_of_units,))
    decoder_state_input_c = Input(shape=(num_of_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs = Input(
        shape=(None, num_decoder_tokens), name='decoder_input')
    # decoder_lstm_1 = LSTM(
    #     decoder_dim, return_sequences=True, return_state=True, name='decoder_lstm_1')
    # decoder_output_1, state_h, state_c = decoder_lstm_1(
    #     decoder_inputs, initial_state=decoder_states_inputs)
    decoder_lstm_2 = LSTM(num_of_units, return_state=True,
                          return_sequences=True, name='decoder_lstm_2')
    decoder_output_2, state_h, state_c = decoder_lstm_2(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_state = [state_h, state_c]
    decoder_dense = Dense(num_decoder_tokens,
                          activation='softmax', name='dense_output')
    decoder_outputs = decoder_dense(decoder_output_2)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_state)
    decoder_model.load_weights(
        configs['model-weight'][exp_name], by_name=True)
    # generate the data needed for inference
    #data_gen_eval = dp.data_generator(exp_name, scenario, start_index,train_eval=False)
    # count = 0
    y_window = configs['y-window']
#    predicted_sequences = []
    # true_sequences = []
    print(f'>>> Model constructed. Doing inference now <<<')
    # while count < steps_test:
    #     count += 1
#    predicted = []
#    encoder_data, decoder_targets = next(data_gen_eval)
#    true_sequences.append(decoder_targets.argmax(2))
#    for data in encoder_data:
    # generate the start of sequence character
    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype=int)
    target_seq[0, 0, 0] = 1
    data = data.reshape(1, data.shape[0], data.shape[1])
    states_value = encoder_model.predict(data)
    stop_condition = False
    decoded_sequence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_digit = np.argmax(output_tokens[0, -1, :])
        # check for exit condition
        if sampled_digit == end_of_seq or len(decoded_sequence) >= y_window:
            stop_condition = True
        else:
            decoded_sequence.append(sampled_digit)
        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype=int)
        target_seq[0, 0, sampled_digit] = 1
        # update states
        states_value = [h, c]
    # predicted.append(decoded_sequence)
    # predicted_sequences.append(predicted)
    return (decoded_sequence)


def model_attention(exp_name):
    """This function constructs the model"""
    # Lets load the json configuration file
    configs = load_config()
    vocabs = json.load(
        open(configs['vocabs'][exp_name], 'r'))
    num_decoder_tokens = len(vocabs.keys()) + 2
    # input_len = configs['x-window']  # from the dataset
    num_of_units = configs['num-units']
    # weights = np.load(
    #     open(configs['embed-weights'][exp_name], 'rb'))
    encoder_inputs = Input(
        shape=(None, num_decoder_tokens - 1), name='encoder_inputs')
    # encoder_embedding = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[
    #     weights], trainable=False, input_length=input_len, name='encoder_embedding')
    # decoder_embed_output = encoder_embedding(encoder_inputs)
    # encoder_output = Bidirectional(LSTM(units=num_of_units, return_sequences=True,
    #                               name='bidirectional_layer'), merge_mode='sum')(embed_output)
    encoder_output, state_h, _ = LSTM(units=num_of_units, go_backwards=True,
                                      name='encoder_lstm_1', return_sequences=True, return_state=True)(encoder_inputs)
    attention = AttentionLayerWithContext(name='attention_layer')
    attention_output = attention(encoder_output)
    encoder_states = [state_h, attention_output]
    # setup the decoder
    decoder_inputs = Input(
        shape=(None, num_decoder_tokens), name='decoder_input')
    # decoder_embedding = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[
    #     weights], trainable=False, input_length=input_len, name='decoder_embedding')
    # decoder_embed_output = decoder_embedding(decoder_inputs)
    decoder_lstm_2 = LSTM(num_of_units, return_state=True,
                          return_sequences=True, name='decoder_lstm_2')
    decoder_output_2, _, _ = decoder_lstm_2(
        decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens,
                          activation='softmax', name='dense_output')
    decoder_outputs = decoder_dense(decoder_output_2)
    # dense_output = Dense(
    #     weights.shape[0], activation='softmax', name='dense_main')(attention_output)
    model = model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # plot the model
    #plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(optimizer=configs['model']['optimizer_function'][1],
                  loss=configs['model']['loss_function'], metrics=configs['model']['metrics'])
    print(f'>>> {exp_name} model constructed <<<')
    return model


def inference_attention(exp_name, data):
    """Make predictions to test the model quality using attention layer"""
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense
    import h5py
    #from keras.optimizers import SGD
    from AttentionLayerWithContext import AttentionLayerWithContext as Attention
#    import data_processor as dp
    configs = load_config()
    vocabs = json.load(
        open(configs['vocabs'][exp_name], 'r'))
    num_decoder_tokens = len(vocabs.keys()) + 2
    max_id = max(vocabs.values())
    end_of_seq = max_id + 1
    num_of_units = configs['num-units']
    # if scenario == 'clean':
    #     with h5py.File(configs['xy-clean'][exp_name], 'r') as db:
    #         ntest = db[configs['encoder-name-db']].shape[0]
    #     # retrieve number of features for training
    #     start_index = 0
    #     data_gen_eval = dp.data_generator(
    #         exp_name, scenario, start_index=start_index, train_eval=False)
    #     steps_test = ntest // configs['batch-size']
    # else:
    #     with h5py.File(configs['xy-anomaly'][exp_name], 'r') as db:
    #         nrows = db[configs['encoder-name-db']].shape[0]
    #     neval = int(0.5 * nrows)
    #     ntest = nrows - neval
    #     data_gen_eval = dp.data_generator(
    #         exp_name, scenario, start_index=ntest, train_eval=False)
    #     steps_test = neval // configs['batch-size']
    encoder_inputs = Input(
        shape=(None, num_decoder_tokens - 1), name='encoder_inputs')
    # encoder_lstm_1 = LSTM(encoder_dim, return_sequences=True,
    #                       go_backwards=True, name='encoder_lstm_1')(encoder_inputs)
    encoder_lstm_2 = LSTM(units=num_of_units, go_backwards=True,
                          name='encoder_lstm_1', return_sequences=True, return_state=True)
    encoder_outputs, state_h, _ = encoder_lstm_2(encoder_inputs)
    attention = Attention(name='attention_layer')
    attention_output = attention(encoder_outputs)
    encoder_states = [state_h, attention_output]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.load_weights(
        configs['model-weight-a'][exp_name], by_name=True)
    # Set up the decoder inference model
    decoder_state_input_h = Input(shape=(num_of_units,))
    decoder_state_input_c = Input(shape=(num_of_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs = Input(
        shape=(None, num_decoder_tokens), name='decoder_input')
    # decoder_lstm_1 = LSTM(
    #     decoder_dim, return_sequences=True, return_state=True, name='decoder_lstm_1')
    # decoder_output_1, state_h, state_c = decoder_lstm_1(
    #     decoder_inputs, initial_state=decoder_states_inputs)
    decoder_lstm_2 = LSTM(num_of_units, return_state=True,
                          return_sequences=True, name='decoder_lstm_2')
    decoder_output_2, state_h, state_c = decoder_lstm_2(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_state = [state_h, state_c]
    decoder_dense = Dense(num_decoder_tokens,
                          activation='softmax', name='dense_output')
    decoder_outputs = decoder_dense(decoder_output_2)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_state)
    decoder_model.load_weights(
        configs['model-weight-a'][exp_name], by_name=True)
    # generate the data needed for inference
    #data_gen_eval = dp.data_generator(exp_name, scenario, start_index,train_eval=False)
    #count = 0
    y_window = configs['y-window']
    predicted_sequences = []
    true_sequences = []
    print(f'>>> Model successfuly loaded... starting inference <<<')
    # while count < steps_test:
    #     count += 1
    #     predicted = []
    #     encoder_data, decoder_targets = next(data_gen_eval)
    #     #print(f'Encoder data shape: {encoder_data.shape}')
    #     #print(f'decoder target: {decoder_targets.shape}')
    #     true_sequences.append(decoder_targets.argmax(2))
    #    for data in encoder_data:
            # generate the start of sequence character
            #print(f'Encoder data shape: {data.shape}')
    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype=int)
    target_seq[0, 0, 0] = 1
    data = data.reshape(1, data.shape[0], data.shape[1])
    states_value = encoder_model.predict(data)
    stop_condition = False
    decoded_sequence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_digit = np.argmax(output_tokens[0, -1, :])
        # check for exit condition
        if sampled_digit == end_of_seq or len(decoded_sequence) >= y_window:
            stop_condition = True
        else:
            decoded_sequence.append(sampled_digit)
        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype=int)
        target_seq[0, 0, sampled_digit] = 1
        # update states
        states_value = [h, c]
        #     predicted.append(decoded_sequence)
        # predicted_sequences.append(predicted)
    return (decoded_sequence)


def load_model(exp_name):
    return model_attention(exp_name)
