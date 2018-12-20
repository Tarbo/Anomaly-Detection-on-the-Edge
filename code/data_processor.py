import os
import re
import dask.dataframe as dd
import pandas as pd
import glob
import numpy as np
import json
import h5py
from keras.utils import to_categorical
from gensim.models.word2vec import Word2Vec, LineSentence


def load_config():
    """Load the configuration file"""
    path = 'configs.json'
    if os.path.isfile(path):
        return json.load(open(path, mode='r'))
    else:
        raise FileNotFoundError()


def strip_pattern(element):
    """Strip the digits and some symbols that make the vocabulary non-generic"""
    pattern = re.compile(r'v/\d*|\-\d*|/\d*|_\d\w+', re.IGNORECASE)
    return pattern.sub('', element)


def read_csv(exp_name, scenario):
    """Read the CVS files for each folder"""
    config = load_config()
    path = os.path.join(config[exp_name][scenario], '*.csv')
    all_files = glob.glob(path, recursive=True)
    #num = len(all_files)
    # read the class and event columns
    df = dd.read_csv(all_files, usecols=[3, 4], names=['class', 'k_event'])
    df = df['class'].str.cat(
        df['k_event'], sep='_').apply(strip_pattern, meta=pd.Series(dtype=str)).compute()
    pattern = re.compile(r'\d+', re.I)
    data = [variable.lower() for variable in df.tolist()
            if not pattern.match(variable)]
    return data


def train_embedding_weight(data, exp_name):
    """Converts the categorical values to integer encoded sequence"""
    config = load_config()
    # load the model parameters
    embed_size = config['embed-model']['embed_size']
    context_win = config['embed-model']['context_window']
    min_count = config['embed-model']['min_count']
    workers = config['embed-model']['workers']
    epochs = config['embed-model']['epochs']
    path = exp_name + '.txt'
    data = read_csv(data, exp_name)  # returns a a list of tokens
    if os.path.exists(path):
        sentences = LineSentence(path)
    else:
        with open(path, "w") as datafile:
            for item in data:
                datafile.write(f'{item}\n')
        sentences = LineSentence(path)
    # cycle for one iteration to get the number of features or vocabs
    model = Word2Vec(
        sentences, size=embed_size, window=context_win, min_count=min_count, sample=0, seed=1, workers=workers, iter=1)
    # num_words = len(model.wv.vocab)
    # del model
    # print(
    #     f'Starting Training for:{exp_name} folder with {num_words} vocabularies')
    # model = Word2Vec(sentences, size=num_words, window=context_win,
    #                  min_count=min_count, sample=0, seed=1000, workers=workers, iter=epochs)
    # weight = model.wv.syn0
    # weight_file = config['embed-weights'][exp_name]
    # np.save(open(weight_file, 'wb'), weight)
    vocab_file = config['vocabs'][exp_name]  # retrive the JSON file name
    vocabs = dict([(key, value.index + 1)
                   for key, value in model.wv.vocab.items()])
    with open(vocab_file, 'w') as jsonfile:
        json.dump(vocabs, jsonfile, indent=4)
    print(f'>>> Vocabulary of: {exp_name}\t saved in: {vocab_file} <<<')
    # print(
    #     f'Weights saved in {weight_file}\nVocabulary of {exp_name} saved in {vocab_file}')
    del model
    return


def integer_encoder(exp_name, scenario):
    """This function encodes the string tokens into integers generated from the embedding layer training"""
    config = load_config()
    try:
        vocabs = json.load(open(config['vocabs'][exp_name], 'r'))
        data = read_csv(exp_name, scenario)
        data = [vocabs[word] for word in data if word in vocabs.keys()]
        return data
    except FileNotFoundError:
        print('Ooops! Looks like the `vocabs` file is not created yet. Run the embedding layer training function to generate this')
        return


def transform_data(exp_name, scenario):
    """Transform the data into features and `labels` for saving in the h5py database"""
    config = load_config()
    data = integer_encoder(exp_name, scenario)
    num_rows = len(data)
    encoder_input = []
    decoder_input = []
    decoder_target = []
    x_window_size = config['x-window']
    y_window_size = config['y-window']
    batch_size = config['batch-size']
    index = 0
    vocabs = json.load(open(config['vocabs'][exp_name], 'r'))
    max_id = max(vocabs.values())
    while (index + x_window_size + y_window_size) <= num_rows:
        encoder_window_data = data[index:(index + x_window_size)]
        decoder_input_window = data[(
            index + x_window_size): (index + x_window_size + y_window_size)]
        encoder_window_data = np.array(
            encoder_window_data).flatten().tolist()
        decoder_input_window = np.array(
            decoder_input_window).flatten().tolist()
        decoder_input_window = np.pad(decoder_input_window, (1, 1),
                                      'constant', constant_values=(0, max_id + 1)).tolist()
        #decoder_input_window = [0] + decoder_input_window + [314]
        decoder_target_window = decoder_input_window[1:] + [
            decoder_input_window[0]]
        encoder_input.append(encoder_window_data)
        decoder_input.append(decoder_input_window)
        decoder_target.append(decoder_target_window)
        index += 1
        # Use this index for attentiondecoder
        # index += x_window_size
        if index % batch_size == 0:
            # Convert from list to 3 dimensional numpy array [batches, timesteps, feature_dim]
            # If there is no embedding layer
            encoder_input = to_categorical(
                encoder_input, num_classes=len(vocabs.values()) + 1)
            decoder_input = to_categorical(
                decoder_input, num_classes=len(vocabs.values()) + 2)
            decoder_target = to_categorical(
                decoder_target, num_classes=len(vocabs.values()) + 2)
            # encoder_input = np.array(encoder_input)
            # decoder_input = np.array(decoder_input)
            # decoder_target = np.array(decoder_target)
            encoder_data_3dim = np.reshape(
                encoder_input, (encoder_input.shape[0], x_window_size, len(vocabs.values()) + 1))
            decoder_input_3dim = np.reshape(
                decoder_input, (decoder_input.shape[0], y_window_size + 2, len(vocabs.values()) + 2))
            decoder_target_3dim = np.reshape(
                decoder_target, (decoder_input.shape[0], y_window_size + 2, len(vocabs.values()) + 2))
            encoder_input = []
            decoder_input = []
            decoder_target = []
            yield (encoder_data_3dim, decoder_input_3dim, decoder_target_3dim)


def save_to_database(exp_name, scenario):
    """Save the transformed data in h5py database"""
    config = load_config()
    # this yields the tranformed data
    data = transform_data(exp_name, scenario)
    # store the `data` in the corresponding database as it is generated.
    chunk_count = 0
    if scenario == 'train':
        xy_db = config['xy-train'][exp_name]
    elif scenario == 'clean':
        xy_db = config['xy-clean'][exp_name]
    else:
        xy_db = config['xy-anomaly'][exp_name]
    encoder_input_db = config['encoder-name-db']
    decoder_input_db = config['decoder-input-name-db']
    decoder_target_db = config['decoder-target-name-db']
    with h5py.File(xy_db, 'w') as db:
        encoder_input, decoder_input, decoder_target = next(data)
        # initialize the hdfs file wit first chunk
        row_count_x = encoder_input.shape[0]  # shape is 3D
        encoder_db = db.create_dataset(encoder_input_db, shape=encoder_input.shape, maxshape=(
            None, None, encoder_input.shape[2]), chunks=True)
        encoder_db[:] = encoder_input
        row_count_y = decoder_input.shape[0]
        decoder_input_database = db.create_dataset(decoder_input_db, shape=decoder_input.shape, maxshape=(
            None, None, decoder_input.shape[2]), chunks=True)
        decoder_input_database[:] = decoder_input
        row_count_y2 = decoder_target.shape[0]
        decoder_target_database = db.create_dataset(
            decoder_target_db, shape=decoder_target.shape, maxshape=(None, None, decoder_target.shape[2]))
        decoder_target_database[:] = decoder_target
        for encoder_input_batch, decoder_input_batch, decoder_target_batch in data:
            # append the encoder_input and decoder_input into the db
            encoder_db.resize(
                row_count_x + encoder_input_batch.shape[0], axis=0)
            encoder_db[row_count_x:] = encoder_input_batch
            row_count_x += encoder_input_batch.shape[0]
            decoder_input_database.resize(
                row_count_y + decoder_input_batch.shape[0], axis=0)
            decoder_input_database[row_count_y:] = decoder_input_batch
            row_count_y += decoder_input_batch.shape[0]
            decoder_target_database.resize(
                row_count_y2 + decoder_target_batch.shape[0], axis=0)
            decoder_target_database[row_count_y2:] = decoder_target_batch
            row_count_y2 += decoder_target_batch.shape[0]
            chunk_count += 1
            print(f'>>> Batch: {chunk_count} <<<', end='\r')
        print(
            f'>>> Transformed `encoder, decoder input and decoder target` data stored in {xy_db} database <<<')
    return


def data_generator(exp_name, scenario, start_index=0, train_eval=True):
    """Load data from the database"""
    config = load_config()
    #weights = np.load(open(config['embed-weights'][exp_name], 'rb'))
    if scenario == 'train':
        xy_db = config['xy-train'][exp_name]
    elif scenario == 'clean':
        xy_db = config['xy-clean'][exp_name]
    else:
        xy_db = config['xy-anomaly'][exp_name]
    encoder_name_db = config['encoder-name-db']
    decoder_input_db = config['decoder-input-name-db']
    decoder_target_db = config['decoder-target-name-db']
    batch_size = config['batch-size']
    with h5py.File(xy_db, 'r') as db:
        index = start_index
        while True:
            encoder_input = db[encoder_name_db][index:index + batch_size]
            decoder_input = db[decoder_input_db][index:index + batch_size]
            decoder_target = db[decoder_target_db][index:index + batch_size]
            # decoder_input = decoder_input.reshape((batch_size, timesteps, decoder_input.shape[1]))
            # decoder_input = np.reshape(decoder_input,(batch_size,x_window_size, decoder_input.shape[1]))
            index += batch_size
            if train_eval:
                yield ([encoder_input, decoder_input], decoder_target)
            else:
                yield(encoder_input, decoder_target)
