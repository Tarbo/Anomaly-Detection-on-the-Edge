
def kernel(error_value):
    from scipy.stats import gaussian_kde
    # Perform Kernel Density Estimation on the Model
    init_len = int(len(error_value) * 0.5)
    print(
        f'>>> Initializing with {init_len} trace error values <<<')
    init_val = error_value[:init_len]
    kernel = gaussian_kde(init_val)
    print(
        f'>>> Evaluating with {len(error_value) - init_len} error values <<<')
    #eval_pdf = kernel.pdf(error_value[init_len:])
    return kernel


def get_evaluation_error(exp_name, scenario, model_weight_key):
    from model import load_model
    import data_processor as dp
    import h5py
    import json
    import numpy as np
    # load the configuration file
    config = json.load(open('configs.json'))
    # load our model architecture
    #scenario = 'train'

    if scenario == 'train':
        with h5py.File(config['xy-train'][exp_name], 'r') as db:
            nrows = db[config['encoder-name-db']].shape[0]
        # retrieve number of features for training
        ntrain = int(config['train-test-split'] * nrows)
        data_gen_eval = dp.data_generator(
            exp_name, scenario, start_index=ntrain)
        neval = nrows - ntrain
        steps_eval = neval // config['batch-size']
    else:
        with h5py.File(config['xy-anomaly'][exp_name], 'r') as db:
            nrows = db[config['encoder-name-db']].shape[0]
        ntrain = int(0.5 * nrows)
        data_gen_eval = dp.data_generator(
            exp_name, scenario, start_index=0)
        neval = nrows - ntrain
        steps_eval = neval // config['batch-size']
    # Sample the model with evaluation dataset
    model = load_model(exp_name)
    model.load_weights(config[model_weight_key][exp_name])
    loss_acc = np.empty(shape=(steps_eval, 2))
    for index in np.arange(steps_eval):
        x_data, y_data = next(data_gen_eval)
        hist = model.evaluate(x_data, y_data, verbose=0)
        loss_acc[index] = hist
    return 1 - loss_acc[:, 1]


# # def predict_sequence(exp_name, kernel, load_config, data_generator, scenario):
#     import json
#     import h5py
#     import numpy as np
#     from model import load_model
#     true_values = []
#     start_id = 0
#     configs = load_config()
#     batch_size = configs['batch-size']
#     if scenario == 'clean':
#         database = configs['xy-clean'][exp_name]
#     else:
#         database = configs['xy-anomaly'][exp_name]
#         with h5py.File(database, 'r') as db:
#             nrows = db[configs['encoder-name-db']].shape[0]
#         start_id = int(nrows * 0.5)
#     data_gen_test = data_generator(exp_name, scenario, start_index=start_id)
#     # load the anomaly database
#     with h5py.File(database, 'r') as db:
#         ntest_rows = db[configs['x-name-db']].shape[0]
#     steps_test = ntest_rows // batch_size
#     print(
#         f'>...{exp_name} anomaly testing on {ntest_rows} rows for {steps_test} steps ')

#     # make the predictions
#     model = load_model(exp_name)
#     model.load_weights(configs['model-weight'][exp_name])
#     predictions = model.predict_generator(generator_yield_x(
#         data_gen_test, true_values), steps=steps_test)
#     print(f'>...{exp_name} predictions done for {scenario} model...<')

#     # Store as an array
#     predictions = np.array(predictions)
#     true_values = np.array(true_values)

#     # Compute the error and get the pdf
#     prediction_error = []
#     index = 0
#     while (index + batch_size) <= ntest_rows:
#         error_test = np.mean(np.argmax(predictions[index:(
#             index + batch_size)], axis=1) != np.argmax(true_values[index:(index + batch_size)], axis=1))
#         prediction_error.append(error_test)
#         index += batch_size

#     prediction_pdfs = kernel.pdf(prediction_error)
#     prediction_pdfs_dic = {'pdf-pred': list(prediction_pdfs)}
#     with open(pdf_file, 'w') as jsonfile:
#         json.dump(prediction_pdfs_dic, jsonfile, indent=4)
#     print(
#         f'>...Predicted PDFs for {exp_name} {scenario} dumped in {pdf_file}...<')
#     return

# def predictor(exp_name, scenario):
#     from data_processor import load_config, data_generator
#     kernel_var = kernel(exp_name)
#     mode = ['clean', 'anomaly']
#     for scenario in mode:
#         predict_sequence(exp_name, kernel, load_config,
#                          data_generator, scenario)
#     return


# def generator_yield_x(data_gen, true_values):
#     for x_data, y_data in data_gen:
#         true_values += list(y_data)
#         yield x_data

# def visualizer(exp_name):
#     import json
#     import numpy as np
#     from model import load_model
#     from plot import plot_results
#     configs = json.load(open('configs.json'))
#     # load the JSON PDFs
#     pdf_eval_file = configs['eval-pdf-norm'][exp_name]
#     pdf_eval_data_dic = json.load(open(pdf_eval_file))
#     pdf_pred_file_norm = configs['pred-pdf-norm'][exp_name]
#     pdf_pred_data_norm_dic = json.load(open(pdf_pred_file_norm))
#     pdf_pred_file_anom = configs['pred-pdf-anom'][exp_name]
#     pdf_pred_data_anom_dic = json.load(open(pdf_pred_file_anom))
#     # Retrive the data
#     pdf_eval_data = pdf_eval_data_dic['pdf-values']
#     pdf_pred_data_norm = pdf_pred_data_norm_dic['pdf-pred']
#     pdf_pred_data_anom = pdf_pred_data_anom_dic['pdf-pred']
#     print(
#         f'>>> Eval data size: {len(pdf_eval_data)} <<<\n>>> Normal data size: {len(pdf_pred_data_norm)} <<<\n>>> Anomaly data size: {len(pdf_pred_data_anom)} <<<')

#     # Visualize the PDFs of the normal and anomalous scenarios
#     x_norm = np.array([x for x in pdf_pred_data_norm if x >= 1.5])
#     #x_norm = np.array(pdf_pred_data_norm)
#     print('X Norm: {}'.format(len(x_norm)))
#     x_eval = np.array(pdf_eval_data)
#     print('X Eval: {}'.format(len(x_eval)))
#     #x_anom = np.array([x for x in pdf_pred_data_norm if x <= 5])
#     x_anom = np.array(pdf_pred_data_anom)
#     print('X Anom: {}'.format(len(x_anom)))
#     plot_results(exp_name, x_eval[:150], x_anom[0:150])
