import argparse
import numpy as np
from NN_Data_Reshaping import generate_epoch_pairs, normalise_data
from FEIS_utils import tasks, experiments_dir
from general_utils import divide_task_data, band_pass_and_notch_filter_data
from CSP_formatting import perform_csp_channel_selection

epochs = [1000, 0]
batch_size = [100, 50]
#epochs = 500

parser = argparse.ArgumentParser(description='Siamese NN Trainer')
parser.add_argument('-t','--task', type=str, help='binary task being learnt', required=True)
parser.add_argument('-f','--folder', type=str, help='participant folder', required=True)
parser.add_argument('-fn','--fold_number', type=str, help='The data fold to work in', required=True)
parser.add_argument('-n','--nothing', type=str, help='alleviate carriage return issue', required=False)
args = vars(parser.parse_args())

task = args['task']

folder = args['folder']

fold_number = args['fold_number']



def run_SNN_Training(folder, fold_number, task, epochs):

    import tensorflow as tf
    from Train_New_SNN import train

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print(folder)
    print(task)

    SNN_data_dir = 'drive/MyDrive/Kaggle_Runner/FEIS_SNN_data/' #'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/kara_one_SNN_data/' #'
    SNN_train_index = SNN_data_dir + folder + '/' + task +'_CSP_5_fold/normalised_train_index_' + fold_number + '.npy'#'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/kara_one_SNN_data/' + folder + '/'+ task +'_CSP/train_labels.npy'

    thinking = np.load(experiments_dir + folder + '/thinking.npy')
    thinking = np.nan_to_num(thinking)
    labels = np.load(experiments_dir + folder + '/labels.npy')
    train_index = np.load(SNN_train_index)
    task_dict = divide_task_data(labels, tasks[task], thinking)
    task_dict['data'] = task_dict['data'][train_index]
    task_dict['labels'] = task_dict['labels'][train_index]
    if np.sum(task_dict['labels']) != len(task_dict['labels'])/2:
        raise "Unbalanced Data!"

    train_data = []
    task_dict['data'] = band_pass_and_notch_filter_data(task_dict['data'], 256)
    combined_indicies, zero_indices, one_indices = perform_csp_channel_selection(task_dict['data'], task_dict['labels'], task, 9)
    for i in range(len(task_dict['data'])):
        train_data.append(task_dict['data'][i][zero_indices].transpose())
        #train_data.append(task_dict['data'][i].transpose())
    #task_dict['train_data'] = np.array(train_data)
    task_dict['train_data'] = np.array(train_data)

    task_dict['train_labels'] = task_dict['labels']
   # task_dict['train_data'] = normalise_data(task_dict['train_data'])

    features_dir = SNN_data_dir + folder + '/' + task + '_CSP_5_fold/'
    print('Creating data pairs ..... ')
    print(task_dict['train_data'].shape)
    train_data = generate_epoch_pairs(task_dict, 0, features_dir)
    print('assigning batch size ...... ')

    total, inp1, inp2, inp3 = train_data[0].shape
    print('Training: ' + folder + ' ' + task + ' ' + str(fold_number))
    train(inp1=inp1, inp2=inp2, inp3=inp3, validation_interval=20, data=train_data, all_epochs=epochs, task=task, folder=folder, batch_size=batch_size, save_per_epoch=100, model_fold=fold_number)




run_SNN_Training(folder, fold_number, task, epochs)