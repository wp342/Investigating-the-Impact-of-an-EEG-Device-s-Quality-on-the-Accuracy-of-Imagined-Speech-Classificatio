import os
import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from Build_Siamese_NN import L1Dist
from Train_New_SNN import custom_contrastive_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from kara_one_utils import tasks, raw_folders, experiments_dir
from general_utils import divide_task_data, band_pass_and_notch_filter_data
from CSP_formatting import perform_csp_channel_selection, down_sample_and_filter
from NN_Data_Reshaping import normalise_data, inst_freq_and_spectral_entropy
import csv
import os.path as op

# folders = ["MM05"]#raw_folders

parser = argparse.ArgumentParser(description='Siamese NN Tester')
parser.add_argument('-t', '--task', type=str, help='binary task being learnt', required=True)
args = vars(parser.parse_args())

task = args['task']

max_epoch = 1001

folders = raw_folders


def get_distances(data, model, labels):
    print('GATHERING Model TEST DATA .....')
    reshaped_data = []
    for data_sample in data:
        data_sample = np.array(data_sample).transpose()
        reshaped_data.append(np.reshape(data_sample, (data_sample.shape[0], data_sample.shape[1], 1)))
    data = reshaped_data
    ML_data = {'data': [], 'label': []}
    task_data = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(data), tf.data.Dataset
                                     .from_tensor_slices(data), tf.data.Dataset.from_tensor_slices(np.ones(len(data)))))
    task_data = task_data.batch(1)
    for index, train in enumerate(task_data):
        train1, train2, label = train
        _, embedding1, _ = model.predict([train1, train2])
        embedding1 = np.reshape(embedding1, 8)
        ML_data['data'].append(list(embedding1))
        ML_data['label'].append(labels[index])
    return np.array(ML_data['data']), np.array(ML_data['label'])


def run_testing(folders, task, max_epoch):
    # for task in tasks:
    for folder in folders:
        RF_folds = {}
        SVM_folds = {}
        KNN_folds = {}
        model_dir = 'SNN_Models/' + folder + '/' + task + '_normalised_CSP_3_fold/'

        if not op.exists(model_dir + ' SNN_model_' + str(100) + '_' + str(0) + '.h5'):
            print(model_dir + ' SNN_model_' + str(100) + '_' + str(0) + '.h5 does not exist')
            continue
        if not op.exists('drive/MyDrive/Kaggle_Runner/SNN_Models/'):
            os.mkdir('drive/MyDrive/Kaggle_Runner/SNN_Models/')
        if not op.exists('drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder):
            os.mkdir('drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder)
        if not op.exists('drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task):
            os.mkdir('drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task)
        if op.exists(
                'drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task + '/' + 'Random_forest_results_evaluation' + '_normalised_SNN_train_3_fold_NDR.csv'):
            print("Skipping Test " + folder)
            continue
        if op.exists(
                'drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task + '/' + 'SVM_results_evaluation' + '_normalised_SNN_train_3_fold_NDR.csv'):
            print("Skipping Test " + folder)
            continue
        if op.exists(
                'drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task + '/' + 'KNN_results_evaluation' + '_normalised_SNN_train_3_fold_NDR.csv'):
            print("Skipping Test " + folder)
            continue

        csv_titles = ['Epoch'] + ['1', '2', '3', '4', '5'] + ['Average']
        RF_file = open(
            'drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task + '/' + 'Random_forest_results_evaluation' + '_normalised_SNN_train_3_fold_NDR.csv',
            'w')
        SVM_file = open(
            'drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task + '/' + 'SVM_results_evaluation' + '_normalised_SNN_train_3_fold_NDR.csv',
            'w')
        KNN_file = open(
            'drive/MyDrive/Kaggle_Runner/SNN_Models/' + folder + '/' + task + '/' + 'KNN_results_evaluation' + '_normalised_SNN_train_3_fold_NDR.csv',
            'w')

        RF_writer = csv.writer(RF_file)
        SVM_writer = csv.writer(SVM_file)
        KNN_writer = csv.writer(KNN_file)

        RF_writer.writerow(csv_titles)
        SVM_writer.writerow(csv_titles)
        KNN_writer.writerow(csv_titles)

        for fold_number in range(5):
            SNN_data_dir = 'drive/MyDrive/Kaggle_Runner/kara_one_SNN_data/'
            SNN_train_index = SNN_data_dir + folder + '/' + task + '_CSP_5_fold/normalised_train_index_' + str(
                fold_number) + '.npy'
            SNN_test_index = SNN_data_dir + folder + '/' + task + '_CSP_5_fold/normalised_test_index_' + str(
                fold_number) + '.npy'
            # 'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/SNN_Models/' + folder + '/' + task + '_CSP/ '

            thinking = np.load(experiments_dir + folder + '/thinking.npy')
            thinking = np.nan_to_num(thinking)
            labels = np.load(experiments_dir + folder + '/labels.npy')
            train_test_index = [np.load(SNN_train_index), np.load(SNN_test_index)]
            train_test_data = []
            train_test_labels = []
            task_dict = divide_task_data(labels, tasks[task], thinking)
            for train_test in train_test_index:
                ML_data = task_dict['data'][train_test]
                ML_labels = task_dict['labels'][train_test]
                if np.sum(ML_labels) != len(ML_labels) / 2:
                    raise "Unbalanced Data!"
                train_data = []
                ML_data = down_sample_and_filter(ML_data)
                #combined_indicies, zero_indices, one_indices = perform_csp_channel_selection(ML_data, ML_labels, task, 9)
                                                                                            # 10)
                for i in range(len(ML_data)):
                    #train_data.append(ML_data[i][zero_indices].transpose())
                    train_data.append(ML_data[i].transpose())

                #train_data = inst_freq_and_spectral_entropy(train_data, 256)
                train_test_data.append(np.array(train_data))
                train_test_labels.append(ML_labels)

            for epoch_num in range(100, max_epoch, 100):
                if epoch_num not in RF_folds:
                    RF_folds[epoch_num] = []
                    SVM_folds[epoch_num] = []
                    KNN_folds[epoch_num] = []

                print('STARTING EPOCH NUMBER ' + str(
                    epoch_num) + ' for task: ' + task + ' and participant: ' + folder + '........')
                try:
                    model = load_model(model_dir + ' SNN_model_' + str(epoch_num) + '_' + str(fold_number) + '.h5',
                                       custom_objects={'L1Dist': L1Dist,
                                                       'custom_contrastive_loss': custom_contrastive_loss},
                                       compile=False)
                except:
                    continue
                model.compile(optimizer="Adam", loss=custom_contrastive_loss)

                ML_train = {}
                ML_train['data'], ML_train['label'] = get_distances(train_test_data[0], model, train_test_labels[0])

                ML_test = {}
                ML_test['data'], ML_test['label'] = get_distances(train_test_data[1], model, train_test_labels[1])

                print('Training Classifiers . .. . ')

                KNNclassifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
                KNNclassifier.fit(ML_train['data'], ML_train['label'])

                RFclassifier = RandomForestClassifier(n_estimators=100, random_state=100)
                RFclassifier.fit(ML_train['data'], ML_train['label'])

                SVMclassifier = SVC(kernel='linear')  # , verbose = True)
                SVMclassifier.fit(ML_train['data'], ML_train['label'])

                print('Testing Classifier . . . . ')
                print(ML_test['label'])
                KNNscore = KNNclassifier.score(ML_test['data'], ML_test['label'])
                KNN_folds[epoch_num].append(KNNscore)
                print("KNN Classification Accuracy is: " + str(KNNscore))

                SVMscore = SVMclassifier.score(ML_test['data'], ML_test['label'])
                SVM_folds[epoch_num].append(SVMscore)
                print("SVM Classification Accuracy is: " + str(SVMscore))

                RANDFORESTscore = RFclassifier.score(ML_test['data'], ML_test['label'])
                RF_folds[epoch_num].append(RANDFORESTscore)
                print("RAND FOREST Classification Accuracy is: " + str(RANDFORESTscore))

        for key in RF_folds:
            RF_average = np.mean(RF_folds[key])
            SVM_average = np.mean(SVM_folds[key])
            KNN_average = np.mean(KNN_folds[key])

            RF_csv_row = [float(key)] + list(RF_folds[key]) + [float(RF_average)]
            SVM_csv_row = [float(key)] + list(SVM_folds[key]) + [float(SVM_average)]
            KNN_csv_row = [float(key)] + list(KNN_folds[key]) + [float(KNN_average)]

            RF_writer.writerow(RF_csv_row)
            SVM_writer.writerow(SVM_csv_row)
            KNN_writer.writerow(KNN_csv_row)


run_testing(folders, task, max_epoch)