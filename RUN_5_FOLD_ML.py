import numpy as np
import csv
from general_utils import divide_task_data
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from Pearson_Data_Reshaping import new_run_pearson_correlation
from CSP_formatting import perform_csp_channel_selection
from three_fold_CSP_SNN_Data_Analyser import data_analyser
import sklearn.decomposition as sklearn
import os
import os.path as op

def run_ML_algorithms(ML, dataset, folders, labels_dir, n_estimators, experiment_dir, exp_filename, results_filename_description, extra_file, i):
    for n_estimator in n_estimators:
        print("TESTING " + str(n_estimator) + " estimations")
        csv_titles = [str(n_estimator)] + folders + ['Average']

        output_file_dir = dataset + ' - ' + ML + ' results/' + extra_file
        if not op.exists(output_file_dir):
            os.mkdir(output_file_dir)
        current_dir_contents = sorted(os.listdir(output_file_dir))
        output_filename = str(n_estimator) + results_filename_description + '_' + str(i)
        if output_filename + ".csv" in current_dir_contents:
            continue
        file = open(output_file_dir + output_filename + ".csv", 'w')
        std_file = open(output_file_dir + output_filename + "_std.csv", 'w')

        writer = csv.writer(file)
        std_writer = csv.writer(std_file)

        writer.writerow(csv_titles)
        csv_titles.remove('Average')
        std_writer.writerow(csv_titles)

        for TASK in tasks:
            print(TASK)
            task = tasks[TASK]
            task_avg = []
            task_std = []
            print("Gathering DATA.......")
            for folder in folders:
                print(folder)
                thinking = np.load(experiment_dir + folder + '/' + exp_filename)  # f + folder +

                labels = np.load(labels_dir + folder + '/labels.npy')

                task_dict = divide_task_data(labels, task, thinking)
                RF_data = task_dict['data']
                RF_labels = task_dict['labels']

                training_datset = RF_data
                if 'split' not in exp_filename and 'pearson' not in results_filename_description and 'CSP' not in results_filename_description and 'ica' not in extra_file:
                    print("Data Gathered of shape " + str(RF_data.shape) + "........Formatting DATA")
                    nsamples, nx, ny = RF_data.shape
                    training_datset = RF_data.reshape((nsamples, nx * ny))
                    print("data reshaped to " + str(training_datset.shape))

                print("data Formatted....splitting test and train......")
                ones = np.where(RF_labels == 1)[0]
                zeros = np.where(RF_labels == 0)[0]
                if abs(len(ones) - len(zeros)) != 0:
                    ones = ones[0:min(len(zeros), len(ones))]
                    zeros = zeros[0:min(len(zeros), len(ones))]
                accuracy_scores = []
                kf = KFold(n_splits=5, shuffle=False)
                kf.get_n_splits(ones)
                for train_index, test_index in kf.split(ones):
                    train_ones = ones[train_index]
                    train_zeros = zeros[train_index]
                    test_ones = ones[test_index]
                    test_zeros = zeros[test_index]
                    comb_train_index = np.concatenate((train_ones, train_zeros))
                    comb_test_index = np.concatenate((test_ones, test_zeros))
                    np.random.shuffle(comb_test_index)
                    np.random.shuffle(comb_train_index)

                    X_train = training_datset[comb_train_index]
                    X_test = training_datset[comb_test_index]
                    y_train = RF_labels[comb_train_index]
                    y_test = RF_labels[comb_test_index]

                    if np.sum(y_train) != len(y_train) / 2:
                        raise "Incorrectly split!!"
                    if np.sum(y_test) != len(y_test) / 2:
                        raise "Incorrectly split!!"

                    if 'split' in exp_filename:
                        train_shape = X_train.shape
                        test_shape = X_test.shape
                        X_train = X_train.reshape(train_shape[0] * train_shape[1], train_shape[2], train_shape[3])
                        X_test = X_test.reshape(test_shape[0] * test_shape[1], test_shape[2], test_shape[3])
                        label_lists = [y_train, y_test]
                        for index, rep_labels in enumerate(label_lists):
                            labels_list = []
                            for rep_label in rep_labels:
                                labels_list.append(np.repeat(rep_label, 5))
                            label_lists[index] = np.array(labels_list).flatten()

                        y_train = label_lists[0]
                        y_test = label_lists[1]
                        train_perm = np.random.permutation(len(X_train))
                        test_perm = np.random.permutation(len(X_test))
                        X_train = np.array(X_train[train_perm])
                        X_test = np.array(X_test[test_perm])
                        y_train = y_train[train_perm]
                        y_test = y_test[test_perm]

                        print("Data Gathered of shape " + str(X_train.shape) + "........Formatting DATA")
                        nsamples, nx, ny = X_train.shape
                        X_train = X_train.reshape((nsamples, nx * ny))
                        print("data reshaped to " + str(X_train.shape))

                        print("Data Gathered of shape " + str(X_test.shape) + "........Formatting DATA")
                        nsamples, nx, ny = X_test.shape
                        X_test = X_test.reshape((nsamples, nx * ny))
                        print("data reshaped to " + str(X_test.shape))

                    if 'pearson' in results_filename_description or 'CSP' in results_filename_description:
                        # pearson correlation code:
                        if 'pearson' in results_filename_description:
                            print('Running Pearson Correlation for ' + str(i) + ' features')
                            pearson_index = new_run_pearson_correlation(X_train)
                            pearson_index = pearson_index[0: i]
                        else:
                            combined_index, zero_index, _ = perform_csp_channel_selection(X_train, y_train, TASK, i)
                            if 'double' in results_filename_description:
                                pearson_index = combined_index
                            else:
                                pearson_index = zero_index
                        X_train = X_train[:, pearson_index, :]
                        X_test = X_test[:, pearson_index, :]

                        # END of pearson code
                    if 'ica' in extra_file:
                        transformer = sklearn.FastICA(n_components=i)
                        trials = []
                        for trial in X_train:
                            trial = transformer.fit_transform(trial.transpose())
                            trials.append(trial.transpose())
                        X_train = np.array(trials)
                        trials = []
                        for trial in X_test:
                            trial = transformer.fit_transform(trial.transpose())
                            trials.append(trial.transpose())
                        X_test = np.array(trials)

                    if 'pearson' in results_filename_description or 'CSP' in results_filename_description or 'ica'in extra_file:
                        nsamples, nx, ny = X_train.shape
                        training_dataset = X_train.reshape((nsamples, nx * ny))
                        print("data reshaped to " + str(training_dataset.shape))
                        X_train = training_dataset
                        nsamples, nx, ny = X_test.shape
                        testing_dataset = X_test.reshape((nsamples, nx * ny))
                        print("data reshaped to " + str(testing_dataset.shape))
                        X_test = testing_dataset

                    if i == 4:
                        print()


                    X_train = np.nan_to_num(X_train)
                    X_test = np.nan_to_num(X_test)

                    print("Training and Testing Classifier............")

                    if ML == 'Random Forest':

                        RFclassifier = RandomForestClassifier(n_estimators=int(n_estimator), random_state=100)  # , max_depth=5) # , verbose = True)
                        RFclassifier.fit(X_train, y_train)

                        score = RFclassifier.score(X_test, y_test)
                    elif ML == 'SVM':
                        SVMclassifier = SVC(kernel=n_estimator, probability=False, degree=2)  # , verbose = True)
                        SVMclassifier.fit(X_train, y_train)

                        score = SVMclassifier.score(X_test, y_test)
                    elif ML == 'KNN':
                        KNNclassifier = KNeighborsClassifier(n_neighbors=3, metric=n_estimator)  # , verbose = True)
                        KNNclassifier.fit(X_train, y_train)
                        score = KNNclassifier.score(X_test, y_test)

                    accuracy_scores.append(score)

                    accuracy_scores.append(score)
                    print("Classification Accuracy is: " + str(score))
                    print()
                    print()

                accuracy_scores = np.array(accuracy_scores)
                average_score = accuracy_scores.mean()
                std = np.std(accuracy_scores)
                print("AVERAGE SCORE IS : " + str(average_score))
                task_avg.append(average_score)
                task_std.append(std)
            task_avg_mean = np.mean(task_avg)
            csv_row = [TASK] + list(task_avg) + [task_avg_mean]
            csv_row_std = [TASK] + list(task_std)
            writer.writerow(csv_row)
            std_writer.writerow(csv_row_std)
            print()
            print()

datasets = [ 'FEIS', 'karaone']
ML = ['KNN', 'SVM', 'Random Forest',]
for ML_dir in ML:
    print('##############################################################################################################')
    print('BEGINNING ' + ML_dir + ' TESTING.........')
    if ML_dir == 'Random Forest':
        n_estimators = [10, 50, 100, 500]
        ML_filename = 'RAND_FOREST'
    elif ML_dir == 'SVM':
        n_estimators = ['linear', 'poly', 'rbf', 'sigmoid']
        ML_filename = ML_dir
    elif ML_dir == 'KNN':
        n_estimators = ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski']
        ML_filename = ML_dir
    else:
        raise "No Machine Learning Specified!"

    for dataset in datasets:
        if dataset == 'FEIS':
            from FEIS_utils import tasks, raw_folders, MFCC_experiments_dir, linear_data_dir, experiments_dir, channels
        elif dataset == 'karaone':
            from kara_one_utils import tasks, raw_folders, MFCC_experiments_dir, linear_data_dir, experiments_dir, channels

        i = ''

        print('Linear feature Testing.......')
        run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, linear_data_dir, 'ica_linear_data.npy', '_' + ML_filename + '_linear_ica_5_fold', '', i)
        run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, linear_data_dir, 'linear_data.npy', '_' + ML_filename + '_linear_5_fold', '', i)
        if dataset == 'FEIS':
            run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, linear_data_dir, 'split_ica_linear_data.npy', '_' + ML_filename + '_split_linear_ica_5_fold', '', i)
            run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, linear_data_dir, 'split_linear_data.npy', '_' + ML_filename + '_split_linear_5_fold', '', i)

        print('Beginning MFCC Testing')
        run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, MFCC_experiments_dir, 'ica_MFCC_data.npy', '_' + ML_filename + '_MFCC_ica_5_fold', '', i)
        run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, MFCC_experiments_dir, 'MFCC_data.npy', '_' + ML_filename + '_MFCC_5_fold', '', i)
        if dataset == 'FEIS':
            run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, MFCC_experiments_dir, 'split_ica_MFCC_data.npy', '_' + ML_filename + '_split_MFCC_ica_5_fold', '', i)
            run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, n_estimators, MFCC_experiments_dir, 'split_MFCC_data.npy', '_' + ML_filename + '_split_MFCC_5_fold', '', i)

        model_dir_ = './' + dataset + ' - ' + ML_dir + ' results/'

        feature_types = ['_linear_', 'MFCC']
        for feature in feature_types:
            highest_average, _ = data_analyser(model_dir_, feature)


            split_filename = highest_average.split('_')
            split = ''
            feature_type = ''
            ica = ''
            pearson_max_value = 0
            if 'split' in highest_average:
                split = 'split_'
            if 'MFCC' in highest_average:
                feature_type = 'MFCC'
                feature_data = MFCC_experiments_dir
                pearson_max_value = channels+1
            else:
                feature_type = 'linear'
                feature_data = linear_data_dir
                pearson_max_value = 82
            if 'ica' in highest_average:
                ica = 'ica_'
            for i in range(3, pearson_max_value):
                run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, [split_filename[0]], feature_data,
                                  split + ica + feature_type + '_data.npy', '_' + ML_filename + '_' + split + ica +
                                  feature_type + '_5_fold_pearson', 'Pearson Correlation/', i)
                run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, [split_filename[0]], feature_data,
                                  split + ica + feature_type + '_data.npy', '_' + ML_filename + '_' + split + ica +
                                  feature_type + '_5_fold_CSP', 'CSP/', i)
                #run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, [split_filename[0]], feature_data,
                 #                 split + ica + feature_type + '_data.npy', '_' + ML_filename + '_' + split + ica +
                  #                feature_type + '_5_fold', 'ica/', i)

            for i in range(3, int(pearson_max_value/2)+1):
                run_ML_algorithms(ML_dir, dataset, raw_folders, experiments_dir, [split_filename[0]], feature_data,
                                  split + ica + feature_type + '_data.npy', '_' + ML_filename + '_' + split + ica +
                                  feature_type + '_5_fold_double_CSP', 'CSP/', i)


            data_analyser(model_dir_)
            data_analyser(model_dir_ + 'Pearson Correlation/')
            data_analyser(model_dir_ + 'CSP/')
            print()
            print()




