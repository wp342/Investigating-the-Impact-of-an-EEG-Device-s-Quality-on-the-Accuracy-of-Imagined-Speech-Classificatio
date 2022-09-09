from FEIS_utils import raw_folders, tasks
import numpy as np
import os.path as op
import csv

#tasks = {#'binary_bilabial': [['/tiy/'], ['/piy/']], # 'pat', 'pot'   , '/iy/', '/n/'
   # 'binary_backness': [['/uw/'], ['/iy/']]}

def data_collation(tasks):
    model_dir_ ="drive/MyDrive/Kaggle_Runner/FEIS FINAL ATTEMPT CSP 9/"
    ML_tests = ['SVM_results_evaluation', 'Random_forest_results_evaluation', 'KNN_results_evaluation']

    for ML in ML_tests:
        for task in tasks:
            task_total_avg_list = {}
            for folder in raw_folders:
                print(folder)
                model_dir = model_dir_ + folder + '/' + task + '/'
                numpy_features = np.genfromtxt(op.join(model_dir, ML + '_normalised_SNN_train_3_fold_NDR.csv'), delimiter=",", dtype=float)
                numpy_features = numpy_features[1:, :]
                numpy_features = numpy_features[~np.isnan(numpy_features).any(axis=1), :]
                for i in range(10):
                    try:
                        task_total_avg_list[str(numpy_features[i][0])].append(numpy_features[i][-1])
                    except:
                        task_total_avg_list[str(numpy_features[i][0])] = []
                        task_total_avg_list[str(numpy_features[i][0])].append(numpy_features[i][-1])

            if task == 'binary_cv':
                for epochs_run in task_total_avg_list:
                    csv_titles = [ML] + raw_folders + ['Average']
                    file = open(model_dir_ + epochs_run + '_epoch_total_' + ML + '_normalised_CSP_SNN_train_3_fold_NDR.csv','w')
                    writer = csv.writer(file)
                    writer.writerow(csv_titles)

            for epochs_run in task_total_avg_list:
                file = open(model_dir_ + epochs_run + '_epoch_total_' + ML + '_normalised_CSP_SNN_train_3_fold_NDR.csv', 'a')
                csv_row = [task] + task_total_avg_list[epochs_run] + [str(np.mean(task_total_avg_list[epochs_run]))]
                writer = csv.writer(file)
                writer.writerow(csv_row)

data_collation(tasks)

