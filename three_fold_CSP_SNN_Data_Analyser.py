import numpy as np
import os.path as op
import os

def data_analyser(model_dir_, filter=None):
    ML_tests = ['SVM_results_evaluation', 'Random_forest_results_evaluation', 'KNN_results_evaluation']

    total_highest_average = 0
    total_highest_average_filename = 'nought'
    total_highest_average_list = []

    total_lowest_std = 10000
    total_lowest_std_filename = 'nought'
    total_lowest_std_list = []

    classifier_highest_average = [0, 0, 0]
    classifier_highest_average_filename = ['', '', '']
    classifier_highest_average_list = [[], [], []]

    test_data_files = sorted(os.listdir(model_dir_))


    task_highest_avg_value = {}
    test_data_file_removers = [s for s in test_data_files if ".csv" not in s]
    for remove in test_data_file_removers:
        test_data_files.remove(remove)

    if filter:
        test_data_files = [x for x in test_data_files if filter in x]
        test_data_files = [x for x in test_data_files if 'split' not in x]

    for test_data_file in test_data_files:
        # if '_DR' not in test_data_file:
        #     continue
        numpy_features = np.genfromtxt(op.join(model_dir_, test_data_file), delimiter=",", dtype=float)
        try:
            tasks = np.genfromtxt(op.join(model_dir_, test_data_file), delimiter=",", dtype=str)[1:, 0]
        except:
            continue
        numpy_features = numpy_features[1:, 1:]
        average_list = numpy_features[:, -1]
        for index, task in enumerate(tasks):
            if task not in task_highest_avg_value:
                task_highest_avg_value[task] = [str(0), '0']
            if float(task_highest_avg_value[task][0]) < average_list[index]:
                task_highest_avg_value[task] = [str(average_list[index]), test_data_file]

        current_average = np.mean(average_list)
        average_std = np.std(average_list)
        for index, ML_test in enumerate(ML_tests):
            if ML_test in test_data_file:
                if current_average > classifier_highest_average[index]:
                    classifier_highest_average_filename[index] = test_data_file
                    classifier_highest_average[index] = current_average
                    classifier_highest_average_list[index] = average_list
        if average_std < total_lowest_std:
            if current_average > 0.53:
                total_lowest_std = average_std
                total_lowest_std_list = average_list
                total_lowest_std_filename = test_data_file
        if current_average > total_highest_average:
            total_highest_average_filename = test_data_file
            total_highest_average = current_average
            total_highest_average_list = average_list


    print("The highest average was " + str(total_highest_average) + " filename: " + total_highest_average_filename)
    print("     With the following average list : " + str(total_highest_average_list))

    print()
    print("The lowest std was " + str(total_lowest_std) + " filename: " + total_lowest_std_filename)
    print("     With the following average list : " + str(total_lowest_std_list))


    print()
    for index, ML_test in enumerate(ML_tests):
        print("The highest " + ML_test + " average was " + str(classifier_highest_average[index]) + " filename: " +
              classifier_highest_average_filename[index])
        print("     With the following average list : " + str(classifier_highest_average_list[index]))

    print()
    for task in task_highest_avg_value:
        print("The largest task average for " + task + " is: " + str(task_highest_avg_value[task][0]))
        print("     This was found in " + task_highest_avg_value[task][1])

    return total_highest_average_filename, total_highest_average


