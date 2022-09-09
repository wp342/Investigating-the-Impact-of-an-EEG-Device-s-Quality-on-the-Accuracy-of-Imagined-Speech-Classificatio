import numpy as np
from general_utils import divide_task_data


# split the data into each binary class then run the pearson correlation on each class
def pearson_feature_class_wise_reshaping(task, labels, thinking):
    # reshape the data back into the 81 features of each epoch
    task_dict = {'0data': [], '1data': [], 'labels': []}
    if labels[0].dtype.name == 'int32':
        for i in range(0, len(labels)):
            if labels[i] == 0:
                task_dict['0data'].append(thinking[i])
            elif labels[i] == 1:
                task_dict['1data'].append(thinking[i])
        task_dict['labels'] = labels
        if len(task_dict['1data']) != np.count_nonzero(labels == 1):
            raise Exception('Incorrect class splitting')

    else:
        for i in range(0, len(labels)):
            if labels[i] in task[0]:
                task_dict['0data'].append(thinking[i])
                task_dict['labels'].append(0)
            elif labels[i] in task[1]:
                task_dict['1data'].append(thinking[i])
                task_dict['labels'].append(1)

    task_dict['0data'] = np.array(task_dict['0data'])
    task_dict['1data'] = np.array(task_dict['1data'])

    pearson_data_average, pearson_data_index, class_reshape_list = correlate_feature_and_reshape(task_dict)

    while len(pearson_data_average['0data']) != len(pearson_data_average['1data']):
        if len(pearson_data_average['0data']) > len(pearson_data_average['1data']):
            min_index = np.where(pearson_data_average['0data'] == np.amin(pearson_data_average['0data']))[0]
            pearson_data_average['0data'] = np.delete(pearson_data_average['0data'], min_index)
            pearson_data_index['0data'] = np.delete(pearson_data_index['0data'], min_index)
        elif len(pearson_data_average['0data']) < len(pearson_data_average['1data']):
            min_index = np.where(pearson_data_average['1data'] == np.amin(pearson_data_average['1data']))[0]
            pearson_data_average['1data'] = np.delete(pearson_data_average['1data'], min_index)
            pearson_data_index['1data'] = np.delete(pearson_data_index['1data'], min_index)

    ml_data = {}
    for data in task_dict:
        if not data.__contains__("data"):
            continue
        new_data = task_dict[data].transpose()[pearson_data_index[data]]
        ml_data[data] = np.array(np.split(new_data, len(class_reshape_list[data]), axis=1))

    # join lists with labels and then join dataset together and shuffle ready for machine learning
    ml_labels = [0] * len(ml_data['0data']) + [1] * len(ml_data['1data'])
    ml_data = np.vstack([ml_data['0data'], ml_data['1data']])
    return ml_labels, ml_data, pearson_data_index


# run the pearson correlation on the entire class as a single dataset
def pearson_feature_task_dataset_reshaping(task, labels, thinking):
    task_dict = divide_task_data(labels, task, thinking)

    pearson_data_average, pearson_data_index, class_reshape_list = correlate_feature_and_reshape(task_dict)

    new_data = task_dict['data'].transpose()[pearson_data_index['data']]
    ml_data = np.array(np.split(new_data, len(class_reshape_list['data']), axis=1))
    ml_labels = task_dict['labels']
    return ml_labels, ml_data


def correlate_feature_and_reshape(task_dict):
    class_reshape_list = {}
    for data in task_dict:
        if not data.__contains__("data"):
            continue
        class_reshape_list[data] = reshape_to_features(task_dict[data])
        task_dict[data] = np.vstack(class_reshape_list[data])
    pearson_data_average, pearson_data_index = run_pearson_correlation(task_dict)

    return pearson_data_average, pearson_data_index, class_reshape_list


def reshape_to_features(raw_data):
    class_reshape_list = []
    for window in raw_data:
        reshape_list = []
        for reshape in window:
            reshape = np.split(reshape, 3)
            for i in range(len(reshape)):
                reshape[i] = reshape[i].reshape(62, 27)
            reshape = np.hstack(reshape)
            reshape_list.append(reshape)
        all_epoch_array = np.vstack(reshape_list)
        class_reshape_list.append(all_epoch_array)

    return class_reshape_list


def new_run_pearson_correlation(data):
    # calculate the pearson correlation of the features to find the most correlated features

    data = np.hstack(np.nan_to_num(data))
    pearson_data = np.corrcoef(data)
    pearson_data = abs(pearson_data)
    pearson_data_average = pearson_data.mean(axis=1)
    pearson_index = np.flip(np.argsort(pearson_data_average))
    if pearson_data_average[pearson_index[-1]] > pearson_data_average[pearson_index[0]]:
        raise "Pearson list in wrong order"
    return pearson_index


def run_pearson_correlation(task_dict):
    # calculate the pearson correlation of the features to find the most correlated features

    pearson_data_average = {}
    pearson_data_index = {}
    for data in task_dict:
        if not data.__contains__("data"):
            continue
        task_dict[data] = np.nan_to_num(task_dict[data])
        pearson_data = np.corrcoef(task_dict[data].transpose())
        pearson_data = abs(pearson_data)
        pearson_data_average[data] = pearson_data.mean(axis=0)
        pearson_data_average_average = np.mean(pearson_data_average[data])

        # extract the features that have a mean correlation higher than the average correlation
        for i in range(len(pearson_data_average[data])):
            if pearson_data_average[data][i] < pearson_data_average_average:
                pearson_data_average[data][i] = 0
        pearson_data_index[data] = pearson_data_average[data].nonzero()[0]
        pearson_data_average[data] = pearson_data_average[data][pearson_data_average[data] != 0.0]
    return pearson_data_average, pearson_data_index

##TODO Possible investigate into reducing the number of channels to those that are most correlated
# def correlated_channel_extraction(ml_data):
#     for data in ml_data:
#         num_of_splits = (ml_data[data].shape[2])/62  # 62 as number of channels
#         split_epoch_list = []
#         for i in range(len(ml_data[data])):
#             split_epoch = np.array(np.split(ml_data[data][i], num_of_splits, axis=1))
#             split_epoch_list.append(split_epoch)
#         ml_data[data] = np.vstack(split_epoch_list)
#         ml_data[data] = np.vstack(ml_data[data])
#
#     pearson_data_average, pearson_data_index = run_pearson_correlation(ml_data)
#
#     print(ml_data)
