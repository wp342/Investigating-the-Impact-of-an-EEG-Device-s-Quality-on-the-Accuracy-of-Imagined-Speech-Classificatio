import numpy as np
from scipy import signal as signal


def divide_task_data(labels, task, thinking):
    task_dict = {'data': [], 'labels': []}

    for i in range(0, len(labels)):
        if labels[i] in task[0]:
            task_dict['data'].append(thinking[i])
            task_dict['labels'].append(0)
        elif labels[i] in task[1]:
            task_dict['data'].append(thinking[i])
            task_dict['labels'].append(1)

    task_dict['data'] = np.array(task_dict['data'])
    task_dict['labels'] = np.array(task_dict['labels'])
    return task_dict


def divide_task_and_class_data(labels, task, thinking):
    task_dict = {'data0': [], 'data1': [], 'labels0': [], 'labels1': []}

    for i in range(0, len(labels)):
        if labels[i] == 0:
            task_dict['data0'].append(thinking[i])
            task_dict['labels0'].append(0)
        elif labels[i] == 1:
            task_dict['data1'].append(thinking[i])
            task_dict['labels1'].append(1)

    task_dict['data0'] = np.array(task_dict['data0'])
    task_dict['labels0'] = np.array(task_dict['labels0'])
    task_dict['data1'] = np.array(task_dict['data1'])
    task_dict['labels1'] = np.array(task_dict['labels1'])
    return task_dict


def band_pass_and_notch_filter_data(thinking, fs):
    band_pass = signal.butter(5, [7, 70], 'bandpass', fs=fs, output='sos')
    notch_filter_a, notch_filter_b = signal.iirnotch(60, 30, fs)
    filtered_raw_eegs = []
    for epoch in thinking:
        epoch = epoch.transpose()
        filtered_raw_eeg = []
        for channel in epoch:
            filtered = signal.sosfilt(band_pass, channel)
            filtered = signal.filtfilt(notch_filter_a, notch_filter_b, filtered)
            filtered_raw_eeg.append(filtered)
        filtered_raw_eegs.append(filtered_raw_eeg)
    return np.array(filtered_raw_eegs)
