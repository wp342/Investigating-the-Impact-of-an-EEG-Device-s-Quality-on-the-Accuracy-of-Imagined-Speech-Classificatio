import numpy as np
from FEIS_utils import tasks, raw_folders, experiments_dir, SNN_index
from general_utils import divide_task_data
import sklearn.decomposition as sklearn
from sklearn.preprocessing import MinMaxScaler
import os.path as op
import os
import mne
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy.signal import hilbert
import antropy

#experiments_dir = 'kara_one_non_zero_raw_data/' #D:/MSc_Software_Systems/Research Project/FEIS/code_classification/
SNN_data_dir = SNN_index # # './FEIS/code_classification/kara_one_SNN_data/' #D:/MSc_Software_Systems/Research Project/FEIS/code_classification/


def split_data_into_tensor(folder, thinking, labels, task):
    print(task)
    if not op.exists(SNN_data_dir + folder + '/' + task + '_CSP/'):
        os.mkdir(SNN_data_dir + folder + '/' + task + '_CSP/')
    features_dir = SNN_data_dir + folder + '/' + task + '_CSP/'
    task_dict = divide_task_data(labels, tasks[task], thinking)
    for i in range(10):
        p = np.random.permutation(len(task_dict['data']))
        task_dict['data'] = task_dict['data'][p]
        task_dict['labels'] = task_dict['labels'][p]
    eighty_index = int(len(task_dict['data']) * 0.8)
    task_dict['train_data'] = task_dict['data'][0:eighty_index]
    task_dict['train_labels'] = task_dict['labels'][0:eighty_index]
    task_dict['test_data'] = task_dict['data'][eighty_index:]
    task_dict['test_labels'] = task_dict['labels'][eighty_index:]

    split_train_index = int(len(task_dict['train_data'])/2)
    np.save(op.join(features_dir, 'test_data'), task_dict['test_data'])
    np.save(op.join(features_dir, 'test_labels'), task_dict['test_labels'])
    np.save(op.join(features_dir, 'SNN_train_data'), task_dict['train_data'][:split_train_index])
    np.save(op.join(features_dir, 'SNN_train_labels'), task_dict['train_labels'][:split_train_index])
    np.save(op.join(features_dir, 'ML_train_data'), task_dict['train_data'][split_train_index:])
    np.save(op.join(features_dir, 'ML_train_labels'), task_dict['train_labels'][split_train_index:])
    task_dict['train_data'] = task_dict['train_data'][:split_train_index]
    task_dict['train_labels'] = task_dict['train_labels'][:split_train_index]

    return task_dict, features_dir

def create_5_fold_split(folder, thinking, labels, task):
    print(task)
    if not op.exists(SNN_data_dir):
        os.mkdir(SNN_data_dir)
    if not op.exists(SNN_data_dir + folder + '/'):
        os.mkdir(SNN_data_dir + folder + '/')
    if not op.exists(SNN_data_dir + folder + '/' + task + '_CSP_5_fold/'):
        os.mkdir(SNN_data_dir + folder + '/' + task + '_CSP_5_fold/')
    features_dir = SNN_data_dir + folder + '/' + task + '_CSP_5_fold/'
    task_dict = divide_task_data(labels, tasks[task], thinking)
    ones = np.where(task_dict['labels'] == 1)[0]
    zeros = np.where(task_dict['labels'] == 0)[0]
    if abs(len(ones) - len(zeros)) != 0:
        ones = ones[0:min(len(zeros), len(ones))]
        zeros = zeros[0:min(len(zeros), len(ones))]
    kf = KFold(n_splits=5, shuffle=False)
    kf.get_n_splits(ones)
    for index, test_train_index in enumerate(kf.split(ones)):
        train_index, test_index = test_train_index
        train_ones = ones[train_index]
        train_zeros = zeros[train_index]
        test_ones = ones[test_index]
        test_zeros = zeros[test_index]
        comb_train_index = np.concatenate((train_ones, train_zeros))
        comb_test_index = np.concatenate((test_ones, test_zeros))
        np.random.shuffle(comb_test_index)
        np.random.shuffle(comb_train_index)
        if np.sum(task_dict['labels'][comb_train_index]) != len(np.where(task_dict['labels'][comb_train_index] == 0)[0]):
            raise "Incorrectly split!!"
        if np.sum(task_dict['labels'][comb_test_index]) != len(np.where(task_dict['labels'][comb_test_index] == 0)[0]):
            raise "Incorrectly split!!"
        print(comb_train_index)
        print(comb_test_index)
        np.save(op.join(features_dir, 'normalised_train_index_' + str(index)), comb_train_index)
        np.save(op.join(features_dir, 'normalised_test_index_' + str(index)), comb_test_index)

    return task_dict, features_dir


def generate_epoch_pairs(task_dict, split_labels_index, features_dir):
    split_labels = [['train_data', 'train_labels'], ['test_data', 'test_labels']]
    split_label = split_labels[split_labels_index]
    epochs1 = []
    epochs2 = []
    labelss = []
    task_data_labels = []
    for index1, epoch1 in enumerate(task_dict[split_label[0]]):
        epoch1 = np.array(epoch1).transpose()
        epoch1 = np.reshape(epoch1, (epoch1.shape[0], epoch1.shape[1], 1))
        for index2, epoch2 in enumerate(task_dict[split_label[0]]):
            if index2 <= index1:
                continue
            label = 1
            if task_dict[split_label[1]][index1] != task_dict[split_label[1]][index2]:
                label = 0
            epoch = np.array(epoch2)
            epoch = epoch.transpose()
            epoch = np.reshape(epoch, (epoch.shape[0], epoch.shape[1], 1))
            epochs1.append(epoch1)
            epochs2.append(epoch)
            labelss.append(np.float32(label))
            # print('DIFF LAB: ' + str(label[0][0][0][0]))
            task_data_labels.append([task_dict[split_label[1]][index1], task_dict[split_label[1]][index2]])
    #task_data = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(epochs1), tf.data.Dataset
        #                             .from_tensor_slices(epochs2), tf.data.Dataset.from_tensor_slices(labelss)))
    epochs1 = np.array(epochs1)
    epochs2 = np.array(epochs2)
    labelss = np.array(labelss)
    task_data = (epochs1, epochs2, labelss)
    task_data_labels = np.array(task_data_labels, dtype=np.float32)
    np.save(features_dir + split_label[1], task_data_labels)

    return task_data


def load_normalise_data(folders=raw_folders, tasks=tasks, pca=False, pca_components=0.95, down_sample=True,
                        down_sample_frequency=256):
    participants = {}
    for folder in folders:
        print(folder)
        if not op.exists(SNN_data_dir + folder + '/'):
            os.mkdir(SNN_data_dir + folder + '/')
        features_dir = SNN_data_dir + folder + '/'
        thinking = np.load(experiments_dir + folder + '/thinking.npy')
        thinking = np.nan_to_num(thinking)
        labels = np.load(experiments_dir + folder + '/labels.npy')

        if down_sample:
            thinking = down_sample_data(thinking, down_sample_frequency, 62, 1000)

        # min-max normalisation
        thinking = normalise_data(thinking)

        if pca:
            pca_data = []
            pca_runner = sklearn.PCA(n_components=pca_components, copy=True, whiten=True, svd_solver='full')
            for data in thinking:
                pca_runner.fit(data)
                pca_data.append(pca_runner.transform(data))
            thinking = np.array(pca_data)

        task_dict = {}
        for task in tasks:
             task_dict, features_dir = split_data_into_tensor(folder, thinking, labels, task)
             task_dict[task] = generate_epoch_pairs(task_dict, 0, features_dir)
        participants[folder] = task_dict
    return participants

def normalise_data(thinking):
    norm_thinking = []
    for index, epoch in enumerate(thinking):
        epoch = epoch.transpose()
        scaler = MinMaxScaler()
        data_rescaled = scaler.fit_transform(epoch)
        print(np.mean(data_rescaled))
        norm_thinking.append(data_rescaled.transpose())
    return np.array(norm_thinking)


def down_sample_data(thinking, down_sample_frequency, num_of_channels, fs):
    resampled_thinking = []
    for index, epoch in enumerate(thinking):
        mne_resample = mne.io.RawArray(epoch.transpose(),
                                       mne.create_info(ch_names=num_of_channels, sfreq=fs, ch_types='eeg'))
        mne_resample.resample(down_sample_frequency)
        resampled_thinking.append(mne_resample[0:len(mne_resample)][0].transpose())
    return np.array(resampled_thinking)

#calculates the inst frequency and spectral entropy per channel then conconatenates them into a single 2d numpy
def inst_freq_and_spectral_entropy(thinking, fs):
    #inst frequency taken from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    new_thinking = []
    for trial in thinking:
        trial = trial.transpose()
        value_num = 520/len(trial)
        trial_samples = [[], []]
        for channel in trial:
            channel = channel[channel != 0]
            step_size = int(len(channel)/value_num)
            t = np.arange(len(channel)) / fs
            analytic_signal = hilbert(channel)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) /
                                       (2.0 * np.pi) * fs)
            spectral_entropy = []
            instantaneous_frequency_list = []
            new_t = []
            for i in range(int(value_num)):
                spectral_entropy.append(antropy.spectral_entropy(channel[i*step_size:(i*step_size+step_size)], fs, method='fft', normalize=True))
                instantaneous_frequency_list.append(instantaneous_frequency[i*step_size])
                new_t.append(t[i*step_size])
            trial_samples[0].append(np.nan_to_num(spectral_entropy))
            trial_samples[1].append(instantaneous_frequency_list)
        trial_samples[0] = np.array(trial_samples[0]).flatten()
        trial_samples[1] = np.array(trial_samples[1]).flatten()
        new_thinking.append(np.array(trial_samples).transpose())
    return np.array(new_thinking)

# for folder in raw_folders:
#     for task in tasks:
#         #SNN_data_dir = './kara_one_SNN_data/'
#         #experiments_dir = './kara_one_non_zero_raw_data/'  # 'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/kara_one_non_zero_raw_data/'

#         thinking = np.load(experiments_dir + folder + '/thinking.npy')
#         thinking = np.nan_to_num(thinking)
#         labels = np.load(experiments_dir + folder + '/labels.npy')
#         create_5_fold_split(folder, thinking, labels, task)

