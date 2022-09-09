import os.path as op
import numpy as np
from FEIS_utils import window_size, raw_folders, experiments_dir, linear_data_dir, MFCC_experiments_dir, fs, channels
import scipy.signal as signal
import sklearn.decomposition as sklearn
import python_speech_features as mfcc
import os
from general_utils import band_pass_and_notch_filter_data
from make_svm_features import make_simple_feats, add_deltas
import mne


def load_data_and_labels(experiment):
    data = np.load(experiments_dir + experiment + '/thinking.npy')
    labels = np.load(experiments_dir + experiment + '/labels.npy')
    return data, labels


# def perform_ica(epochs, experiment):
#     # window and ICA on channels to reduce to 40
#     transformer = sklearn.FastICA(n_components=channels, random_state=None, whiten='unit-variance')
#     print('Commencing Windowing and ICA')
#     preprocessed_epochs = []
#     for index, sample in enumerate(filtered_raw_eeg[0:1]):
#         preprocessed_windows = []
#         print(str(1+index) + '/' + str(len(filtered_raw_eeg)) + ' ICA COMPLETE')
#         sample = sample.transpose()
#         for i in range(0, len(sample) - 500, 250):
#             window = sample[i:i + 500]
#             ica_data = transformer.fit_transform(window)
#             preprocessed_windows.append(ica_data)
#         preprocessed_epochs.append(preprocessed_windows)
#     preprocessed_epochs = np.array(preprocessed_epochs)
#
#     print('Pre-Processing Complete for ' + experiment + ' final shape ' + str(preprocessed_epochs.shape))
#     return preprocessed_epochs


def window_data(data, window_size):
    windowed_data = []
    for index, sample in enumerate(data):
        sample_windows = []
        for i in range(0, len(sample) - int(window_size)+1, int(window_size/2)):
            sample_windows.append(sample[i:i + int(window_size)])
        windowed_data.append(sample_windows)
    windowed_data = np.array(windowed_data)
    return windowed_data


##### FEATURE EXTRACTION ##############################
## Mel Frequency Cepstral Coefficients https://arxiv.org/ftp/arxiv/papers/1003/1003.4083.pdf
def mel_freq_coeff(ica):
    transformer = sklearn.FastICA(n_components=channels, random_state=None, whiten='unit-variance')
    for experiment in raw_folders:
        data, labels = load_data_and_labels(experiment)
        data = band_pass_and_notch_filter_data(data, fs)
        trials = []
        for trial in data:
            trial = trial.transpose()
            if ica:
                trial = transformer.fit_transform(trial)
            trials.append(trial)
        data = window_data(trials, window_size)
        sample_coeff = []
        print('Computing MFCC .........')
        for index, sample in enumerate(data):
            print(str(1 + index) + '/' + str(len(data)) + ' MFCC COMPLETE')
            window_coeff = []
            for window in sample:
                window = window.transpose()
                channel_coeff = []
                for channel in window:
                    channel_coeff.append(np.sum(mfcc.mfcc(channel, samplerate=fs), axis=0))
                window_coeff.append(channel_coeff)
            window_coeff = np.hstack(window_coeff)
            sample_coeff.append(window_coeff)
        if not op.exists(MFCC_experiments_dir + experiment):
            os.mkdir(MFCC_experiments_dir + experiment)
        if ica:
            filename = 'ica_MFCC_data'
        else:
            filename = 'MFCC_data'
        np.save(MFCC_experiments_dir + experiment + '/' + filename, sample_coeff)
        print(experiment + ' Complete')


## Linear data
def compute_linear_features(ica):
    transformer = sklearn.FastICA(n_components=channels, random_state=None, whiten='unit-variance')
    for experiment in raw_folders:
        data, labels = load_data_and_labels(experiment)
        data = band_pass_and_notch_filter_data(data, fs)
        trials = []
        for trial in data:
            trial = trial.transpose()
            if ica:
                trial = transformer.fit_transform(trial)
            trials.append(trial)
        data = window_data(trials, window_size)
        epochs = []
        for epoch in data:
            epoch = make_simple_feats(epoch)
            epoch = add_deltas(epoch)
            epoch_shape = epoch.shape
            epoch = epoch.reshape((epoch_shape[0], channels, int(epoch_shape[1] / channels)))
            epoch = np.vstack(epoch)
            epoch = epoch.transpose()
            epochs.append(epoch)

        if not op.exists(linear_data_dir + experiment):
            os.mkdir(linear_data_dir + experiment)
        if ica:
            filename = 'ica_linear_data'
        else:
            filename = 'linear_data'
        np.save(linear_data_dir + experiment + '/' + filename, epochs)
        print(experiment + ' Complete')



#split 5 into 2000ms overlapping windows
def FEIS_split_compute_linear_features(ica):
    transformer = sklearn.FastICA(n_components=channels, random_state=None, whiten='unit-variance')
    for experiment in raw_folders:
        data, labels = load_data_and_labels(experiment)
        data = band_pass_and_notch_filter_data(data, fs)
        trials = []
        for trial in data:
            trial = trial.transpose()
            trial_splits = []
            jumps = [0, fs / 2, fs, fs, fs / 2]
            start = 0
            for jump in jumps:
                start = start + jump
                trial_split = trial[int(start): int(start+2*fs), :]
                #print(np.array(trial_split).shape)
                if ica:
                    trial_split = transformer.fit_transform(trial_split)
                trial_splits.append(trial_split)
            trials.append(trial_splits)


        datas = []
        for trial in trials:
            data = window_data(trial, window_size)
            datas.append(data)

        all_epochs = []
        for data in datas:
            epochs = []
            for epoch in data:
                epoch = make_simple_feats(epoch)
                epoch = add_deltas(epoch)
                epoch_shape = epoch.shape
                epoch = epoch.reshape((epoch_shape[0], channels, int(epoch_shape[1] / channels)))
                epoch = np.vstack(epoch)
                epoch = epoch.transpose()
                epochs.append(epoch)
            all_epochs.append(epochs)

        if not op.exists(linear_data_dir + experiment):
            os.mkdir(linear_data_dir + experiment)
        if ica:
            filename = 'split_ica_linear_data'
        else:
            filename = 'split_linear_data'
        np.save(linear_data_dir + experiment + '/' + filename, all_epochs)
        print(experiment + ' Complete')


## Mel Frequency Cepstral Coefficients https://arxiv.org/ftp/arxiv/papers/1003/1003.4083.pdf
def FEIS_split_mel_freq_coeff(ica):
    transformer = sklearn.FastICA(n_components=channels, random_state=None, whiten='unit-variance')
    for experiment in raw_folders:
        data, labels = load_data_and_labels(experiment)
        data = band_pass_and_notch_filter_data(data, fs)
        trials = []
        for trial in data:
            trial = trial.transpose()
            trial_splits = []
            jumps = [0, fs / 2, fs, fs, fs / 2]
            start = 0
            for jump in jumps:
                start = start + jump
                trial_split = trial[int(start): int(start + 2 * fs), :]
                # print(np.array(trial_split).shape)
                if ica:
                    trial_split = transformer.fit_transform(trial_split)
                trial_splits.append(trial_split)
            trials.append(trial_splits)
        datas = []
        for trial in trials:
            data = window_data(trial, window_size)
            datas.append(data)

        print('Computing MFCC .........')
        all_sample_coeff = []
        for ind, data in enumerate(datas):
            print(str(1 + ind) + '/' + str(len(datas)) + ' MFCC COMPLETE')
            sample_coeff = []
            for index, sample in enumerate(data):
                window_coeff = []
                for window in sample:
                    window = window.transpose()
                    channel_coeff = []
                    for channel in window:
                        channel_coeff.append(np.sum(mfcc.mfcc(channel, samplerate=fs), axis=0))
                    window_coeff.append(channel_coeff)
                window_coeff = np.hstack(window_coeff)
                sample_coeff.append(window_coeff)
            all_sample_coeff.append(sample_coeff)
        if not op.exists(MFCC_experiments_dir + experiment):
            os.mkdir(MFCC_experiments_dir + experiment)
        if ica:
            filename = 'split_ica_MFCC_data'
        else:
            filename = 'split_MFCC_data'
        np.save(MFCC_experiments_dir + experiment + '/' + filename, all_sample_coeff)
        print(experiment + ' Complete')

FEIS_split_mel_freq_coeff(False)
FEIS_split_compute_linear_features(False)
FEIS_split_mel_freq_coeff(True)
FEIS_split_compute_linear_features(True)

compute_linear_features(True)
compute_linear_features(False)
mel_freq_coeff(True)
mel_freq_coeff(False)

#
# def hurst_exponent(component_data):
#     H, C, data = compute_Hc(component_data)
#     return H
#      filtered_raw_eeg.append(mne.filter.filter_data(epoch, sfreq=1000, l_freq=50, h_freq=None))
#  filtered_raw_eeg = np.array(filtered_raw_eeg)
#  filtered_raw_eeg = mne.io.RawArray(filtered_raw_eeg, mne.create_info(channels, fs, ch_types='eeg', verbose=None), first_samp=0, copy='auto', verbose=None)
#  #filtered_raw_eeg = filtered_raw_eeg.pick_types(eeg=True, stim=True).load_data()
#  #events = mne.find_events(filtered_raw_eeg)
#  #filtered_raw_eeg.set_eeg_reference(projection=True).apply_proj()
#  # Apply small laplacian filter
# # raw_csd = mne.preprocessing.compute_current_source_density(filtered_raw_eeg)
# # raw_csd.plot()
# # print()
#  ica = mne.preprocessing.ICA()
#  ica = ica.fit(filtered_raw_eeg)
#  ica_data = ica.apply(filtered_raw_eeg)
#  print()


# Window the data into 500 ms blocks with 250 ms overlap

# Apply ICA on the dataset
