import os

dataset = 'FEIS'
# sampling frequency
fs = 256

channels = 14

window_size = (5*fs)/10

tasks = {
    'binary_cv': [['goose', 'fleece'], ['m', 'n']],
    'binary_nasal': [['m', 'n', 'ng'], ['t', 'sh', 'p']],
    'binary_bilabial': [['p'], ['k']],
    'binary_backness': [['fleece'], ['goose']],
}
#raw_experiments_dir = "D:/MSc_Software_Systems/Research Project/FEIS/data_eeg/"

experiments_dir = 'drive/MyDrive/Kaggle_Runner/FEIS - Raw Data/'

#MFCC_experiments_dir = 'drive/MyDrive/Kaggle_Runner/FEIS - MFCC data/'

#linear_data_dir = "D:/MSc_Software_Systems/Research Project/FEIS/code_classification/FEIS - Linear data/"

SNN_index = 'drive/MyDrive/Kaggle_Runner/FEIS_SNN_data/'

raw_folders = sorted(os.listdir(experiments_dir))


