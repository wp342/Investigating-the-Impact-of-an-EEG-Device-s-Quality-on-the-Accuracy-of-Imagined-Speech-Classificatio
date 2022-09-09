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
raw_experiments_dir = "./FEIS/data_eeg/" # FEIS Raw data path

experiments_dir = './FEIS - Raw Data/'

MFCC_experiments_dir = './FEIS - MFCC data/'

linear_data_dir = "./FEIS - Linear data/"

SNN_index = 'FEIS_SNN_data/'

raw_folders = sorted(os.listdir(raw_experiments_dir))


