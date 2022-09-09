import os

dataset = 'karaone'
# sampling frequency
fs = 1000

channels = 62

window_size = 500

tasks = {
    'binary_cv': [['/uw/', '/iy/'], ['/m/', '/n/']],
    'binary_nasal': [['/tiy/', '/piy/', 'pat'], ['/m/', '/n/', 'gnaw']],
    'binary_bilabial': [['/tiy/'], ['/piy/']],
    'binary_backness': [['/uw/'], ['/iy/']],
}

#raw_experiments_dir = "/Research Project/karaone/"  #Kara One Raw data path



MFCC_experiments_dir = './karaone - MFCC data/'

raw_folders = sorted(os.listdir(MFCC_experiments_dir))

linear_data_dir = "./karaone - Linear data/"

experiments_dir = './kara_one_non_zero_raw_data/'


