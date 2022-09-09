import os

dataset = 'karaone'
# sampling frequency
fs = 1000

channels = 62

window_size = 500

tasks = {
    'binary_cv': [['/uw/', '/iy/'], ['/m/', '/n/']],
    'binary_nasal': [['/tiy/', '/piy/', 'pat'], ['/m/', '/n/', 'gnaw']], # , 'pot' 'knew',
    'binary_bilabial': [['/tiy/'], ['/piy/']], # 'pat', 'pot'   , '/iy/', '/n/'
    'binary_backness': [['/uw/'], ['/iy/']],
}
#'binary_iy': [['/tiy/', '/iy/', '/piy/', '/diy/'], ['gnaw', 'pat', 'knew', 'pot']],
#'binary_uw': [['/uw/', 'knew'], ['/iy/', 'gnaw']],
#'binary_voice': [['/tiy/'], ['/diy/']],

#raw_experiments_dir = "D:/MSc_Software_Systems/Research Project/karaone/"

MFCC_experiments_dir = './karaone - MFCC data/' # D:/MSc_Software_Systems/Research Project/FEIS/code_classification

linear_data_dir = "./karaone - Linear data/" #D:/MSc_Software_Systems/Research Project/FEIS/code_classification

experiments_dir = 'drive/MyDrive/Kaggle_Runner/kara_one_non_zero_raw_data/' #D:/MSc_Software_Systems/Research Project/FEIS/code_classification

raw_folders = sorted(os.listdir(experiments_dir))

