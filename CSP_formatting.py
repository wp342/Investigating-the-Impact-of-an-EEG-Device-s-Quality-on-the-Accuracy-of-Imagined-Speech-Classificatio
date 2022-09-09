from NN_Data_Reshaping import down_sample_data
from kara_one_utils import tasks
from general_utils import divide_task_and_class_data, band_pass_and_notch_filter_data
import numpy as np


def down_sample_and_filter(thinking):
    thinking = down_sample_data(thinking, 256, 62, 1000)
    filtered_raw_eegs = band_pass_and_notch_filter_data(thinking, 256)
    return filtered_raw_eegs


#input 'thinking' should be downsampled and appropriately filtered data
def perform_csp_channel_selection(thinking, labels, task, n):
    norm_cov_matrix = []
    for filtered_raw_eeg in thinking:
        # normalised spatial covariance
        multy_transpose = np.matmul(filtered_raw_eeg, filtered_raw_eeg.transpose())
        cov_matrix = multy_transpose / (np.sum(np.diagonal(multy_transpose)))
        norm_cov_matrix.append(cov_matrix)

    norm_cov_dict = divide_task_and_class_data(labels, tasks[task], norm_cov_matrix)

    # averaged_norm_covariance
    zero_avg_norm_cov = np.mean(norm_cov_dict['data0'], axis=0)
    one_avg_norm_cov = np.mean(norm_cov_dict['data1'], axis=0)
    norm_cov_avg_sum = one_avg_norm_cov + zero_avg_norm_cov

    w, v = np.linalg.eig(norm_cov_avg_sum)

    padded_w = np.zeros((len(w), len(w)))

    # negative root eigenvalues
    for index, value in enumerate(w):
        padded_w[index][index] = value**(-1/2)

    # if np.matmul(np.matmul(v, padded_w), v.transpose()) != norm_cov_avg_sum:
    #     raise Exception("v * padded_w * v.transpose() should equal norm_cov_avg_sum")

    whitening_transform_matrix = np.matmul(padded_w, v.transpose())

    zero_shared = np.matmul(np.matmul(whitening_transform_matrix, zero_avg_norm_cov), whitening_transform_matrix.transpose())
    one_shared = np.matmul(np.matmul(whitening_transform_matrix, one_avg_norm_cov), whitening_transform_matrix.transpose())

    w_zero, v_zero = np.linalg.eig(zero_shared)
    w_one, v_one = np.linalg.eig(one_shared)

    zero_indices = (-w_zero).argsort()[:n]
    one_indices = (-w_one).argsort()[:n]
    combined_indices = [*zero_indices, *one_indices]

    #task_dict = divide_task_data(labels, tasks[task], thinking)
    #training_data = []
    #for i in range(len(task_dict['data'])):
       # training_data.append(task_dict['data'][i][combined_indices])
    return combined_indices, zero_indices, one_indices #np.array(training_data)


#training_data = perform_csp_channel_selection(thinking, labels, task)
#training_data = normalise_data(training_data)

print()