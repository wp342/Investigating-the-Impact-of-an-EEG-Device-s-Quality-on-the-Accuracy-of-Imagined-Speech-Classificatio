import matplotlib.pyplot as plt
import os
import os.path as op
import numpy as np
from three_fold_CSP_SNN_Data_Analyser import data_analyser

plot_dir = './Results Plots/'

data_types = ["linear", "linear_ica", "MFCC", "MFCC_ica"]


def get_ML_var(filename):
    return int(filename.split('_')[0])


ML_list = ['Random Forest', 'KNN', 'SVM']
do_not_add = ['sigmoid', 'chebyshev']
datasets = ["karaone", "FEIS"]
for dataset in datasets:
    for ML in ML_list:
        labels = []
        plot_dict = {}
        model_dir = './' + dataset + ' - ' + ML + ' results/'
        test_data_files = sorted(os.listdir(model_dir))

        task_highest_avg_value = {}
        test_data_file_removers = [s for s in test_data_files if ".csv" not in s or "_std.csv" in s]
        for remove in test_data_file_removers:
            test_data_files.remove(remove)

        test_data_files = [s for s in test_data_files if "SNN" not in s]
        if "RAND_FOREST" in test_data_files[0]:
            test_data_files.sort(key=get_ML_var)

        for test_data_file in test_data_files:
            # if '_DR' not in test_data_file:
            #     continue
            try:
                numpy_features = np.genfromtxt(op.join(model_dir, test_data_file), delimiter=",", dtype=float)
                tasks = np.genfromtxt(op.join(model_dir, test_data_file), delimiter=",", dtype=str)[1:, 0]
            except:
                continue
            numpy_features = numpy_features[1:, 1:]*100
            average_list = numpy_features[:, -1]
            list_mean = np.mean(average_list)
            std = np.std(average_list)
            #low_error = list_mean - min(average_list)
            #high_error = max(average_list) - list_mean
            split_fname = test_data_file.split('_')
            ML_var = split_fname[0]
            if 'SNN' in split_fname:
                continue
            split_fname.remove(ML_var)
            split_fname = '_'.join(split_fname)
            if ML_var in do_not_add:
                continue

            if ML_var not in labels:
                labels.append(ML_var)
            for data_type in data_types:
                if data_type not in plot_dict:
                    plot_dict[data_type] = []
                    plot_dict["split_" + data_type] = []
                    plot_dict["split_" + data_type + "_std"] = []
                    plot_dict[data_type + "_std"] = []
                if data_type in split_fname:
                    if "ica" in split_fname and "ica" not in data_type:
                        continue
                    if "split" in split_fname:
                        plot_dict["split_" + data_type].append(list_mean)
                        plot_dict["split_" + data_type + "_std"].append(std)

                        continue
                    plot_dict[data_type].append(list_mean)
                    plot_dict[data_type + "_std"].append(std)
                    continue

        x = np.arange(len(labels))  # the label locations
        x = x * 2
        width = 0.40  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, plot_dict[data_types[0]], width, label=data_types[0],
                        yerr=plot_dict[data_types[0] + "_std"], capsize=3)
        rects2 = ax.bar(x - 1.5 * width, plot_dict[data_types[1]], width, label=data_types[1],
                        yerr=plot_dict[data_types[1] + "_std"],capsize=3)
        rects3 = ax.bar(x + 1.5 * width, plot_dict[data_types[2]], width, label=data_types[2],
                        yerr=plot_dict[data_types[2] + "_std"], capsize=3)
        rects4 = ax.bar(x + width / 2, plot_dict[data_types[3]], width, label=data_types[3],
                        yerr=plot_dict[data_types[3] + "_std"], capsize=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Total Percentage Accuracy (%)')

        #ax.set_title('Comparison of ' + dataset + ' ' + ML + ' Variables')
        ax.set_xticks(x, labels)
        ax.legend(loc=(1.01, 0.4))
        #ax.legend()

        if 'Forest' in ML:
            ax.set_xlabel('Number of Trees')
        elif 'SVM' in ML:
            ax.set_xlabel('Kernel Type')
        elif 'KNN' in ML:
            ax.set_xlabel('Distance Metric')

        # ax.bar_label(rects1, padding=2, fmt='%.2f', label_type='center')
        # ax.bar_label(rects2, padding=2, fmt='%.2f', label_type='center')
        # ax.bar_label(rects3, padding=2, fmt='%.2f', label_type='center')
        # ax.bar_label(rects4, padding=2, fmt='%.2f', label_type='center')

        fig.tight_layout()

        plt.axis([None, None, 45, 60])  # max(plot_dict[data_types[0]])+0.1]
        plt.savefig(plot_dir + 'Comparison of ' + dataset + ' ' + ML + ' Variables.png')
        plt.show()


        if dataset == "FEIS" and ML != "SVM":
            split_fig, split_ax = plt.subplots()
            rects5 = split_ax.bar(x - width / 2, plot_dict["split_" + data_types[0]], width, label=data_types[0],
                                  yerr=plot_dict[data_types[3] + "_std"],capsize=3)
            rects6 = split_ax.bar(x - 1.5 * width, plot_dict["split_" + data_types[0]], width, label=data_types[1],
                                  yerr=plot_dict[data_types[3] + "_std"],capsize=3)
            rects7 = split_ax.bar(x + 1.5 * width, plot_dict["split_" + data_types[0]], width, label=data_types[2],
                                  yerr=plot_dict[data_types[3] + "_std"],capsize=3)
            rects8 = split_ax.bar(x + width / 2, plot_dict["split_" + data_types[0]], width, label=data_types[3],
                                  yerr=plot_dict[data_types[3] + "_std"],capsize=3)
            # Add some text for labels, title and custom x-axis tick labels, etc.
            split_ax.set_ylabel('Total Percentage Accuracy (%)')

            #split_ax.set_title('Comparison of Split ' + dataset + ' ' + ML + ' Variables')
            split_ax.set_xticks(x, labels)
            split_ax.legend(loc=(1.01, 0.4))

            if 'Forest' in ML:
                split_ax.set_xlabel('Number of Trees')
            elif 'SVM' in ML:
                split_ax.set_xlabel('Kernel Type')
            elif 'KNN' in ML:
                split_ax.set_xlabel('Distance Metric')

            # ax.bar_label(rects1, padding=2, fmt='%.2f', label_type='center')
            # ax.bar_label(rects2, padding=2, fmt='%.2f', label_type='center')
            # ax.bar_label(rects3, padding=2, fmt='%.2f', label_type='center')
            # ax.bar_label(rects4, padding=2, fmt='%.2f', label_type='center')

            split_fig.tight_layout()

            plt.axis([None, None, 45, 60])  # max(plot_dict[data_types[0]])+0.1]
            plt.savefig(plot_dir + 'Comparison of Split ' + dataset + ' ' + ML + ' Variables.png', bbox_inches='tight')
            plt.show()


def get_channels_selected(filename):
    return int(filename.split('_')[-1].split('.')[0])

current_data_size = 0
channel_selection = ["CSP", "", "Pearson Correlation"]

for dataset in datasets:
    for ML in ML_list:
        plt.figure(1)
        plt.xlabel('Number of Feature Channels Selected')
        plt.ylabel('Total Average Accuracy (%)')
        for channel in channel_selection:
            plot_dict = {}
            model_dir = './' + dataset + ' - ' + ML + ' results/' + channel + '/'
            if channel == "":
                _, value = data_analyser(model_dir)
                plt.plot(labels, [value*100]*current_data_size, label='No Channel Selection')
                continue
            test_data_files = sorted(os.listdir(model_dir))

            task_highest_avg_value = {}
            test_data_file_removers = [s for s in test_data_files if ".csv" not in s or "_std.csv" in s]

            for remove in test_data_file_removers:
                test_data_files.remove(remove)
            test_data_dict = {#'linear': [s for s in test_data_files if "linear" in s and "double" not in s],
                              'MFCC': [s for s in test_data_files if "MFCC" in s and "double" not in s]}

            #if "RAND_FOREST" in test_data_files[0]:
            #test_data_dict['linear'].sort(key=get_channels_selected)
            test_data_dict['MFCC'].sort(key=get_channels_selected)

            for feature in test_data_dict:
                list_of_means = []
                error_list = [[], []]
                labels = []
                for test_data_file in test_data_dict[feature]:
                    # if '_DR' not in test_data_file:
                    #     continue
                    numpy_features = np.genfromtxt(op.join(model_dir, test_data_file), delimiter=",", dtype=float)
                    try:
                        tasks = np.genfromtxt(op.join(model_dir, test_data_file), delimiter=",", dtype=str)[1:, 0]
                    except:
                        continue
                    numpy_features = numpy_features[1:, 1:]*100
                    average_list = numpy_features[:, -1]
                    list_mean = np.mean(average_list)
                    low_error = list_mean - min(average_list)
                    high_error = max(average_list) - list_mean
                    list_of_means.append(list_mean)
                    error_list[0].append(low_error)
                    error_list[1].append(high_error)
                    labels.append(get_channels_selected(test_data_file))
                current_data_size = len(labels)

                plt.plot(labels, list_of_means, label=channel)

        #plt.title("Accuracy of " + dataset + ' ' + feature + ' ' + ML + ' with feature selection using ' + channel)
        plt.axis([None, None, 45, 60])
        plt.legend(loc=(1.01, 0.4))
        plt.savefig(plot_dir + 'Accuracy of ' + dataset + ' ' + feature + ' ' + ML + '.png', bbox_inches='tight')
        plt.show()

channel_selection = ["", "CSP", "Pearson Correlation"]

ML_list.append('SNN')
feature_extractors = ['_linear_', 'MFCC', 'SNN']

highest_per_ML_per_DS = {}
highest_per_ML_per_channel_selection = {}
highest_per_ML_per_DS_per_feature = {}
for dataset in datasets:
    highest_per_ML_per_DS[dataset] = {}
    highest_per_ML_per_channel_selection[dataset] = {}
    highest_per_ML_per_DS_per_feature[dataset] = {}
    for ML in ML_list:
        highest_per_ML_per_channel_selection[dataset][ML] = {}
        highest_per_ML_per_DS_per_feature[dataset][ML] = {}
        for channel in channel_selection:
            if channel != "" and ML == "SNN":
                continue
            model_dir = './' + dataset + ' - ' + ML + ' results/' + channel + '/'
            #for feature in feature_extractors:
            highest_filename, highest_average = data_analyser(model_dir) # , feature)
            highest_per_ML_per_channel_selection[dataset][ML][channel] = highest_filename
            if ML not in highest_per_ML_per_DS[dataset]:
                highest_per_ML_per_DS[dataset][ML] = ['0', 'uninitialised']
            if float(highest_per_ML_per_DS[dataset][ML][0]) < float(highest_average):
                highest_per_ML_per_DS[dataset][ML][0] = str(highest_average)
                highest_per_ML_per_DS[dataset][ML][1] = model_dir + highest_filename
            for feature in feature_extractors:
                highest_filename, highest_average = data_analyser(model_dir, feature)
                if feature not in highest_per_ML_per_DS_per_feature[dataset][ML]:
                    highest_per_ML_per_DS_per_feature[dataset][ML][feature] = ['0', 'uninitialised']
                if float(highest_per_ML_per_DS_per_feature[dataset][ML][feature][0]) < float(highest_average):
                    highest_per_ML_per_DS_per_feature[dataset][ML][feature][0] = str(highest_average)
                    highest_per_ML_per_DS_per_feature[dataset][ML][feature][1] = model_dir + highest_filename


for trial_dataset in highest_per_ML_per_DS:
    plot_dict = {}
    for top_ML in highest_per_ML_per_DS[trial_dataset]:
        numpy_features = np.genfromtxt(highest_per_ML_per_DS[trial_dataset][top_ML][1], delimiter=",", dtype=float)
        participants = np.genfromtxt(highest_per_ML_per_DS[trial_dataset][top_ML][1], delimiter=",", dtype=str)[0, 1:-1]
        tasks = np.genfromtxt(highest_per_ML_per_DS[trial_dataset][top_ML][1], delimiter=",", dtype=str)[1:, 0]
        numpy_features = numpy_features[1:, 1:-1]*100
        plot_dict[top_ML] = np.mean(numpy_features, axis=0)
        plot_dict[top_ML + "_std"] = np.std(numpy_features, axis=0)
        plot_dict[top_ML + "_tasks"] = np.mean(numpy_features, axis=1)
        plot_dict[top_ML + "_tasks_std"] = np.std(numpy_features, axis=1)

    x = np.arange(len(participants))  # the label locations
    x = x * 3.5
    width = 1  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width, plot_dict[ML_list[0]], width, label=ML_list[0],
                    yerr=plot_dict[ML_list[0] + "_std"],capsize=3)
    rects2 = ax.bar(x, plot_dict[ML_list[1]], width, label=ML_list[1],
                    yerr=plot_dict[ML_list[1] + "_std"],capsize=3)
    rects3 = ax.bar(x + width, plot_dict[ML_list[2]], width, label=ML_list[2],
                    yerr=plot_dict[ML_list[2] + "_std"],capsize=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage Accuracy (%)')

    #ax.set_title('Top Performing ' + trial_dataset + ' ML Result Comparison per Participant')
    ax.set_xticks(x, participants)
    ax.legend(loc=(1.01, 0.4))

    fig.tight_layout()
    #ax.bar_label(rects1, padding=2, fmt='%.2f')
    #ax.bar_label(rects2, padding=2, fmt='%.2f')
    #ax.bar_label(rects3, padding=2, fmt='%.2f')

    ax.set_xlabel('Participants')

    fig.tight_layout()

    plt.axis([None, None, 35, 85])  # max(plot_dict[data_types[0]])+0.1]

    fig.set_figwidth(15)
    plt.savefig(plot_dir + 'Top Performing ' + trial_dataset + ' ML Result Comparison per Participant.png', bbox_inches='tight')
    plt.show()


x = np.arange(len(tasks))  # the label locations
# x = x * 2
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, plot_dict[ML_list[0] + "_tasks"], width, label=ML_list[0],
                yerr=plot_dict[ML_list[0] + "_tasks_std"],capsize=3)
rects2 = ax.bar(x, plot_dict[ML_list[1] + "_tasks"], width, label=ML_list[1],
                yerr=plot_dict[ML_list[1] + "_tasks_std"],capsize=3)
rects3 = ax.bar(x + width, plot_dict[ML_list[2] + "_tasks" ], width, label=ML_list[2],
                yerr=plot_dict[ML_list[2] + "_tasks_std"],capsize=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage Accuracy (%)')

#ax.set_title('Top Performing ' + trial_dataset + ' ML Result Comparison per Task')
ax.set_xticks(x, tasks)
ax.legend(loc=(1.01, 0.4))

ax.set_xlabel('Tasks')

fig.tight_layout()
#ax.bar_label(rects1, padding=2, fmt='%.2f')
#ax.bar_label(rects2, padding=2, fmt='%.2f')
#ax.bar_label(rects3, padding=2, fmt='%.2f')

plt.axis([None, None, 35, 85])  # max(plot_dict[data_types[0]])+0.1]

fig.set_figwidth(15)
plt.savefig(plot_dir + 'Top Performing ' + trial_dataset + ' ML Result Comparison per Task.png', bbox_inches='tight')
plt.show()

# channel_select = [['FEIS', 'KNN'], ['karaone', 'Random Forest']]
#
# plot_dict = {}
# labels = ['FEIS', 'Kara One']
# for top_channel in channel_select:
#     for channel in channel_selection:
#         model_dir = 'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/' + top_channel[0] + ' - ' + \
#                     top_channel[1] + ' results/' + channel + '/'
#         numpy_features = np.genfromtxt(model_dir + highest_per_ML_per_channel_selection[top_channel[0]][top_channel[1]][channel],
#                                        delimiter=",", dtype=float)
#         numpy_features = numpy_features[1:, -1] * 100
#         try:
#             plot_dict[channel].append(np.mean(numpy_features))
#             plot_dict[channel + "_std"].append(np.std(numpy_features, axis=0))
#         except:
#             plot_dict[channel] = []
#             plot_dict[channel + "_std"] = []
#             plot_dict[channel].append(np.mean(numpy_features))
#             plot_dict[channel + "_std"].append(np.std(numpy_features, axis=0))
#
# x = np.arange(len(labels))  # the label locations
# width = .2  # the width of the bars
#
# fig, ax = plt.subplots()
#
# rects1 = ax.bar(x - width, plot_dict[channel_selection[0]], width, label='No Channel Selection',
#                 yerr=plot_dict[channel_selection[0] + "_std"])
# rects2 = ax.bar(x, plot_dict[channel_selection[1]], width, label=channel_selection[1],
#                 yerr=plot_dict[channel_selection[1] + "_std"])
# rects3 = ax.bar(x + width, plot_dict[channel_selection[2]], width, label=channel_selection[2],
#                 yerr=plot_dict[channel_selection[2] + "_std"])
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Percentage Accuracy (%)')
#
# # ax.set_title('Top Performing ' + trial_dataset + ' ML Result Comparison per Participant')
# ax.set_xticks(x, labels)
# ax.legend(loc=(1.01, 0.4))
#
# fig.tight_layout()
# # ax.bar_label(rects1, padding=2, fmt='%.2f')
# # ax.bar_label(rects2, padding=2, fmt='%.2f')
# # ax.bar_label(rects3, padding=2, fmt='%.2f')
#
# ax.set_xlabel('DataSets')
#
# fig.tight_layout()
#
# plt.axis([None, None, 50, 60])  # max(plot_dict[data_types[0]])+0.1]
#
# plt.savefig(plot_dir + 'Top Channel Selection Comparisons.png',
#             bbox_inches='tight')
# plt.show()

#FEIS VS KARA ONE COMPARISONS

plot_dict = {}
for trial_dataset in highest_per_ML_per_DS:
    plot_dict[trial_dataset] = {}
    for top_ML in highest_per_ML_per_DS[trial_dataset]:
        numpy_features = np.genfromtxt(highest_per_ML_per_DS[trial_dataset][top_ML][1], delimiter=",", dtype=float)
        participants = np.genfromtxt(highest_per_ML_per_DS[trial_dataset][top_ML][1], delimiter=",", dtype=str)[0, 1:-1]
        tasks = np.genfromtxt(highest_per_ML_per_DS[trial_dataset][top_ML][1], delimiter=",", dtype=str)[1:, 0]
        average_list = numpy_features[1:, -1]
        numpy_features = numpy_features[1:, 1:-1]*100
        plot_dict[trial_dataset][top_ML] = np.mean(numpy_features, axis=0)
        plot_dict[trial_dataset][top_ML + "_std"] = np.std(numpy_features, axis=0)
        plot_dict[trial_dataset][top_ML + "_tasks"] = np.mean(numpy_features, axis=1)
        plot_dict[trial_dataset][top_ML + "_tasks_std"] = np.std(numpy_features, axis=1)
        plot_dict[trial_dataset][top_ML + "_mean_std"] = np.std(average_list*100)

x = np.arange(len(tasks))  # the label locations
x = x * 1.25
width = 0.15  # the width of the bars

fig, ax = plt.subplots()

dataset = ['FEIS', 'karaone']

rects1 = ax.bar(x - 3.5 *width, plot_dict[dataset[0]][ML_list[0] + "_tasks"], width, label=dataset[0] + " " + ML_list[0],
                yerr=plot_dict[dataset[0]][ML_list[0] + "_tasks_std"],capsize=3)
rects2 = ax.bar(x - 2.5* width, plot_dict[dataset[1]][ML_list[0] + "_tasks"], width, label=dataset[1] + " " + ML_list[0],
                yerr=plot_dict[dataset[1]][ML_list[0] + "_tasks_std"],capsize=3)
rects3 = ax.bar(x - 1.5*width, plot_dict[dataset[0]][ML_list[1] + "_tasks"], width, label=dataset[0] + " " + ML_list[1],
                yerr=plot_dict[dataset[0]][ML_list[1] + "_tasks_std"],capsize=3)
rects4 = ax.bar(x - .5*width, plot_dict[dataset[1]][ML_list[1] + "_tasks"], width, label=dataset[1] + " " + ML_list[1],
                yerr=plot_dict[dataset[1]][ML_list[1] + "_tasks_std"],capsize=3)
rects5 = ax.bar(x + .5*width, plot_dict[dataset[0]][ML_list[2] + "_tasks" ], width, label=dataset[0] + " " + ML_list[2],
                yerr=plot_dict[dataset[0]][ML_list[2] + "_tasks_std"],capsize=3)
rects6 = ax.bar(x + 1.5*width, plot_dict[dataset[1]][ML_list[2] + "_tasks" ], width, label=dataset[1] + " " + ML_list[2],
                yerr=plot_dict[dataset[1]][ML_list[2] + "_tasks_std"],capsize=3)
rects7= ax.bar(x + 2.5*width, plot_dict[dataset[0]][ML_list[3] + "_tasks" ], width, label=dataset[0] + " " + ML_list[3],
                yerr=plot_dict[dataset[0]][ML_list[3] + "_tasks_std"],capsize=3)
rects8 = ax.bar(x + 3.5*width, plot_dict[dataset[1]][ML_list[3] + "_tasks" ], width, label=dataset[1] + " " + ML_list[3],
                yerr=plot_dict[dataset[1]][ML_list[3] + "_tasks_std"],capsize=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage Accuracy (%)')

#ax.set_title('Comparison across FEIS and Kara One datasets on Top Performing ML Result Comparison per Task')
ax.set_xticks(x, tasks)
ax.legend(loc=(1.01, 0.25))

ax.set_xlabel('Tasks')

fig.tight_layout()
#ax.bar_label(rects1, padding=2, fmt='%.2f')
#ax.bar_label(rects2, padding=2, fmt='%.2f')
#ax.bar_label(rects3, padding=2, fmt='%.2f')
#ax.bar_label(rects4, padding=2, fmt='%.2f')
#ax.bar_label(rects5, padding=2, fmt='%.2f')
#ax.bar_label(rects6, padding=2, fmt='%.2f')
#ax.bar_label(rects7, padding=2, fmt='%.2f')
#ax.bar_label(rects8, padding=2, fmt='%.2f')

plt.axis([None, None, 35, 85])  # max(plot_dict[data_types[0]])+0.1]

fig.set_figwidth(30)
#plt.savefig(plot_dir + 'Comparison across FEIS and Kara One datasets on Top Performing ML Result Comparison per Task.png', bbox_inches='tight')
plt.show()


#SNN MLs Epochs VS Accuracy

def get_epoch(filename):
    return int(filename.split('_')[0].split('.')[0])

ML_list = ['Random_forest', 'KNN', 'SVM']
SNN_result_sets = ['SNN', 'SNN CSP']
for dataset in datasets:
    for SNN_result_set in SNN_result_sets:
        model_dir = './' + dataset + ' - ' + SNN_result_set + ' results/'
        test_data_files = sorted(os.listdir(model_dir))
        SNN_Results = [s for s in test_data_files if ".csv" in s and "_std.csv" not in s]
        plot_dict = {}
        for ML in ML_list:
            plot_dict[ML] = [s for s in SNN_Results if ML in s]
            plot_dict[ML].sort(key=get_epoch)
        plotting_dict = {}
        for ML in plot_dict:
            for test_run in plot_dict[ML]:
                numpy_features = np.genfromtxt(model_dir + test_run, delimiter=",", dtype=float)
                numpy_features = numpy_features[1:, 1:]*100
                #task_average  = num
                average_column = numpy_features[:, -1]
                try:
                    plotting_dict[ML].append(np.mean(average_column))
                    plotting_dict[ML + "_std"].append(np.std(average_column))
                except:
                    plotting_dict[ML] = []
                    plotting_dict[ML + "_std"] = []
                    plotting_dict[ML].append(np.mean(average_column))
                    plotting_dict[ML + "_std"].append(np.std(average_column))

                #plot_dict[top_ML] = np.mean(numpy_features, axis=0)
                #plot_dict[top_ML + "_std"] = np.std(numpy_features, axis=0)
                #plot_dict[top_ML + "_tasks"] = np.mean(numpy_features, axis=1)
                #plot_dict[top_ML + "_tasks_std"] = np.std(numpy_features, axis=1)

        x = np.arange(len(range(10)))  # the label locations
        x = x * 2
        width = 0.5  # the width of the bars

        fig, ax = plt.subplots()

        rects1 = ax.bar(x - width, plotting_dict[ML_list[0]], width, label=ML_list[0],
                        yerr=plotting_dict[ML_list[0] + "_std"], capsize=3)
        rects2 = ax.bar(x, plotting_dict[ML_list[1]], width, label=ML_list[1],
                        yerr=plotting_dict[ML_list[1] + "_std"], capsize=3)
        rects3 = ax.bar(x + width, plotting_dict[ML_list[2]], width, label=ML_list[2],
                        yerr=plotting_dict[ML_list[2] + "_std"], capsize=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Percentage Accuracy (%)')

        #ax.set_title(dataset + ' ' + SNN_result_set + ' ML accuracies as Training epochs Increase')
        ax.set_xticks(x, range(100, 1001, 100))
        ax.legend(loc=(1.01, 0.4))

        fig.tight_layout()
       # ax.bar_label(rects1, padding=2, fmt='%.2f')
       # ax.bar_label(rects2, padding=2, fmt='%.2f')
       # ax.bar_label(rects3, padding=2, fmt='%.2f')

        ax.set_xlabel('Epochs')

        fig.tight_layout()

        plt.axis([None, None, 45, 55])  # max(plot_dict[data_types[0]])+0.1]

        fig.set_figwidth(15)
        plt.savefig(plot_dir + dataset + ' ' + SNN_result_set + ' ML accuracies as Training epochs Increase', bbox_inches='tight')
        plt.show()

channel_selection = ["", "CSP/", "Pearson Correlation/"]
ML_list = ['Random forest', 'KNN', 'SVM']
highest_total_average_per_feature_extractor = {}
for dataset in datasets:
    highest_total_average_per_feature_extractor[dataset] = {}
    for feature_extractor in feature_extractors:
        for ML in ML_list:
            for channel in channel_selection:
                model_dir = 'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/' + dataset + ' - ' + ML + ' results/' + channel
                highest_filename, highest_average = data_analyser(model_dir, filter=feature_extractor)
                try:
                    if highest_total_average_per_feature_extractor[dataset][feature_extractor][0] < highest_average:
                        highest_total_average_per_feature_extractor[dataset][feature_extractor] = (highest_average, model_dir + highest_filename)
                except:
                    highest_total_average_per_feature_extractor[dataset][feature_extractor] = (highest_average, model_dir + highest_filename)
plot_dict = {}
for trial_dataset in datasets:
    plot_dict[trial_dataset] = {}
    for feature_extractor in feature_extractors:
        numpy_features = np.genfromtxt(highest_total_average_per_feature_extractor[trial_dataset][feature_extractor][1], delimiter=",", dtype=float)
        participants = np.genfromtxt(highest_total_average_per_feature_extractor[trial_dataset][feature_extractor][1], delimiter=",", dtype=str)[0, 1:-1]
        tasks = np.genfromtxt(highest_total_average_per_feature_extractor[trial_dataset][feature_extractor][1], delimiter=",", dtype=str)[1:, 0]
        average_list = numpy_features[1:, -1]
        numpy_features = numpy_features[1:, 1:-1]*100
        plot_dict[trial_dataset][feature_extractor] = np.mean(numpy_features, axis=0)
        plot_dict[trial_dataset][feature_extractor + "_std"] = np.std(numpy_features, axis=0)
        plot_dict[trial_dataset][feature_extractor + "_tasks"] = np.mean(numpy_features, axis=1)
        plot_dict[trial_dataset][feature_extractor + "_tasks_std"] = np.std(numpy_features, axis=1)
        plot_dict[trial_dataset][feature_extractor + "_mean_std"] = np.std(average_list * 100)
        plot_dict[trial_dataset][feature_extractor + "_av_list"] = average_list
        plot_dict[trial_dataset][feature_extractor + "_parts"] = np.mean(numpy_features, axis=0)
        plot_dict[trial_dataset][feature_extractor + "_parts_std"] = np.std(numpy_features, axis=0)
        plot_dict[trial_dataset][feature_extractor + "_parts_labels"] = participants

x = np.arange(len(tasks))  # the label locations
x = x * 1.25
width = 0.15  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - 2.5 *width, plot_dict[datasets[0]][feature_extractors[0] + "_tasks"], width, label=datasets[0] + " " + feature_extractors[0].replace('_',''),
                yerr=plot_dict[datasets[0]][feature_extractors[0] + "_tasks_std"],capsize=3)
rects2 = ax.bar(x - 1.5* width, plot_dict[datasets[1]][feature_extractors[0] + "_tasks"], width, label=datasets[1] + " " + feature_extractors[0].replace('_',''),
                yerr=plot_dict[datasets[1]][feature_extractors[0] + "_tasks_std"],capsize=3)
rects3 = ax.bar(x - .5*width, plot_dict[datasets[0]][feature_extractors[1] + "_tasks"], width, label=datasets[0] + " " + feature_extractors[1],
                yerr=plot_dict[datasets[0]][feature_extractors[1] + "_tasks_std"],capsize=3)
rects4 = ax.bar(x + .5*width, plot_dict[datasets[1]][feature_extractors[1] + "_tasks"], width, label=datasets[1] + " " + feature_extractors[1],
                yerr=plot_dict[datasets[1]][feature_extractors[1] + "_tasks_std"],capsize=3)
rects5 = ax.bar(x + 1.5*width, plot_dict[datasets[0]][feature_extractors[2] + "_tasks" ], width, label=datasets[0] + " " + feature_extractors[2],
                yerr=plot_dict[datasets[0]][feature_extractors[2] + "_tasks_std"],capsize=3)
rects6 = ax.bar(x + 2.5*width, plot_dict[datasets[1]][feature_extractors[2] + "_tasks" ], width, label=datasets[1] + " " + feature_extractors[2],
                yerr=plot_dict[datasets[1]][feature_extractors[2] + "_tasks_std"],capsize=3)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage Accuracy (%)')

#ax.set_title('Comparison across FEIS and Kara One datasets on Top Performing ML Result Comparison per Task')
ax.set_xticks(x, tasks)
ax.legend(loc=(1.01, 0.25))

ax.set_xlabel('Tasks')

fig.tight_layout()
plt.axis([None, None, 35, 85])  # max(plot_dict[data_types[0]])+0.1]

# ax.bar_label(rects1, padding=2, fmt='%.2f')
# ax.bar_label(rects2, padding=2, fmt='%.2f')
# ax.bar_label(rects3, padding=2, fmt='%.2f')
# ax.bar_label(rects4, padding=2, fmt='%.2f')
# ax.bar_label(rects5, padding=2, fmt='%.2f')
# ax.bar_label(rects6, padding=2, fmt='%.2f')


fig.set_figwidth(20)
plt.savefig(plot_dir + 'Comparison across FEIS and Kara One datasets on Top Performing ML Result Comparison per Task.png', bbox_inches='tight')
plt.show()


for dataset in datasets:
    x = np.arange(len(plot_dict[dataset][feature_extractors[0] + "_parts_labels"]))# the label location
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width, plot_dict[dataset][feature_extractors[0] + "_parts"], width, label=str(feature_extractors[0].replace('_', '')),
                    yerr=plot_dict[dataset][feature_extractors[0] + "_parts_std"], capsize=3)
    rects2 = ax.bar(x, plot_dict[dataset][feature_extractors[1] + "_parts"], width, label=feature_extractors[1],
                    yerr=plot_dict[dataset][feature_extractors[1] + "_parts_std"], capsize=3)
    rects3 = ax.bar(x + width, plot_dict[dataset][feature_extractors[2] + "_parts" ], width, label=feature_extractors[2],
                    yerr=plot_dict[dataset][feature_extractors[2] + "_parts_std"], capsize=3)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage Accuracy (%)')

    #ax.set_title('Comparison across FEIS and Kara One datasets on Top Performing ML Result Comparison per Task')
    ax.set_xticks(x, plot_dict[dataset][feature_extractors[0] + "_parts_labels"])
    ax.legend(loc=(1.01, 0.4))

    ax.set_xlabel('Participants')

    fig.tight_layout()
    plt.axis([None, None, 35, 85])  # max(plot_dict[data_types[0]])+0.1]

    # ax.bar_label(rects1, padding=2, fmt='%.2f')
    # ax.bar_label(rects2, padding=2, fmt='%.2f')
    # ax.bar_label(rects3, padding=2, fmt='%.2f')
    # ax.bar_label(rects4, padding=2, fmt='%.2f')
    # ax.bar_label(rects5, padding=2, fmt='%.2f')
    # ax.bar_label(rects6, padding=2, fmt='%.2f')


    fig.set_figwidth(10)
    plt.savefig(plot_dir + 'Comparison across ' + dataset + ' dataset on Top Performing ML Result Comparison per Participant.png', bbox_inches='tight')
    plt.show()

ML_list = ['Random Forest', 'KNN', 'SVM']
for ML in ML_list:
    plot_dict = {}
    for dataset in datasets:
        plot_dict[dataset] = {}
        for feature in feature_extractors:
            numpy_features = np.genfromtxt(highest_per_ML_per_DS_per_feature[dataset][ML][feature][1], delimiter=",", dtype=float)
            participants = np.genfromtxt(highest_per_ML_per_DS_per_feature[dataset][ML][feature][1], delimiter=",", dtype=str)[0, 1:-1]
            tasks = np.genfromtxt(highest_per_ML_per_DS_per_feature[dataset][ML][feature][1], delimiter=",", dtype=str)[1:, 0]
            numpy_features = numpy_features[1:, 1:-1]*100
            plot_dict[dataset][feature] = np.mean(numpy_features, axis=0)
            plot_dict[dataset][feature + "_std"] = np.std(numpy_features, axis=0)
            plot_dict[dataset][feature + "_tasks"] = np.mean(numpy_features, axis=1)
            plot_dict[dataset][feature + "_tasks_std"] = np.std(numpy_features, axis=1)

    x = np.arange(len(tasks))  # the label locations
    x = x * 3.5
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - 2.5*width, plot_dict[datasets[0]]['_linear__tasks'], width, label=datasets[0] + ' Linear',
                    yerr=plot_dict[datasets[0]]['_linear__tasks' + "_std"],capsize=3)
    rects2 = ax.bar(x - 1.5*width, plot_dict[datasets[1]]['_linear__tasks'], width, label=datasets[1] + ' Linear',
                    yerr=plot_dict[datasets[1]]['_linear__tasks' + "_std"],capsize=3)
    rects3 = ax.bar(x - 0.5*width, plot_dict[datasets[0]]['MFCC_tasks'], width, label=datasets[0] + ' MFCC',
                    yerr=plot_dict[datasets[0]]['MFCC_tasks'+ "_std"],capsize=3)
    rects4 = ax.bar(x + 0.5*width, plot_dict[datasets[1]]['MFCC_tasks'], width, label=datasets[1] + ' MFCC',
                    yerr=plot_dict[datasets[1]]['MFCC_tasks' + "_std"], capsize=3)
    rects5 = ax.bar(x + 1.5*width, plot_dict[datasets[0]]['SNN_tasks'], width, label=datasets[0] + ' SNN',
                    yerr=plot_dict[datasets[0]]['SNN_tasks' + "_std"], capsize=3)
    rects6 = ax.bar(x + 2.5*width, plot_dict[datasets[1]]['SNN_tasks'], width, label=datasets[1] + ' SNN',
                    yerr=plot_dict[datasets[1]]['SNN_tasks' + "_std"], capsize=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage Accuracy (%)')

    #ax.set_title('Top Performing ' + trial_dataset + ' ML Result Comparison per Participant')
    ax.set_xticks(x, tasks)
    ax.legend(loc=(1.01, 0.4))

    fig.tight_layout()
    #ax.bar_label(rects1, padding=2, fmt='%.2f')
    #ax.bar_label(rects2, padding=2, fmt='%.2f')
    #ax.bar_label(rects3, padding=2, fmt='%.2f')

    ax.set_xlabel('Tasks')

    fig.tight_layout()

    plt.axis([None, None, 35, 85])  # max(plot_dict[data_types[0]])+0.1]

    fig.set_figwidth(15)
    plt.savefig(plot_dir + 'Comparison of Kara One Vs FEIS across the feature selection methods and ML algorithm ' + ML , bbox_inches='tight')
    plt.show()

