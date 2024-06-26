
# get from https://github.com/Wbpython-az/timeseries-sensor-anomaly-detection/blob/main/multiple_timeseries_anomaly.ipynb
import os, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import warnings
# warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt

from data_prep.data_prep import * # (concat_raw_data, average_signal_dataset, dir_list)
from models.plots_Pycaret_BiLSTM import *

matplotlib.use("TkAgg")
np.random.seed(1234)

# ================== Load Data ======================= #
# raw_data_PATH = "/Users/maxchen/Documents/Working/ASTRI——Summer2024/datasets/NASA_bearing"
#
# datasets = {'dataset_path1' : raw_data_PATH + '/1st_test/1st_test',
#            'dataset_path2' : raw_data_PATH + '/2nd_test/2nd_test',
#            'dataset_path3' : raw_data_PATH + '/3rd_test/4th_test/txt'}
#
# csv_data = os.path.join(os.getcwd(), 'csv_data')
# os.makedirs(csv_data, exist_ok=True)
# datasets.items()
# concat_data = False
# def generate_datasets(datasets_dict=datasets, csv_path=csv_data):
#     for i, (k, v) in enumerate(datasets_dict.items()):
#         dataset_num = i + 1
#         average_signal_dataset(v, dataset=dataset_num, csv_path=csv_path)
#         # concat_raw_data(v, csv_path, dataset=dataset_num, fourier_tr=False, detrends=False)
#
#
# if concat_data:
#     generate_datasets(datasets)
#
#
# fft_dataset = [d for d in dir_list(csv_data) if 'fft' in d]
# avg_dataset = [d for d in dir_list(csv_data) if not 'fft' in d]
#
# def read_concat_data(path=None, fname=None):
#     df = pd.read_csv(os.path.join(path, fname), index_col=0)
#     df.index = pd.to_datetime(df.index)
#     return df
#
# avg_dataset
#
# avg_df1 = read_concat_data(csv_data, fname=avg_dataset[1])
# print(f'Dataset size: {len(avg_df1)}')
# avg_df1.describe()
#
# avg_df2 = read_concat_data(csv_data, fname=avg_dataset[2])
# print(f'Dataset size: {len(avg_df2)}')
# avg_df2.describe()
#
# avg_df3 = read_concat_data(csv_data, fname=avg_dataset[3])
# print(f'Dataset size: {len(avg_df3)}')
# avg_df3.describe()

# ================================================ EDA ============================================= #
save_plots_dir = "../Anomaly_Detection_Methods/plots/"
os.makedirs(save_plots_dir, exist_ok=True)

view_all(avg_df2, plot_name='avg_dataset2')
view_all(avg_df3, plot_name='avg_dataset3')
plt.savefig(save_plots_dir+"EDA/avg_df2")

view_per_channel(avg_df2, plot_name='view_per_channel_avgdf2')
plt.savefig(save_plots_dir+"EDA/avg_df2_per_channel")

view_per_channel(avg_df3, plot_name='view_per_channel_avgdf3')


# ========================= Unsupervised Anomaly detection using PyCaret ======================= #
csv_files = os.path.join(os.getcwd(), 'csv_files')
os.makedirs(csv_files, exist_ok=True)

from pycaret.anomaly import *

exp = setup(avg_df2, session_id=123, log_experiment=False, use_gpu=False) # initial setup, train using avg_df2
exp.get_config()
exp.dataset # this is avg_df2

# Show avaialable models
all_anomaly_models = models()
all_anomaly_models
selected_models = ['cluster', 'histogram', 'iforest', 'knn', 'mcd', 'svm'] # picked 6 clustering methods
# `cluster`: Clustering-Based Local outlier
# `histogram`: Histogram-based Outlier Detection
# `iforest`: Isolation Forest
# `knn`: K-Nearest Neighbors
# `mcd`: Minimum Covariance Determinant
# `svm`: One Class SVM


sns.set(rc={'figure.figsize': (4, 3)})
sns.set_style('whitegrid')
sns.distplot(avg_df2.values[:,:], bins=30, kde=True)
plt.grid(color ='k', ls= '--', lw=0.5)
plt.xlabel('Avg signal')
plt.title('Dataset distribution avg_df2')
plt.show()


def train_model(model_name=None, fraction=None):
    print(f'Assigning labels: {model_name} model')
    model = create_model(model_name, fraction=fraction)
    model_anomalies = assign_model(model) # add anomaly label and anomaly score to the original df
    unique_labels = model_anomalies.Anomaly.unique()
    print(f'Unique labels: {unique_labels}')
    save_model(model, f'{model_name}_pipeline')
    return model_anomalies


def main(anomly_model_list=None, fraction=None):

    results_dict = {}
    anomaly_scores = {}
    for m in anomly_model_list:
        results = train_model(m, fraction=fraction)
        results_dict[m] = results.Anomaly
        anomaly_scores[m] = results.Anomaly_Score
        results_filter = results[results.Anomaly == 1]
        print(f'Anomalies detected by {m} model')
        print(f'Anomalous readings: {len(results_filter)}')
        print(' ')

    df_anomalies = pd.DataFrame.from_dict(results_dict)
    df_scores = pd.DataFrame.from_dict(anomaly_scores)

    return df_anomalies, df_scores

# train on avg_df2 using selected models
anomaly_df, scores_df = main(selected_models, fraction=0.05) # contamination (prop of outliers) 0.05, to avoid overfitting
# Save results for future comparison
anomaly_df.to_csv(os.path.join(csv_files, 'unsupervised_anomaly_pred.csv'))
scores_df.to_csv(os.path.join(csv_files, 'unsupervised_scores_pred.csv'))

# Visualize the unsupervised anomaly clustering results
print('Anomalies detected by model', anomaly_df.sum(axis=0), sep='\n')
scatter_anomalies_plot(anomaly_df, 'anomalies')

train_anomalies = anomaly_df[(anomaly_df.index >= '2004-02-15') & (anomaly_df.sum(axis=1)>=1)]
print(f'The first anomaly detected from 2004-02-15 was on {train_anomalies.index[0]}')
print(f'Model: {train_anomalies.columns[np.where(train_anomalies.iloc[0].values)[0]][0]}')

# from datetime import datetime
# anomaly_df.loc[datetime(2004,2,18): datetime(2004,2,19), "svm"].values

avg_df2.loc[train_anomalies.index].plot(xlabel='date', ylabel='average signal', figsize=(8,3))
plt.grid(True, lw=0.5)
plt.show()

common_detected_anomaly = train_anomalies[train_anomalies.sum(axis=1) == 6].index
print(f'All models detected anomalies in {len(common_detected_anomaly)} common timestamps')

avg_df2.loc[common_detected_anomaly].plot(xlabel='date', ylabel='average signal', figsize=(8,2.5))
plt.grid(True, lw=0.5)
plt.show()

plot_scores_distribution(scores_df, plot_name='train_scores')

models_ = train_anomalies[train_anomalies.sum(axis=1)>1].iloc[0]
print('The earliest that a failure could have been detected by more than one model was:')
print(f'{models_.iloc[np.where(models_.values ==1)]}')

print('The earliest that the failure could have been detected by all models was:')
print(f'{common_detected_anomaly[0]}')


print(f'Time for preventive maintenace before failure with at least one model detecting an anomaly:')
print(train_anomalies.index[-1] - train_anomalies.index[0])
print(' ')
print(f'Time for preventive maintenace before failure detected by all models:')
print(train_anomalies.index[-1] - train_anomalies.nlargest(1, 'svm').index[0])


# predict anomaly on avg_df3
def predict_anomaly(unseen_data=None, anomly_model_list=None, thresh=0.10, plot_3d=False,
                    fraction=0.05):

    best_models = dict()
    scores = dict()
    best_models_list = list()

    print(f'Unseen data size: {len(unseen_data)}')
    print('-'*58)
    for m in anomly_model_list:
        path_to_pipeline = os.path.join(os.getcwd(), f'{m}_pipeline')
        model = load_model(path_to_pipeline)
        model.fraction = fraction
        predictions = predict_model(model, data=unseen_data)
        anomalies = predictions[predictions.Anomaly == 1]
        print(f'{m} model detected {len(anomalies)} anomalies in the unseen data')
        print('='*58)
        if len(anomalies) >= (thresh * len(predictions)):
            continue
        else:
            if plot_3d:
                plot_model(model, plot='tsne')
            best_models[m] = predictions.Anomaly
            scores[m] = predictions.Anomaly_Score
            best_models_list.append(m)

    df_predictions = pd.DataFrame.from_dict(best_models)
    df_scores = pd.DataFrame.from_dict(scores)

    return  df_predictions, df_scores, best_models_list

unseen_preds, unseen_scores, models_list  = predict_anomaly(avg_df3, selected_models)

unseen_preds.to_csv(os.path.join(csv_files, 'unsupervised_test_anomaly.csv'))
unseen_scores.to_csv(os.path.join(csv_files, 'unsupervised_test_scores.csv'))
plot_scores_distribution(unseen_scores, size=(3, 2), plot_name='test_scores')


test_anomalies = unseen_preds[unseen_preds.sum(axis=1) >= 1]
test_anomalies[(test_anomalies.index >= '2004-04-15') & (test_anomalies.sum(axis=1)>=1)][0:5]
earliest_tail = test_anomalies[(test_anomalies.sum(axis=1)>=1) & (test_anomalies.index >= '2004-04-15')]
if True:
    print(f'The earliest that the failure could have been detected was on :')
    print(f"{test_anomalies[(test_anomalies.index >= '2004-04-15')].index[0]}")
    print(f'Model: {test_anomalies.columns[np.where(test_anomalies.iloc[0].values)[0]][0]}')
    print(f'The earliest that a failure could have been detected by more than one model was:')
    print(f'{earliest_tail[earliest_tail.sum(axis=1)>1].index[0]}')
    print(earliest_tail.columns[np.where(earliest_tail[earliest_tail.sum(axis=1)>1].iloc[0]==1)].values)
    print('The earliest the three models could have detected the failure was')
    print(f'{earliest_tail[earliest_tail.sum(axis=1)==3].index[0]}')
    time_left = earliest_tail.index[-1] - earliest_tail.index[0]
    print(f'Time to prevent failure: {time_left}')

plot_predicted_anomalies(unseen_scores, test_anomalies, plot_name='test_anomaly_prediction')

del unseen_preds
del test_anomalies
del unseen_scores
del earliest_tail
gc.collect()










