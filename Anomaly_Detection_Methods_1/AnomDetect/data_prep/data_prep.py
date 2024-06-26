import os
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.fft import fft
from scipy.signal import detrend
import torch as tc
# from torch.utils.data.pipdataloader import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader

"""
Data Preprocessing for LSTM_VAE, BiLSTM and Clustering with Pycaret models. 
"""


def dir_list(path=None):
    """
    :param path: path to the dataset
    :return: an order directory list
    """
    list_dir = natsorted(os.listdir(path))
    return list_dir


def concat_raw_data(path=None, csv_path=None, dataset=None, fourier_tr=True,
                    detrends=True):
    """
    :param detrends: boolean to detrend the signal
    :param fourier_tr: boolean tranfor singal with furier transform
    :param path: a path to the dataset file
    :param dataset: number of the dataset to be processes from the three dataset available
    :param csv_path: path save the contenated files into a csv file
    :return: data dataset with average and std from each file at each time step
    """

    list_dir = dir_list(path)

    col_dual = list()
    for b in range(0, 4):
        b1 = f'b{b + 1}_ch{b * 2 + 1}'
        b2 = f'b{b + 1}_ch{b * 2 + 2}'
        col_dual.extend([b1, b2])

    col_names = [f'b{i + 1}_ch{i + 1}' for i in range(0, 4)]

    dataset_dict = {}

    for i, f in enumerate(list_dir):
        temp_df = pd.read_csv(os.path.join(path, f), sep='\t', header=None)
        if len(temp_df.columns) == 8:
            temp_df.columns = col_dual
        else:
            temp_df.columns = col_names
        temp_df.insert(0, 'date', len(temp_df) * [f])
        dataset_dict[f] = temp_df

    df = pd.concat(list(dataset_dict.values()), ignore_index=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')

    os.makedirs(csv_path, exist_ok=True)

    if fourier_tr:
        return fourier_transforms(df, path=csv_path, dataset=dataset, detrends=detrends)

    else:
        fname = os.path.join(csv_path, f'concat_dataset_{dataset}.csv')

        return df.to_csv(fname)


def fourier_transforms(data_frame, path=None, dataset=1, detrends=True):
    """
    :param data_frame: datafram with the concatenated raw data
    :param path: a path to store the csv file
    :param dataset: number of dataste processed
    :param detrends: a boolean to detrend before applyin fourier transformations
    :return: save dataframe as csv
    """

    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f'fft_dataset_{dataset}.csv')
    df_fft = data_frame.copy()

    for col in df_fft.columns:
        if detrends:
            fft_col = fft(detrend(df_fft[col].values))
        else:
            fft_col = fft(df_fft[col].values)

        df_fft[col] = np.abs(fft_col)

    return df_fft.to_csv(fname)


def average_signal_dataset(path=None, dataset=1, csv_path=None):
    """
    :param path:  str a path to the dataset file
    :param dataset:  int the number of the dataset to be process
    :param csv_path:  str a path to save the datframes to csv file
    :return: data dataset with average and std from each file at each time step
    """
    list_dir = dir_list(path)

    if dataset == 1:
        col_names = list()
        for b in range(0, 4):
            b1 = f'b{b + 1}_ch{b * 2 + 1}'
            b2 = f'b{b + 1}_ch{b * 2 + 2}'
            col_names.extend([b1, b2])
    else:
        col_names = [f'b{i + 1}_ch{i + 1}' for i in range(0, 4)]

    dataset_dict = {}

    for file in list_dir:
        temp_df = pd.read_csv(os.path.join(path, file), sep='\t', header=None)
        # mean_std_values = np.append(temp_df.abs().mean().values, temp_df.abs().std().values)
        mean = temp_df.abs().mean().values
        dataset_dict[file] = mean

    df = pd.DataFrame.from_dict(dataset_dict, orient='index', columns=col_names)
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')
    os.makedirs(csv_path, exist_ok=True)

    fname = os.path.join(csv_path, f'avg_concat_dataset_{dataset}.csv')

    return df.to_csv(fname)


def generate_datasets(datasets_dict=None, csv_path=None):
    for i, (k, v) in enumerate(datasets_dict.items()):
        dataset_num = i + 1
        average_signal_dataset(v, dataset=dataset_num, csv_path=csv_path)
        # concat_raw_data(v, csv_path, dataset=dataset_num, fourier_tr=True, detrends=False)


# =================== Generate avg_dfs ======================= #
"""
Create average signals of each bearing for each file. 
Raw datasets: Totaling N files (time slots of 5 or 10min length), each has 20480 time points.  
"""

raw_data_PATH = "../datasets/NASA_bearing"
    # "/Users/maxchen/Documents/Working/ASTRI——Summer2024/datasets/NASA_bearing"
    #
datasets = {'dataset_path1': raw_data_PATH + '/1st_test/1st_test',
            'dataset_path2': raw_data_PATH + '/2nd_test/2nd_test',
            'dataset_path3': raw_data_PATH + '/3rd_test/4th_test/txt'}

concat_data = False
csv_data = os.path.join(os.getcwd(), 'csv_data')
os.makedirs(csv_data, exist_ok=True)

if len(dir_list(csv_data)) == 0:
    csv_dir = os.path.join(os.getcwd(), 'csv_data')
    generate_datasets(datasets, csv_path=csv_dir)

# fft_dataset = [d for d in dir_list(csv_data) if 'fft' in d]
avg_dataset = [d for d in dir_list(csv_data) if not 'fft' in d]


def read_concat_data(path=None, fname=None):
    df = pd.read_csv(os.path.join(path, fname), index_col=0)
    df.index = pd.to_datetime(df.index)
    return df



avg_df1 = read_concat_data(csv_data, fname=avg_dataset[0])
# print(f'Dataset size: {len(avg_df1)}')
# avg_df1.describe()

avg_df2 = read_concat_data(csv_data, fname=avg_dataset[1])
# print(f'Dataset size: {len(avg_df2)}')
# avg_df2.describe()

avg_df3 = read_concat_data(csv_data, fname=avg_dataset[2])
# print(f'Dataset size: {len(avg_df3)}')
# avg_df3.describe()

csv_files = os.path.join(os.getcwd(), 'csv_files')
os.makedirs(csv_files, exist_ok=True) # directory created to save BiLSTM and Pycaret results


# ====================== torch.Dataset for LSTM_VAE ======================= #
def split_train_test(df, test=0.3):
    n = df.shape[0]
    i = int(n * test)
    train_df = df.iloc[:-i, :]
    test_df = df.iloc[-i:, :]
    return train_df, test_df

class NASABearingDataSet(Dataset):
    def __init__(self, df, seq_len):
        self.data = tc.tensor(df.values).to(dtype=tc.float32)  # [total len, data_dim]
        self.seq_len = seq_len  # in avg_df2, time interval is 10min, seq_len=30 means 5hr window

    def __len__(self):
        return self.data.size(0) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx: (idx + self.seq_len), :]





if __name__ == '__main__':
    datasets = {'dataset_path1': './ims_bearing/1st_test/1st_test',
                'dataset_path2': './ims_bearing/2nd_test/2nd_test',
                'dataset_path3': './ims_bearing/3rd_test/4th_test/txt'}

    csv_dir = os.path.join(os.getcwd(), 'csv_data')
    os.makedirs(csv_dir, exist_ok=True)

    # generate_datasets(datasets, csv_path=csv_dir)