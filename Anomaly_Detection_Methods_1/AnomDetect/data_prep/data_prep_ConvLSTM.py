import matplotlib
import matplotlib.pyplot as plt
# import pylab
import numpy as np
import os
# import warnings
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
matplotlib.use("TKagg")
random.seed(0)

# get from https://www.kaggle.com/code/kyklosfraunhofer/anomaly-detection-on-nasa-bearings-dataset/notebook


# =============== Load raw data =============== #
def load_from_csv(DIR):
    '''
    helper function to load all data from one directory
    :param DIR: directory to load data from
    :return x_values: values read from the files
    '''
    filenames = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    samples_per_row = len(pd.read_csv(os.path.join(DIR, filenames[0]), sep="\t", nrows=1).columns)
    x_values = np.zeros([len(filenames), 20480, samples_per_row])
    filenames = sorted(filenames)
    for i,file in  tqdm (enumerate(filenames),  desc="Reading Data",ascii=False, ncols=100, total=len(filenames)):
        x_values[i:i+1,:,:] =np.fromfile(os.path.join(DIR, file), dtype=float, sep=" ").reshape(20480,samples_per_row)
    return x_values


raw_data_PATH = "/Users/maxchen/Documents/Working/ASTRI——Summer2024/datasets/NASA_bearing"


def load_raw_data(force=False):
    '''
    loads the data from all three datasets if the data is not already stored in a pickle, the loaded data is stored in seperate pickles.
    Because loading all three into memory is quite memory intensive.
    :param force: defines, whether the program is forced to reload the data from the csv files and ignore any existing pickles.
    :return : Nothing is returned since the data is stored in a pickle instead
    '''
    DIRS = [raw_data_PATH+'/1st_test/1st_test/',
            raw_data_PATH+'/2nd_test/2nd_test/',
            raw_data_PATH+'/3rd_test/4th_test/txt/',
           ]
    for i in range(3):
        if "test"+str(i)+".pkz" in os.listdir(".") and not force:
            print("test",i, "already loaded.")
            continue
        x = load_from_csv(DIRS[i])
        os.makedirs("ConvLSTM_data", exist_ok=True)
        with open("ConvLSTM_data/" + "test"+str(i)+".pkz", "wb") as file:
            pickle.dump(x, file)


load_raw_data(force=False)


# Compare file_no = 90 vs 970, bearing 1 signal changed dramatically
with open("ConvLSTM_data/test1.pkz", "rb") as file:
    raw_data_test2 = pickle.load(file)
    raw_data_test2.shape
    matplotlib.use("TkAgg")
    fig, ax = plt.subplots(4,1 ,sharex=True, sharey=True)
    file_no = 90
    for i in range(raw_data_test2.shape[-1]):
        ax[i].plot(raw_data_test2[file_no,:,i], label=i)
    ax[0].set_title("File No. {}".format(file_no))
    ax[0].set_ylim([-2,2])


# =============== Feature Engineering ================= #

def binning(bins, raw_data):
    '''
    takes raw_data values and calculates the fft analysis of them. Then divides the fft data into bins and takes the mean of each bin.
    :param bins: bins to divide the data into
    :param raw_data: data to analyse and put into bin afterwards
    :return values: the values for each bin with shape:(length of test, number of bearings, number of bins)
    '''
    values = np.zeros((raw_data.shape[0], raw_data.shape[2], len(bins) - 1))
    for j in tqdm(range(raw_data.shape[2]), desc="Binning Frequencies", ascii=True, ncols=100):
        f = np.fft.fft(raw_data[:, :, j]) # (length of tests, total time steps per test)
        freq = np.fft.fftfreq(20480) * 20000
        for i in range(len(bins) - 1):
            values[:, j, i] += np.absolute(f[:, (freq > bins[i]) & (freq <= bins[i + 1])]).mean(axis=1) # taking average of each bin
    return values


bins = np.array([0,250,1000,2500,5000,10000])           # define bins to sort frequencies into
with open("test1.pkz", "rb") as file:
    raw_data_test2 = pickle.load(file)
    raw_data_test2.shape # (length of tests, total time steps per term, dimension of bearing)
    f = np.fft.fft(raw_data_test2[:,:,0])  # (length of tests, total time steps per term)
    freq = np.fft.fftfreq(20480, d=1/20000)
    values = np.zeros((raw_data_test2.shape[0], raw_data_test2.shape[2], len(bins) - 1))
    for i in range(len(bins) - 1):
        values[:, 0, i] += np.absolute(f[:, (freq > bins[i]) & (freq <= bins[i + 1])]).mean(axis=1)


def feature_engeneering(raw_data):
    '''
    engineers features of raw data: for each bearing following features are engineered: maximums, standard deviation and frequency bins
    beacause test 1 measures two values per bearing every other value is dropped so the tests are compareable.
    :param raw_data: data to engineer features from
    :return values: engineered values with shape (length of test, number of bearings*number of engineered features)
    '''
    if raw_data.shape[2] == 8:
        raw_data = raw_data[:, :, [0, 2, 4, 6]]
    bins = np.array([0, 250, 1000, 2500, 5000, 10000])
    values = binning(bins, raw_data)
    maxs = np.expand_dims(abs(raw_data).max(axis=1), 2) # (length of tests, num of bearings, 1)
    stds = np.expand_dims(raw_data.std(axis=1), 2) # (length of tests, num of bearings, 1)
    values = np.concatenate((maxs, stds, values), axis=2)   # (length of tests, number of bearings, 2 + num of bins)

    values = np.swapaxes(values, 1, 2)
    values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    return values, bins


def load_data(force=False):
    '''
    loads raw_data from pickle files and then engineers feature from that data.
    if data.pkz already exists it just loads this pickle
    :param force: force function to engineer features again even though data.pkz exists
    :return data: data with engineered features for each test has shape:
            ((length of test 1, number of bearings*number of engineered features ),
             (length of test 2, number of bearings*number of engineered features ),
             (length of test 3, number of bearings*number of engineered features ))
    '''
    if "data.pkz" in os.listdir(".") and not force:
        print("Data already engineered. Loading from pickle")
        with open("data.pkz", "rb") as file:
            data = pickle.load(file)
    else:
        data = []
        for i in range(3):
            with open("test" + str(i) + ".pkz", "rb") as file:
                raw_data = pickle.load(file)
            values, bins = feature_engeneering(raw_data)
            data.append(values)
        data = np.array(data, dtype=object)
        with open("data.pkz", "wb") as file:
            pickle.dump(data, file)
    return data

data = load_data(force = False) # contains 3 test profiles, each (len of tests, num of bearing * num of engineered features)


# ============= Prepare data (Rescaling) ================ #
def scale(data, test_size=0.5, minmax=True):
    '''
    scales data with the Standard or MinMaxScaler from Scikit-Learn
    :param data: array to be scaled
    :param test_size: percentage of the dataset to be treated as test set
    :param minmax: use Minmax Scaler instead of standard scaler
    :return values: scaled values
    '''
    l = int(data.shape[0] * (1 - test_size))
    if minmax:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(data[:l])
    values = scaler.transform(data)
    return values


def generate_sequences_no_padding(data, seq_len):
    '''
    generates sequences from data without padding
    :param data: data from which the sequence should be generated
    :param seq_len: length of each sequence (must be int)
    :return X: sequences stored in an array with shape:
            (length of test - sequence length, sequence length, number of bearings*number of features engineered)
    :return y: values to be predicted. Next value after each sequence has shape:
            (length of test - sequence length, number of bearings*number of features engineered)
    '''
    X = np.zeros([data.shape[0] - seq_len, seq_len, data.shape[1]])
    for i in tqdm(range(0, seq_len), desc="Generating sequences", ascii=True, ncols=100):
        X[:, i, :] = data[i:-seq_len + i, :]
    y = data[seq_len:, :]
    return X, y


with open("data.pkz", "rb") as file:
    data = pickle.load(file)[1]
    # data.shape
    seq_len = 30
    X = np.zeros([data.shape[0] - seq_len, seq_len, data.shape[1]])
    for i in tqdm(range(0, seq_len), desc="Generating sequences", ascii=True, ncols=100):
        X[:, i, :] = data[i:-seq_len + i, :]
    y = data[seq_len:, :]


def generate_sequences_pad_front(data, seq_len):
    '''
    generates sequences from data with padding zeros in front
    :param data: data from which the sequence should be generated
    :param seq_len: length of each sequence (must be int)
    :return X: sequences stored in an array with shape:
            (length of test, sequence length, number of bearings*number of features engineered)
    :return y: values to be predicted. Next value after each sequence has shape:
            (length of test, number of bearings*number of features engineered)
    '''
    X = np.zeros([data.shape[0], seq_len, data.shape[1]])
    d = np.pad(data, ((seq_len, 0), (0, 0)), 'constant')  # d[seq_len:, :] == data
    for i in tqdm(range(0, seq_len), desc="Generating sequences", ascii=True, ncols=100):
        X[:, i, :] = d[i:-seq_len + i, :]
    y = data[:, :]
    return X, y


def split_data_set(X, y, test_size=0.5):
    '''
    splits data set into train and test set
    :param X: data to spilt for X_train and X_test
    :param y: data to spilt for y_train and y_test
    :param test_size: percentage of data that should be in the test sets
    :return X_train, X_test, y_train, y_test: X and y values for train and test
    '''
    length = X.shape[0]
    X_train = X[:int(-length * test_size)]
    y_train = y[:int(-length * test_size)]
    X_test = X[int(-length * test_size):]
    y_test = y[int(-length * test_size):]
    return X_train, X_test, y_train, y_test


def prepare_data_series(data, seq_len, test_size=0.5):
    '''
    Generates X_train, X_test, y_train, y_test
    Each of the four arrays contains a dataset for each of the test runs. So if you want to
    train on the first test your data set would be called by X_train[0].
    Addiotanally X_train and y_train have the possibility to train on all test at the same time.
    The values for that are stored in X_train[3] and y_train[3]
    :param data: data to be used for generation of train and test sets
    :param seq_len:  length of each sequence (must be int)
    :param test_size: percentage of data that should be in the test sets
    :return X_train_series, X_test_series, y_train, y_test: Data sets for test and train, the X_values for each are in sequential form.
    '''
    prepared_data = []
    for d in data:
        d = scale(d, test_size=test_size, minmax=True)
        X_series, y_series = generate_sequences_no_padding(d, seq_len)
        prepared_data.append(split_data_set(X_series, y_series, test_size))
    prepared_data = np.array(prepared_data, dtype=object)
    X_train_series = np.array([prepared_data[i][0] for i in range(len(prepared_data))], dtype=object)
    X_test_series = np.array([prepared_data[i][1] for i in range(len(prepared_data))], dtype=object)
    y_train = np.array([prepared_data[i][2] for i in range(len(prepared_data))], dtype=object)
    y_test = np.array([prepared_data[i][3] for i in range(len(prepared_data))], dtype=object)

    # Append combination of all three Training Sets to X_train_series and to y_train
    _X_train_series = [X_train_series[i] for i in range(3)]
    _X_train_series.append(np.concatenate(X_train_series))
    X_train_series = np.array(_X_train_series, dtype=object)

    _y_train = [y_train[i] for i in range(3)]
    _y_train.append(np.concatenate(y_train))
    y_train = np.array(_y_train, dtype=object)

    return X_train_series, X_test_series, y_train, y_test


test_size = 0.6                 # define size of test set
for i in range(3):
    data[i] = scale(data[i], test_size=test_size, minmax = True)  # scale data
bins = np.array([0,250,1000,2500,5000,10000])           # define bins to sort frequencies into
test_names = ["1st", "2nd", "3rd"]                      # test names
data_about_tests = [{"name": "1st", "length": 2156, "broken": [2,3]},
                    {"name": "2nd", "length": 984, "broken": [0]},
                    {"name": "3rd", "length": 6324, "broken": [2]}] # data about test displayed in plots


seq_len=30 # sequence length
X_train_series, X_test_series, y_train, y_test = prepare_data_series(data,seq_len, test_size=test_size) # generate train and test sets
