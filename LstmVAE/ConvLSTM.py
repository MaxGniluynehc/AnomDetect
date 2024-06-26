from data_preprocessing import X_train_series, X_test_series, y_train, y_test

import matplotlib
import matplotlib.pyplot as plt
import pylab

import numpy as np
import os
import warnings
import pandas as pd
import pickle
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed,Flatten, Conv3D, MaxPooling3D
from tensorflow.keras import Sequential


subsequences = 5    # number of subsequences look at in 3D Convolutional layers
timesteps = seq_len//subsequences   #timesteps left in each subsequence
X_train_series_sub = np.array([X_train_series[i].reshape((X_train_series[i].shape[0],
        subsequences, timesteps,4,X_train_series[i].shape[-1]//4,1)) for i in range(4)]) # generate X_train with sub sequences
X_test_series_sub = np.array([X_test_series[i].reshape((X_test_series[i].shape[0],
        subsequences, timesteps,4,X_train_series[i].shape[-1]//4,1))for i in range(3)])  # generate X_test with sub sequences

print('Train set shape', [X_train_series_sub[i].shape for i in range(4)])
print('Test set shape', [X_test_series_sub[i].shape for i in range(3)])