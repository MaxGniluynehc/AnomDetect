from data_prep.data_prep_ConvLSTM import (X_train_series, X_test_series, y_train, y_test, bins, seq_len,
                                data_about_tests, generate_sequences_pad_front)
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
matplotlib.use("TkAgg")


# get from https://www.kaggle.com/code/kyklosfraunhofer/anomaly-detection-on-nasa-bearings-dataset/notebook


subsequences = 5    # number of subsequences look at in 3D Convolutional layers
# seq_len=30
timesteps = seq_len//subsequences   #timesteps left in each subsequence
X_train_series_sub = np.array([X_train_series[i].reshape((X_train_series[i].shape[0],
        subsequences, timesteps,4,X_train_series[i].shape[-1]//4,1)) for i in range(4)], dtype=object) # generate X_train with sub sequences
X_test_series_sub = np.array([X_test_series[i].reshape((X_test_series[i].shape[0],
        subsequences, timesteps,4,X_train_series[i].shape[-1]//4,1))for i in range(3)], dtype=object)  # generate X_test with sub sequences


print('Train set shape', [X_train_series_sub[i].shape for i in range(4)])
print('Test set shape', [X_test_series_sub[i].shape for i in range(3)])


# ================= Building ConvLSTM ==================== #
test=3
cnn_lstm = Sequential()
cnn_lstm.add(TimeDistributed(Conv3D(filters=70, kernel_size=(1,2,3), activation='relu'),
                input_shape=(X_train_series_sub[test].shape[1:])))
cnn_lstm.add(TimeDistributed(MaxPooling3D(pool_size=(X_train_series_sub[test].shape[2], 2,3))))
cnn_lstm.add(TimeDistributed(Flatten()))
cnn_lstm.add(Dropout(0.3))
cnn_lstm.add(LSTM(50))

cnn_lstm.add(Dense(y_train[test].shape[-1]))
cnn_lstm.compile(loss='mse', optimizer="adam")

cnn_lstm.summary()


# =================== Training ================== #
# test=3
cnn_lstm.fit(X_train_series_sub[test] ,y_train[test] , epochs=150, batch_size=16, validation_split=0.2, verbose=1, shuffle = True,
callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40, verbose=0, mode='min',
                                   restore_best_weights=True)])



# ================== Evaluation ===================== #
def plot(y_true, bins, data_about_tests, y_pred=np.array([]), anomalies1 = np.array([]), anomalies2 = np.array([]), cols = 4, seperate_legend=False):
    '''
    Plots the data in seperate plots if no other constraints are set this will be a grid of 4 columns and 7 rows.
    Each column represents a bearing from one a particular test run. Each row represents an engineered feature.
    :param y_true: true data that was measured in the test run will be displayed in blue
    :param bins: features that were engineered. In this case it will be [max, std, 0Hz-250Hz, ... , 5000Hz-10000Hz]
    :param data_about_tests: Dictionary with data about test-run containing the key "broken"
    :param y_pred: predicted data will be displyed in orange
    :param anomalies1: boolean array containing which True if an anomaly1 alarm was fired at that position
    :param anomalies2: array for each bearing containing the anomaly2 scores
    :param cols: number of columns you want to plot should be equivalent with number of bearings
    :param seperate_legend: if you want a seperate legend outside of the plot set this to True
    :return fig: figure containing all the subplots if seperate_legend figure containing only the legend is also returned
    '''
    print("Plotting")
    rows = y_true.shape[1] // cols

    fig = pylab.figure(figsize=(cols*4,rows*3))
    if seperate_legend:
        figlegend = pylab.figure(figsize=(3,2))

    axs = np.empty((rows,cols), dtype=object)
    axs2 = np.empty((rows,cols), dtype=object)

    y_position_of_title=0.85
    labels=[]
    lines=[]
    ano1 = True
    for k in range(y_true.shape[-1]):
        i = k%cols
        j = k//cols
        axs[j,i] = fig.add_subplot(rows, cols, k+1 , sharey = axs[j,0], )
        axs[j,i] .tick_params(axis='y', labelcolor="tab:blue")

        lines.append(axs[j,i].plot(y_true[:,k])[0])
        labels.append("True_values" if j == 0 and i ==0 else "_True_values")

        if y_pred.size!=0:
            lines.append(axs[j,i].plot(y_pred[:,k])[0])
            labels.append("Predicted_values" if j == 0 and i ==0 else "_Predicted_values")

        if anomalies1.size!=0:
            w = 1.5
            for xc in np.arange(anomalies1.shape[0])[anomalies1[:,k]]:
                lines.append(axs[j,i].axvspan(xc-w, xc+w,  facecolor="red", ymax=1, alpha = 0.4))
                labels.append("Anomaly level 1 alarm" if ano1 else "_Anomaly1")
                ano1=False
        if anomalies2.size!=0:
            axs2[j,i] = axs[j,i] .twinx()  # instantiate a second axes that shares the same x-axis
            axs2[j,i].get_shared_y_axes().joined(axs2[j,i], axs2[j,0])
            color = 'black'
            lines.append(axs2[j,i].plot((anomalies2[:,k%cols]), color = color)[0])
            axs2[j,i].tick_params(axis='y', labelcolor=color)
            labels.append("Anomaly level 2 score" if j==0 and i ==0 else "_Anomaly2")
        if j == 0:
            if i in data_about_tests["broken"]:
                axs[j,i].set_title("Bearing "+ str(i)+"\nBreaks in the end\n\n Maximum Values", y = y_position_of_title)
            else:
                axs[j,i].set_title("Bearing "+ str(i)+"\n\n\n Maximum Values", y = y_position_of_title)
        elif j == 1:
            axs[j,i].set_title("Standard Deviation", y = y_position_of_title)
        else:
            axs[j,i].set_title(str(bins[j-2])+"Hz-"+str(bins[j-2+1])+"Hz", y = y_position_of_title)
    if seperate_legend:
        figlegend.legend(lines,labels,   "center" )
        return fig, figlegend
    else:
        fig.legend(lines, labels,  bbox_to_anchor = (0.8, 0.96))
        return fig


def evaluate(model, X, y_true, test_size, test_number,slice_to_plot=np.s_[:], anomaly_1_factor = 5, window_size=30,
            show_y_pred=True, show_anomaly1 = True, show_anomaly2=True, cols=4):
    '''
    calculates the error between predicted and true values. Then calculates a boundary how much
    the error may differ from the true value. If the error exceeds that boundary a level one anomaly alarm is stored.
    Then calculates a level two anomly score with the percentage of level one alarms in last 30 timesteps.
    :param model: machine learning model used for prediction
    :param X: X_values that get fed into the model for prediction
    :param y_true: true labels for the data in X
    :param test_size: the size of the test set, important because the the boundary is only calculated on the train_set
    :param test_number: which test-run the data is from. Can only be 0,1,2
    :param slice_to_plot: if you only want to plot a certain slice of the plots. e.g. if you want to plot only the last
                        1000 values set this to [-1000:] or if you only want to plot bearing 0 set this to [:,[0,4,8,12,16,20,24]]
                        and also set cols to 1
    :param anomaly_1_factor: by how the standard deviation is multiplied to calculate the boundary
    :param window_size: size of the window over which the level two anomaly score is calculated
    :param show_y_pred: wether to show y_pred in the plots
    :param show_anomaly1: wether to show level one anomalies in the plots
    :param show_anomaly2: wether to show the level anomaly score in the plots
    :param cols: how many columns you want to plot in, should be number of bearings you want to plot
    :return fig: figure containing the subplots
    '''
    global data_about_tests
    train_size = int(X.shape[0]*(1-test_size))
    y_pred = model.predict(X, batch_size=10000)
    error = y_true-y_pred # same size as y_true, (length of test, num of bearings * num of engineered features)
    boundary = error[:train_size].mean(axis=0) + anomaly_1_factor*error[:train_size].std(axis=0)
    anomalies = error**2 > boundary # compare squared error with the boundary over each time step, (length of test, num of bearings * num of engineered features)
    anomalies2, _ = generate_sequences_pad_front(anomalies[slice_to_plot],window_size) #Always look at anomalies in window
    anomalies2 = anomalies2.reshape((anomalies2.shape[0],window_size, anomalies2.shape[-1]//cols,cols)) # (length of test, window size, num of engineered features, num of bearings)

    anomalies2 = anomalies2.mean(axis=1)
    anomalies2 = anomalies2.mean(axis=1) # (length of test, num of bearings)
    print("2nd level alarm over 0.5:")
    [print(np.where(anomalies2[:,i]>0.5)[0][:10]) for i in range(cols)]
    fig = plot(y_true[slice_to_plot], bins, data_about_tests[test_number], y_pred[slice_to_plot] if show_y_pred else np.array([]),
        anomalies1 =  anomalies[slice_to_plot] if show_anomaly1 else np.array([]) , anomalies2 = anomalies2[:] if show_anomaly2 else np.array([]), cols=cols)
    fig.suptitle(data_about_tests[test_number]["name"]+"_test\nstd_factor: "+str(anomaly_1_factor)+"\nwindow_size:"+str(window_size), fontsize=20)
    return fig


test=2
test_size = 0.6
X = np.concatenate((X_train_series_sub[test],X_test_series_sub[test]))
y_true = np.concatenate((y_train[test],y_test[test]))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fig = evaluate(cnn_lstm, X, y_true, test_size ,test, slice_to_plot=np.s_[:],anomaly_1_factor=3, window_size=30,
                show_y_pred=True, show_anomaly1 = True, show_anomaly2 = True)


