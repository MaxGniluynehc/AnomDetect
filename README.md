# Anomaly Detection Methods

This repo summarizes some anomaly detecton methods from the literature and website. 

In the [model](Anomaly_Detection_Methods_1/AnomDetect/model) folder, we have: 
- [LSTM_VAE](Anomaly_Detection_Methods_1/AnomDetect/model/LSTM_VAE.py): applys the module introduced by D. Park, et al., in *A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder*
- [BiLSTM](Anomaly_Detection_Methods_1/AnomDetect/model/BiLSTM.py) and [Clustering_Pycaret](Anomaly_Detection_Methods_1/AnomDetect/model/Clustering_Pycaret.py):
  are retrieved from this [repo](https://github.com/Wb-az/timeseries-sensor-anomaly-detection/blob/main/multiple_timeseries_anomaly.ipynb).
- [ConvLSTM](Anomaly_Detection_Methods_1/AnomDetect/model/ConvLSTM.py): is retrieved from this [Kaggle notebook](https://www.kaggle.com/code/kyklosfraunhofer/anomaly-detection-on-nasa-bearings-dataset/notebook).

This repo is based on the NASA bearing dataset from this [Kaggle webpage](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset). 

To preprocess the data, please refer to the [data_prep](Anomaly_Detection_Methods_1/AnomDetect/model) folder. For model training and evaluation, please refer to the [train](Anomaly_Detection_Methods_1/AnomDetect/train) folder. 

## Some open source dataset for Lift Anomaly Detection 
1. [Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset/data)
  - Mostly relevant to our project. But it lacks explanation of some features and anomaly labels. The target of this dataset is to predict the absolute value of vibration, which (they claimed) is an indicator of anomaly. 
2. [Machine Failure Prediction using Sensor data](https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data)
  - Similar to our project. Dataset well documented, and contains labels of machine failures. 
3. [HVAC (Heating, Ventilation and Air Conditioning) Systems Anomaly Detection using ML](https://www.kaggle.com/datasets/shashwatwork/hvac-systems-anomaly-detection-using-ml)
  - Still relevant to IoT. Dataset well documented, but unlabeled. 
4. [Controlled Anomalies Time Series (CATS) Dataset](https://www.kaggle.com/datasets/patrickfleith/controlled-anomalies-time-series-dataset?select=SLXENGDE-CATS-DDD+-+Controlled+Anomalies+Time+Series+-+02.00.pdf)
  - A comprehensive dataset with multi-dimensional features, long-enough time stamps, and various types of anomalies. 
