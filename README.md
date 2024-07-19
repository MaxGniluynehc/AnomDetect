# Anomaly Detection Methods

This repo summarizes some anomaly detecton methods from the literature and website. 

In the [model](Anomaly_Detection_Methods_1/AnomDetect/model) folder, we have: 
- [LSTM_VAE](Anomaly_Detection_Methods_1/AnomDetect/model/LSTM_VAE.py): applys the module introduced by D. Park, et al., in *A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder*
- [BiLSTM](Anomaly_Detection_Methods_1/AnomDetect/model/BiLSTM.py) and [Clustering_Pycaret](Anomaly_Detection_Methods_1/AnomDetect/model/Clustering_Pycaret.py):
  are retrieved from this [repo](https://github.com/Wb-az/timeseries-sensor-anomaly-detection/blob/main/multiple_timeseries_anomaly.ipynb).
- [ConvLSTM](Anomaly_Detection_Methods_1/AnomDetect/model/ConvLSTM.py): is retrieved from this [Kaggle notebook](https://www.kaggle.com/code/kyklosfraunhofer/anomaly-detection-on-nasa-bearings-dataset/notebook).

This repo is based on the NASA bearing dataset from this [Kaggle webpage](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset). 

To preprocess the data, please refer to the [data_prep](Anomaly_Detection_Methods_1/AnomDetect/model) folder. For model training and evaluation, please refer to the [train](Anomaly_Detection_Methods_1/AnomDetect/train) folder. 
