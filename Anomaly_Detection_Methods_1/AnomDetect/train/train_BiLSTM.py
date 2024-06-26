import os, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import warnings
# warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
from data_prep.data_prep import *
from models.plots_Pycaret_BiLSTM import *
from sklearn.preprocessing import MinMaxScaler
from collections import namedtuple
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from models.BiLSTM.BiLSTM import AnomalyBiLSTM
matplotlib.use("TkAgg")
np.random.seed(1234)

# ==========================  processing data ============================= #
scaler = MinMaxScaler()
train_data = torch.tensor(scaler.fit_transform(avg_df2.values), dtype=torch.float32)
test_data = torch.tensor(scaler.transform(avg_df3.values), dtype=torch.float32)

# ============================= Define Training / Evaluation functions =============================== #
def train_one_epoch(**kwargs):
    """
    :param kwargs: hyperparameters dictionary
    :return:training loss for one epoch
    """

    total_loss = 0
    model = kwargs['model'].to(kwargs['device'])
    model.train()

    for true_seq in kwargs['train_loader']:
        kwargs['optimizer'].zero_grad()
        true_seq = true_seq.to(kwargs["device"])
        pred_seq = model(true_seq)
        loss = kwargs['loss_fn'](pred_seq, true_seq)
        loss.backward()
        kwargs['optimizer'].step()
        total_loss += loss.item()

    return total_loss / len(kwargs['train_loader'])


def evaluate(**kwargs):
    """
    :param kwargs: a hyperparameters dictionary
    :return: evalution loss
    """

    total_loss = 0
    model = kwargs['model'].eval()
    with torch.no_grad():
        for true_seq in kwargs['eval_loader']:
            true_seq = true_seq.to(kwargs['device'])
            pred_seq = model(true_seq)
            loss = kwargs['loss_fn'](pred_seq, true_seq)
            total_loss += loss.item()

    return total_loss / len(kwargs['eval_loader'])


def time_minutes(s):
    """
    :param s:  time in seconds
    :return: time in hh:mm:ss format
    """
    return time.strftime('%H:%M:%S', time.gmtime(s))


def train_lstm(**kwargs):
    """
    :param kwargs: a hyperparameters dictionary
    :return: namedtuple - statistics
    """

    Stats = namedtuple('Stats', ['train_loss'])
    train_loss_log = np.zeros(kwargs['epochs'])
    best_loss = np.inf

    epoch_start_time = time.time()
    print('........Training Starting.......')
    for epoch in tqdm(range(kwargs['epochs'])):
        train_loss = train_one_epoch(**kwargs)
        train_loss_log[epoch] = train_loss
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({'epoch': epoch + 1, 'model': kwargs['model'].state_dict(),
                'optim': kwargs['optimizer'].state_dict()},
                os.path.join(kwargs['params'], f"BiLSTM_best_exp_{kwargs['exp']}.pt"))

        if (epoch + 1) % kwargs['logging'] == 0:
            print(' ')
            print('Epoch %d: | train loss: %.4f' % (epoch + 1, train_loss))

    total_time = time.time() - epoch_start_time
    print('Training time {}'.format(time_minutes(total_time)))
    stats = Stats(train_loss=train_loss_log)

    return stats


def predict(model, data_loader, loss_fn=None, device=None):
    preds_log, loss_log = [], []
    with torch.no_grad():
        model = model.eval()
        for seqs_batch in data_loader:
            for seq in seqs_batch:
                seq = seq.to(device)
                seq_pred = model(seq)
                loss = loss_fn(seq_pred, seq)
                preds_log.append(seq_pred.cpu().numpy().flatten())
                loss_log.append(loss.item())
    return preds_log, loss_log


exps =[i for i in range(1, 5)]
network = ['bilstm'] * 4
exp_optim = sorted(['adam', 'adamw'] * 2)
exp_loss = ['mae_loss', 'huber_loss'] * 2

exp_setup = pd.DataFrame({'Exp':exps, 'Model':network, 'Loss': exp_loss, 'Optim':exp_optim },
                         index=exps)
exp_setup.set_index('Exp', drop=True, inplace=True)



# =========================== Training/hyperparam config ============================ #
batch = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bilstm_model = AnomalyBiLSTM(input_size=4, hidden_dim=32, num_layers=1, p=0.1).to(device)
print(bilstm_model)
learning_rate = 0.0002
mae_loss = nn.L1Loss('mean')
huber_loss = nn.SmoothL1Loss('mean')
adam_optim = optim.Adam(bilstm_model.parameters(), lr=learning_rate)
adamw_optim = optim.AdamW(bilstm_model.parameters(), lr=learning_rate)
train_loader = data.DataLoader(train_data, shuffle=False, batch_size=batch)
test_loader = data.DataLoader(test_data, shuffle=False, batch_size=batch)

exp = 4
weights_dir = os.path.join(os.getcwd(), 'saved_models')
os.makedirs(weights_dir, exist_ok=True)

config = {'exp': exp, 'epochs': 50,  'model':bilstm_model, 'loss_fn':mae_loss,
          'optimizer': adamw_optim, 'train_loader': train_loader,'eval_loader': test_loader,
          'device': device, 'logging': 10, 'params' : weights_dir}
if exp in [1, 2]:
    config['optimizer'] = adam_optim
if exp in [2, 4]:
    config['loss_fn'] = huber_loss
elif exp > 4:
    assert exp > 4, 'Experiment not in the experimental setup'

# Run experiments
print(f'Running experiment {exp}', f'{exp_setup.loc[exp]}', sep='\n')
training_metrics = train_lstm(**config)


# loss function visualization
plot_loss(training_metrics, f'bilstm_exp{exp}', figs_dir=plots_dir)



# ============================== Model Evaluation ============================= #
# reconstruction on train dataset
best_params = torch.load(os.path.join(weights_dir, f'best_exp_{exp}.pt'))
bilstm_model.load_state_dict(best_params['model'], strict=False)
train_preds, train_losses = predict(bilstm_model, train_loader, config['loss_fn'], config['device'])

save_plots_dir = "../Anomaly_Detection_Methods/plots/anomaly_scores/"

sns.set(rc={'figure.figsize': (4, 3)})
sns.set_style('whitegrid')
sns.displot(train_losses, bins=30, kde=True)
plt.grid(color ='k', ls= '--', lw=0.5)
plt.xlabel('Loss')
plt.title('Training dataset avg_df2')
plt.xticks(rotation=90)
save_plot(plots_dir, plot_name=f'train_dist_exp{exp}')
plt.show()

# calculate threshold in terms of quantiles
np.percentile(train_losses, [0.85, 90, 91, 95, 97, 98, 99])
threshold = np.round(np.percentile(train_losses, 90), 4) # set threshold as the 90-percentile of the training loss


# train_score is training loss
train_scores = pd.DataFrame(index=avg_df2.index)
loss_column = 'huber_loss' if config['loss_fn'] == huber_loss else 'mae_loss'
train_scores[loss_column] = train_losses
train_scores['Anomaly'] =  [0 if l < threshold else 1 for l in train_losses]
train_scores.to_csv(os.path.join(csv_files, f'bilstm_train_exp{exp}.csv'))

# detect anomaly on the training scores, breaching threshold means anomaly
threshold_ = np.array([threshold]*len(train_scores))
plot_anomaly_threshold(train_scores, plot_name=f'bilstm_thresh_exp{exp}', thresh=threshold_)

fig,ax = plt.subplots(1,1)
ax.plot(train_scores.loc[:,"huber_loss"], label="anomaly_score")
train_scores["threshold"] = threshold_
ax.plot(train_scores.loc[:,"threshold"], label="threshold (90%)")
idx = train_scores["Anomaly"] == 1
ax.plot(train_scores.loc[idx,"huber_loss"], marker="o", ms=5, fillstyle="none", linestyle="none", color="red", label="detected")
ax.legend()
fig.savefig(save_plots_dir+"BiLSTM")

bilstm_predicted_anomalies(train_scores, size=(10, 2.5), plot_name=f'bilstm_val_pred_anom_exp{exp}')


# reconstruction of the 4 bearings
preds_array = scaler.inverse_transform(np.array(train_preds))
plot_reconstruction(avg_df2, preds_array, plot_name=f'bilstm_rec_exp{exp}')

del train_preds, train_losses, train_scores
gc.collect()


# ====================== Prediction on unseen data (avg_df3) ======================= #

bilstm_model.load_state_dict(best_params['model'], strict=False)
test_preds, test_losses = predict(bilstm_model, test_loader, config['loss_fn'])

sns.set(rc={'figure.figsize': (4, 3)})
sns.set_style('whitegrid')
sns.displot(avg_df3.values, bins=30, kde=True)
plt.grid(color ='k', ls= '--', lw=0.5)
plt.xlabel('Avg signal')
plt.title('Dataset distribution avg_df3')
plt.show()

np.quantile(avg_df3.values, [0.8, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99])

plt.figsize = (5,5)
sns.set_style('whitegrid', {"grid.color": 'k', 'grid.linestyle': '--'})
sns.displot(test_losses, bins=30, kde=True)
plt.title('Test loss of avg_df3')
plt.xlabel('Loss')
plt.grid(lw=0.5, c='k')
plt.show()

np.percentile(test_losses, [90, 95, 97, 98, 99]) # selection of test threshold is very vague
test_thresh = np.round(np.percentile(test_losses, 99), 4)
test_thresh
threshold_ = np.array([test_thresh]*len(test_losses))

test_scores = pd.DataFrame(index=avg_df3.index)
test_scores[loss_column] = test_losses
test_scores['Anomaly'] =  [0 if l < test_thresh else 1 for l in test_losses]
test_scores.to_csv(os.path.join(csv_files, f'bilstm_test_exp{exp}.csv'))

plot_anomaly_threshold(test_scores, thresh=threshold_, plot_name=f'test_thres_bilstm_exp_{exp}')
# Loss scores with anomalies
bilstm_predicted_anomalies(test_scores, size=(10, 2.5), plot_name=f'test_bilstm_anom_exp{exp}')

# reconstruction of test dataset
test_preds_array = scaler.inverse_transform(np.array(test_preds)) # Revert scaling for reconstuction
plot_reconstruction(avg_df3, test_preds_array, f'bilstm_test_preds_exp{exp}')

del threshold_, test_preds, test_losses, test_scores, test_preds_array
gc.collect()










