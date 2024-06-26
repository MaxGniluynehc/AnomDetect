import os
import torch as tc
from data_prep.data_prep import split_train_test, NASABearingDataSet, avg_df2
from torch.utils.data.dataloader import DataLoader
from models.LSTM_VAE.LSTM_VAE_tc import LSTM_VAE

# get from https://github.com/TimyadNyda/Variational-Lstm-Autoencoder
saved_model_dir = "../Anomaly_Detection_Methods/saved_models/"



# ======================== Train ========================= #
def train_one_epoch(model:LSTM_VAE, optim, dataloader):
    avg_loss = 0
    for i, batch in enumerate(dataloader):
        batch = batch.permute((1,0,2))
        # model.zero_grad()
        optim.zero_grad()

        if i == 0:
            x_mus, x_lsgs, x, (hn_enc, cn_enc), z_mus, z_lsgs, z, (hn_dec, cn_dec), = (
                model.forward(batch, hidden_enc=None, hidden_dec=None,
                              train=True, pass_hidden=True, corrupt_data=True, return_state=True))
        else:
            x_mus, x_lsgs, x, (hn_enc, cn_enc), z_mus, z_lsgs, z, (hn_dec, cn_dec), = (
                model.forward(batch, hidden_enc=(hn_enc.detach(), cn_enc.detach()), hidden_dec=(hn_dec.detach(), cn_dec.detach()),
                              train=True, pass_hidden=True, corrupt_data=True, return_state=True))

        loss_i = model.loss(batch, x_mus, x_lsgs, z_mus, z_lsgs, hidden_enc=(hn_enc.detach(), cn_enc.detach()), train=True, progress_based=True)
        loss_i.backward(retain_graph=False)
        optim.step()

        avg_loss += loss_i.item()
    return avg_loss/(i+1)


def train(Nepoch, model:LSTM_VAE, optimizer, lr, traindataloader, evaldataloader):
    optim = optimizer(model.parameters(), lr=lr, maximize=True)
    # model.initialize_linears()
    train_loss_list = []
    eval_loss_list = []
    for epoch in range(Nepoch):

        model.train(True)
        train_loss_at_epoch = train_one_epoch(model, optim, traindataloader)

        model.eval()
        with tc.no_grad():
            eval_loss = 0
            for i, batch in enumerate(evaldataloader):
                batch = batch.permute((1,0,2))
                if i == 0:
                    x_mus, x_lsgs, x, (hn_enc, cn_enc), z_mus, z_lsgs, z, (hn_dec, cn_dec), = (
                        model.forward(batch, hidden_enc=None, hidden_dec=None,
                                      train=False, pass_hidden=True, corrupt_data=True, return_state=True))
                else:
                    x_mus, x_lsgs, x, (hn_enc, cn_enc), z_mus, z_lsgs, z, (hn_dec, cn_dec), = (
                        model.forward(batch, hidden_enc=(hn_enc.detach(), cn_enc.detach()), hidden_dec=(hn_dec.detach(), cn_dec.detach()),
                                      train=False, pass_hidden=True, corrupt_data=True, return_state=True))
                eval_loss_i = model.loss(batch, x_mus, x_lsgs, z_mus, z_lsgs, hidden_enc=(hn_enc.detach(), cn_enc.detach()), train=False)
                eval_loss += eval_loss_i.item()
            eval_loss_at_epoch = eval_loss/(i+1)

        print('Epoch {}/{}: LOSS train {:.4f} valid {:.4f}'.format(epoch, Nepoch, train_loss_at_epoch, eval_loss_at_epoch))
        train_loss_list.append(train_loss_at_epoch)
        eval_loss_list.append(eval_loss_at_epoch)
    return train_loss_list, eval_loss_list, (hn_enc, cn_enc), (hn_dec, cn_dec)


batch_size = 10
avg_df2_train, avg_df2_test = split_train_test(avg_df2, test=0.3)
ds_train, ds_test = NASABearingDataSet(avg_df2_train, 30), NASABearingDataSet(avg_df2_test, 30)
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

self = LSTM_VAE(2, avg_df2_train.shape[-1],3,3,0.5)
self.initialize_linears()

self.load_model(saved_model_dir+"LSTM")

# data = next(iter(dl_train)).permute((1,0,2))
# z_mus, z_lsgs, z, (hn_enc, cn_enc) = self.encode(data, return_hidden=True)
# x_mus, x_lsgs, x, (hn_dec, cn_dec) = self.decode(z, return_hidden=True)
# x_mus, x_lsgs, x, (hn_enc, cn_enc), z_mus, z_lsgs, z, (hn_dec, cn_dec) = (
#     self.forward(data, train=True, pass_hidden=True, corrupt_data=True, return_state=True))
# anoms = self.anomaly_score(data)
# p_theta = MultivariateNormal(x_mus, tc.matrix_exp(x_lsgs))
# q_phi = MultivariateNormal(z_mus, tc.matrix_exp(z_lsgs))
# self.loss(data, x_mus, x_lsgs, z_mus, z_lsgs).backward(retain_graph=False)
# optim = tc.optim.Adam(self.parameters(), lr=1e-3, maximize=True)

train_loss_list, eval_loss_list, hidden_enc, hidden_dec = train(100, self, tc.optim.Adam, 1*1e-3, dl_train, dl_test)
self.save_model("LSTM_VAE_trained")




# ====================== Model Evaluation ====================== #
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use("TkAgg")

df2_test = tc.tensor(avg_df2.values, dtype=tc.float32).unsqueeze(1)
anomaly_scores = self.anomaly_score(df2_test, train=False, pass_hidden=True).view(-1)
anom_scores = anomaly_scores.detach().numpy()
threshold = self.threshold(state_based=True, data=df2_test, c=0)


save_plots_dir = "../Anomaly_Detection_Methods/plots/anomaly_scores/"
os.makedirs(save_plots_dir, exist_ok=True)


df_toplot = pd.DataFrame({"anom_score": anom_scores,
                          "threshold": threshold},
                         index=avg_df2.index)

fig,ax = plt.subplots(1,1)
ax.plot(df_toplot.loc[:,"anom_score"], label="anomaly_score")
ax.plot(df_toplot.loc[:,"threshold"], label="state_based threshold")
idx = df_toplot.loc[:, "anom_score"] > df_toplot.loc[:, "threshold"]
ax.plot(df_toplot.loc[idx, "anom_score"], linestyle="none", marker="o", fillstyle="none", ms=5, label="detected", color="red")
ax.legend()
fig.savefig(save_plots_dir + "LSTM_VAE")



