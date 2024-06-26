import os
import torch as tc
from torch.nn import LSTM, Linear, Dropout
from torch.distributions import Normal, MultivariateNormal
from torch.distributions.kl import kl_divergence
from sklearn.svm import SVR
# get from https://github.com/TimyadNyda/Variational-Lstm-Autoencoder


class LSTM_VAE(tc.nn.Module):
    def __init__(self, state_dim, data_dim, hidden_dim, num_layers, kld_coef, dropout=0.5):
        super().__init__()

        self.state_dim = state_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.kld_coef = kld_coef
        self.num_layers = num_layers
        self.drop = Dropout(dropout)

        # each batch of data x: (seq_len, batch_size, data_dim)
        # each corresponding state z: (seq_len, batch_size, state_dim)

        self.enc_lstm = LSTM(input_size=data_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)
        self.enc_ff_mu = Linear(self.hidden_dim, self.state_dim)
        self.enc_ff_cov = Linear(self.hidden_dim, self.state_dim * self.state_dim, bias=False)

        self.dec_lstm = LSTM(input_size=state_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)
        self.dec_ff_mu = Linear(self.hidden_dim, self.data_dim)
        self.dec_ff_cov = Linear(self.hidden_dim, self.data_dim * self.data_dim, bias=False)

        # global batch_size

    def initialize_lstm(self, batch_size, init_hidden=True):
        if init_hidden:
            h0, c0 = tc.zeros([self.num_layers, batch_size, self.hidden_dim]), tc.zeros(
                [self.num_layers, batch_size, self.hidden_dim])
            return (h0, c0)
        else:
            return None

    def initialize_linears(self, name = None):
        if name is not None:
            getattr(self, name).weight.data.uniform_(-0.1, 0.1)
        else:
            for name in ["enc_ff_mu", "enc_ff_cov", "dec_ff_mu", "dec_ff_cov"]:
                getattr(self, name).weight.data.uniform_(-0.1, 0.1)
                # if "cov" in name: # weight of cov layers should be symmetric so that cov is symmetric
                #     getattr(self, name).weight = getattr(self, name).weight.triu() + getattr(self, name).weight.triu(1).transpose(-1, -2)

    def encode(self, data, hidden=None, train=False, return_hidden=False, corrupt_data=False, corruption_sigma=0.1):
        if len(data.shape) < 3 and len(data.shape) == 2:
            data = data.unsqueeze(1)
        else:
            ValueError("Data is in wrong shape, should be either 2d [seq_len, data_dim] or 3d [seq_len, batch_size, data_dim].")
        if corrupt_data:
            data = data.add(tc.normal(tc.zeros(data.size()), tc.ones(data.size())*corruption_sigma))
        batch_size = data.size(1)

        if any([hidden is None]):
            hidden = self.initialize_lstm(batch_size=batch_size)

        output, (hn, cn) = self.enc_lstm(data, hidden)
        if train:
            output = self.drop(output.view(-1, self.hidden_dim)).view(-1, batch_size, self.hidden_dim)
        z_mus = self.enc_ff_mu(output) # z_mus: [seq_len, batch_size, state_dim]
        z_lsgs= self.enc_ff_cov(output).view(-1, batch_size, self.state_dim, self.state_dim) # z_lsgs: [seq_len, batch_size, state_dim, state_dim]
        if train:
            z_mus = self.drop(z_mus.view(-1, self.state_dim)).view(-1, batch_size, self.state_dim)
            z_lsgs = self.drop(z_lsgs.view(-1, self.state_dim * self.state_dim)).view(-1, batch_size, self.state_dim, self.state_dim)
        z_lsgs = z_lsgs.add(tc.transpose(z_lsgs, 2, 3)) / 2 # convert z_lsgs as symmetric matrix
        q_phi = MultivariateNormal(z_mus, tc.matrix_exp(z_lsgs))
        z = q_phi.rsample(tc.ones(1).size()).view(z_mus.size()) # z: [seq_len, batch_size, state_dim]

        if return_hidden:
            hidden_enc = (hn, cn)
            return z_mus, z_lsgs, z, hidden_enc
        else:
            return z_mus, z_lsgs, z

    def decode(self, state, hidden=None, train=False, return_hidden=False):
        if len(state.shape) < 3 and len(state.shape) == 2:
            state = state.unsqueeze(1)
        else:
            ValueError("State is in wrong shape, should be either 2d [seq_len, state_dim] or 3d [seq_len, batch_size, state_dim].")
        batch_size = state.size(1)

        if any([hidden is None]):
            hidden = self.initialize_lstm(batch_size=batch_size)

        output, (hn, cn) = self.dec_lstm(state, hidden)
        if train:
            output = self.drop(output.view(-1, self.hidden_dim)).view(-1, batch_size, self.hidden_dim)
        x_mus = self.dec_ff_mu(output) # x_mus: [seq_len, batch_size, data_dim]
        x_lsgs= self.dec_ff_cov(output).view(-1, batch_size, self.data_dim, self.data_dim)  # x_lsgs: [seq_len, batch_size, data_dim, data_dim]

        if train:
            x_mus = self.drop(x_mus.view(-1, self.data_dim)).view(-1, batch_size, self.data_dim)
            x_lsgs = self.drop(x_lsgs.view(-1, self.data_dim*self.data_dim)).view(-1, batch_size, self.data_dim, self.data_dim)
        x_lsgs = x_lsgs.add(tc.transpose(x_lsgs, 2, 3)) / 2  # convert x_lsgs as symmetric matrix
        p_theta = MultivariateNormal(x_mus, tc.matrix_exp(x_lsgs))
        x = p_theta.rsample(tc.ones(1).size()).view(x_mus.size()) # x: [seq_len, batch_size, data_dim]

        if return_hidden:
            hidden_dec = (hn, cn)
            return x_mus, x_lsgs, x, hidden_dec
        else:
            return x_mus, x_lsgs, x

    def forward(self, data, hidden_enc=None, hidden_dec=None, train=False, pass_hidden=False, corrupt_data=False, return_state=False):
        if pass_hidden:
            z_mus, z_lsgs, z, hidden_enc = self.encode(data, hidden_enc, train, return_hidden=True, corrupt_data=corrupt_data)

            x_mus, x_lsgs, x, hidden_dec = self.decode(z, hidden_enc, train, return_hidden=True)

        else:
            z_mus, z_lsgs, z, hidden_enc = self.encode(data, hidden_enc, train, return_hidden=True, corrupt_data=corrupt_data)
            x_mus, x_lsgs, x, hidden_dec = self.decode(z, hidden_dec, train, return_hidden=True)

        if return_state:
            return x_mus, x_lsgs, x, hidden_enc, z_mus, z_lsgs, z, hidden_dec
        else:
            return x_mus, x_lsgs, x, hidden_enc

    def anomaly_score(self, data, hidden_enc=None, hidden_dec=None, train=False, pass_hidden=False):
        x_mus, x_lsgs, x, _ = self.forward(data, hidden_enc, hidden_dec, train, pass_hidden)
        p_theta = MultivariateNormal(x_mus, tc.matrix_exp(x_lsgs))
        return - p_theta.log_prob(data)

    def threshold(self, state_based = True, data=None, anom_scores=None, c=0.01):
        if state_based:
            x_mus, x_lsgs, x, hidden_enc, z_mus, z_lsgs, z, hidden_dec = self.forward(data, train=False, return_state=True, pass_hidden=True)
            state = z.squeeze(1).detach().numpy()
            svr = SVR()
            if anom_scores is None:
                anom_scores = self.anomaly_score(data, train=False, pass_hidden=True).detach().numpy()
            svr.fit(state, anom_scores)
            threshold = svr.predict(state)
            threshold += c
            return threshold
        else:
            return c

    def state_prior(self, data, hidden_enc=None, train=False, progress_based=False):
        # assert all([state is None, data is None]), ValueError("State and data cannot be None at the same time!")
        z_mus, z_lsgs, state = self.encode(data, hidden_enc, train)
        if progress_based:
            return MultivariateNormal(z_mus, tc.matrix_exp(z_lsgs))
        else:
            I = tc.eye(self.state_dim).view(1,1,self.state_dim,self.state_dim).repeat(z_lsgs.size(0), z_lsgs.size(1), 1,1)
            return MultivariateNormal(tc.zeros_like(z_mus), I)

    def loss(self, data, x_mus, x_lsgs, z_mus, z_lsgs, hidden_enc=None, train=False, progress_based=False):

        prior = self.state_prior(data, hidden_enc, train, progress_based)
        q_phi = MultivariateNormal(z_mus, tc.matrix_exp(z_lsgs))
        kl_d = kl_divergence(q_phi, prior)  # kl_d: [seq_len, batch_size]

        p_theta = MultivariateNormal(x_mus, tc.matrix_exp(x_lsgs))
        log_p_theta = p_theta.log_prob(data) # log_p_theta: [seq_len, batch_size]

        return log_p_theta.add(-kl_d).mean(-1).sum() # (-kl_d + log_p_theta).mean(-1).sum() # average over batch, sum over time

    def save_model(self, name, saved_model_dir=None):
        saved_model_dir = "../Anomaly_Detection_Methods/saved_models/" if saved_model_dir is None else saved_model_dir
        os.makedirs(saved_model_dir, exist_ok=True)
        tc.save(self.state_dict(), saved_model_dir+name)

    def load_model(self, state_dict_path, saved_model_dir=None):
        # saved_model_dir = "../Anomaly_Detection_Methods/saved_models/" if saved_model_dir is None else saved_model_dir
        self.load_state_dict(tc.load(state_dict_path))


