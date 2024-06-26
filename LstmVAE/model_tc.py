import torch as tc
from torch.nn import LSTM, Linear, KLDivLoss
import numpy as np

class LSTM_VAE(tc.nn.Module):
    def __init__(self, state_dim, data_dim, T, hidden_dim, kld_coef, **kwargs):
        super().__init__()

        self.state_dim = state_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.kld_coef = kld_coef
        self.T = T

        # each batch of data x: (N, T, data_dim)
        # each corresponding state z: (N, T, state_dim)

        self.enc_lstm = LSTM(input_size=data_dim, hidden_size=self.hidden_dim, num_layers=self.T, batch_first=True)
        self.enc_ff_mu = Linear(self.hidden_dim, self.state_dim)
        self.enc_ff_cov = Linear(self.hidden_dim, self.T)

        self.dec_lstm = LSTM(input_size=state_dim, hidden_size=self.hidden_dim, num_layers=self.T, batch_first=True)
        self.dec_ff_mu = Linear(self.hidden_dim, self.data_dim)
        self.dec_ff_cov = Linear(self.hidden_dim, self.data_dim)

    def initialize_lstm(self, **kwargs):
        h0, c0 = tc.zeros([self.T, kwargs["batch_size"], self.hidden_dim]), tc.zeros([self.T, kwargs["batch_size"], self.hidden_dim])
        return h0, c0

    def encode(self, data, h0 = None, c0=None, **kwargs):
        if any((h0, c0) is None):
            h0,c0 = self.initialize_lstm(**kwargs)
        output, (hn, cn) = self.enc_lstm(data, (h0, c0))
        z_mus = self.enc_ff_mu(output)
        z_logsigmas = self.enc_ff_cov()

        return output

    def decode(self, state):
        pass

    def forward(self, data):
        pass







#==========================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from timeit import default_timer as timer

class LSTMVarAutoencoder(nn.Module):
    def __init__(self, intermediate_dim, z_dim, n_dim, kulback_coef=0.1):
        super(LSTMVarAutoencoder, self).__init__()
        self.z_dim = z_dim
        self.n_dim = n_dim
        self.intermediate_dim = intermediate_dim
        self.kulback_coef = kulback_coef

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(n_dim, intermediate_dim, batch_first=True)
        self.z_mean_layer = nn.Linear(intermediate_dim, z_dim)
        self.z_log_sigma_layer = nn.Linear(intermediate_dim, z_dim)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(z_dim, intermediate_dim, batch_first=True)
        self.output_layer = nn.Linear(intermediate_dim, n_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Encoder
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n[-1]  # Take the last hidden state
        z_mean = self.z_mean_layer(h_n)
        z_log_sigma = self.z_log_sigma_layer(h_n)
        z = self.gaussian_sampling(z_mean, z_log_sigma)

        # Repeat z for each time step
        repeated_z = z.unsqueeze(1).repeat(1, seq_len, 1)

        # Decoder
        decoded_seq, _ = self.decoder_lstm(repeated_z)
        x_reconstr_mean = self.output_layer(decoded_seq)

        return x_reconstr_mean, z_mean, z_log_sigma

    def gaussian_sampling(self, mean, log_sigma):
        eps = torch.randn_like(log_sigma)
        return mean + torch.exp(0.5 * log_sigma) * eps

    def loss_function(self, x, recon_x, z_mean, z_log_sigma):
        reconstr_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + z_log_sigma - z_mean.pow(2) - z_log_sigma.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return reconstr_loss + self.kulback_coef * kl_loss

def fit(model, X, learning_rate=0.001, batch_size=100, num_epochs=200, verbose=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nTraining...\n")
    start = timer()

    for epoch in range(num_epochs):
        train_error = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x_batch = batch[0]
            recon_x, z_mean, z_log_sigma = model(x_batch)
            loss = model.loss_function(x_batch, recon_x, z_mean, z_log_sigma)
            loss.backward()
            optimizer.step()
            train_error += loss.item()

        mean_loss = train_error / len(dataloader)
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch + 1} Loss {mean_loss:.5f}")

    end = timer()
    print("\nTraining time {:0.2f} minutes".format((end - start) / 60))

def reconstruct(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        recon_X, _, _ = model(X_tensor)
    return recon_X.numpy()

def reduce(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        _, z_mean, _ = model(X_tensor)
    return z_mean.numpy()

# Example usage:
# model = LSTMVarAutoencoder(intermediate_dim=128, z_dim=64, n_dim=1)
# fit(model, X_train)
# reconstructed_X = reconstruct(model, X_test)
# reduced_X = reduce(model, X_test)

