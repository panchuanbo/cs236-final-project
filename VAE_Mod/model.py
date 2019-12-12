import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.l1 = nn.Conv1d(input_dim, 512, kernel_size=1)
        self.l2 = nn.Conv1d(512, 256, kernel_size=1)
        self.l3 = nn.Conv1d(256, 128, kernel_size=1)

        self.mu = nn.Conv1d(128, latent_dim, kernel_size=1)
        self.var = nn.Conv1d(128, latent_dim, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        mean = self.mu(x)
        log_var = self.var(x)

        return mean, log_var

class CDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, n_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_classes = n_classes

        self.l1 = nn.Conv1d(latent_dim+n_classes, 512, kernel_size=1)
        self.l2 = nn.Conv1d(512, 256, kernel_size=1)
        self.l3 = nn.Conv1d(256, 128, kernel_size=1)

        self.out = nn.Conv1d(128, output_dim, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        out = F.sigmoid(self.out(x))

        return out

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, n_classes):
        super().__init__()

        self.encoder = CEncoder(input_dim, latent_dim, n_classes)
        self.decoder = CDecoder(latent_dim, output_dim, n_classes)

    def forward(self, x, y):
        z_mu, z_var = self.encoder(x)

        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(z_mu)

        z = torch.cat((z, y), axis=1)

        out = self.decoder(z)

        return out, z_mu, z_var

    def convert(self, x, src, dst, device):
        with torch.no_grad():
            x = torch.tensor(x).to(torch.float)
            n_samples, input_dim, depth = x.shape
            src = (torch.zeros(depth) if src == 0 else torch.ones(depth)).to(torch.int64)
            dst = (torch.zeros(depth) if dst == 0 else torch.ones(depth)).to(torch.int64)

            src = F.one_hot(src, num_classes=2).to(torch.float).T
            dst = F.one_hot(dst, num_classes=2).to(torch.float).T

            (src, dst) = (src.reshape((1, 2, depth)), dst.reshape((1, 2, depth)))
            src = torch.cat([src]*n_samples)
            dst = torch.cat([dst]*n_samples)

            z_mu, z_var = self.encoder(x.to(device))

            std = torch.exp(z_var / 2)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(z_mu)
            z = torch.cat((z, dst.to(device)), axis=1)

            out = self.decoder(z)

            return out, z_mu, z_var
