import torch
import torch.nn as nn  ##Implements deep learning models.
import torch.optim as optim #: For optimization and loss functions.
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset  #Handles dataset batching and loading.
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score  #Computes AUC, precision, and recall.
from sklearn.preprocessing import StandardScaler
import scapy.all as scapy  #Extracts network traffic features from PCAP files.
from scipy.io import arff

class Encoder(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(32 * (input_dim // 4), latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 32 * (input_dim // 4)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 32, self.input_dim // 4)
        return self.decoder(x)


# Discriminator for Adversarial Training
class Discriminator(nn.Module):
    def __init__(self, latent_dim=16):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


# Deep SVDD Model
class DeepSVDD(nn.Module):
    def __init__(self, latent_dim=16):
        super(DeepSVDD, self).__init__()
        self.center = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
        self.radius = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, z):
        # Calculate the distance from the center
        return torch.norm(z - self.center, dim=1)
class AAELoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, adversarial_weight=0.1):
        super(AAELoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_weight = reconstruction_weight
        self.adversarial_weight = adversarial_weight

    def forward(self, real_data, reconstructed_data, disc_real, disc_fake):
        rec_loss = self.reconstruction_loss(reconstructed_data, real_data)
        adv_loss_real = self.adversarial_loss(disc_real, torch.ones_like(disc_real))
        adv_loss_fake = self.adversarial_loss(disc_fake, torch.zeros_like(disc_fake))

        total_loss = (self.reconstruction_weight * rec_loss +
                      self.adversarial_weight * (adv_loss_real + adv_loss_fake))

        return total_loss, rec_loss, adv_loss_real + adv_loss_fake
class DSVDDLoss(nn.Module):
    def __init__(self, center, nu=0.1, dist_weight=0.5):
        super(DSVDDLoss, self).__init__()
        self.center = center
        self.nu = nu  # Regularization parameter
        self.dist_weight = dist_weight  # Weight for distribution constraint

    def forward(self, z, aae_z):
        # DSVDD loss: distance from center with regularization
        distance = torch.norm(z - self.center, dim=1)
        svdd_loss = torch.mean(distance) - self.nu * torch.norm(self.center)

        # Distribution constraint: KL divergence between AAE and DSVDD distributions
        # Simplified as MSE between normalized latent vectors
        z_norm = F.normalize(z, p=2, dim=1)
        aae_z_norm = F.normalize(aae_z, p=2, dim=1)
        dist_loss = F.mse_loss(z_norm, aae_z_norm)

        # Combined loss
        total_loss = svdd_loss + self.dist_weight * dist_loss

        return total_loss, svdd_loss, dist_loss


# Normalization function to ensure consistency between AAE and DSVDD outputs
def normalize_latent(z, eps=1e-8):
    """
    Normalize latent vectors to unit norm
    """
    z_norm = z / (torch.norm(z, dim=1, keepdim=True) + eps)
    return z_norm


# Initialize DSVDD center using AAE encoder outputs
def init_center(encoder, data_loader, device):
    """
    Initialize the center of DSVDD as the mean of encoded representations
    """
    encodings = []
    encoder.eval()
    with torch.no_grad():
        for data in data_loader:
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
            data = data.to(device)
            encoded = encoder(data)
            encodings.append(encoded)

    encodings = torch.cat(encodings, dim=0)
    center = torch.mean(encodings, dim=0)
    return center


