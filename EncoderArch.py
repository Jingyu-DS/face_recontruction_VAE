import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.flatten_dim = 32 * 3 * 3
        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
    
    def sample_latent_features(self, distribution):
        distribution_mean, distribution_logvar = distribution
        batch_size = distribution_logvar.shape[0]
        random = torch.randn(batch_size, distribution_logvar.shape[1], device=distribution_logvar.device)
        return distribution_mean + torch.exp(0.5 * distribution_logvar) * random


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        z = self.sample_latent_features((mean, logvar))  

        return mean, logvar, z
    


"""
Example Testing:

encoder = Encoder(latent_dim=128)
sample_input = torch.randn(1, 3, 45, 45)

mean, logvar, latent_encoding = encoder(sample_input)
print("Mean Shape:", mean.shape)
print("Log Variance Shape:", logvar.shape)
print("Latent Encoding Shape:", latent_encoding.shape)
"""