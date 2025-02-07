import torch 
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.deconv1 = nn.ConvTranspose2d(in_channels = latent_dim, out_channels = 64, kernel_size = 3)
        self.upsample1 = nn.Upsample(scale_factor = 2, mode = "nearest")

        self.deconv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size=3)
        self.upsample2 = nn.Upsample(scale_factor = 2, mode='nearest')

        self.deconv3 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size=5)
        self.upsample3 = nn.Upsample(scale_factor = 2, mode='nearest')

        self.deconv4 = nn.ConvTranspose2d(in_channels = 16, out_channels = 3, kernel_size=6)
    
    
    def forward(self, z):
        x = z.view(-1, self.latent_dim, 1, 1)
        x = F.relu(self.deconv1(x))
        x = self.upsample1(x)

        x = F.relu(self.deconv2(x))
        x = self.upsample2(x)

        x = F.relu(self.deconv3(x))
        x = self.upsample3(x)

        x = F.relu(self.deconv4(x))  

        return x 

"""
Testing Decoder:

decoder = Decoder(latent_dim=LATENT_SPACE_SIZE)
latent_vector = torch.randn(1, LATENT_SPACE_SIZE)  
output_image = decoder(latent_vector)
print(f"Decoder output shape: {output_image.shape}")

"""