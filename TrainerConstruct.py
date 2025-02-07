import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Loss import get_loss


class Trainer:
    def __init__(self, trainloader, testloader, Encoder, Decoder, latent_dim, device = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder.to(self.device)
        self.decoder = Decoder.to(self.device)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3)
        self.dataloader = trainloader
        self.testdataloader = testloader
        self.latent_dim = latent_dim
    

    def train(self, num_epochs = 50, factor = 100):
        self.encoder.train()
        self.decoder.train()

        for epoch in range(num_epochs):
            total_loss = 0
            test_loss = 0
            for batch_idx, x in enumerate(self.dataloader):
                x = x.permute(0, 3, 1, 2)
                x = x.to(self.device)
                mean, logvar, z = self.encoder(x)
                x_recontructed = self.decoder(z)

                loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size = x.shape[0])
                loss = loss_fn(x, x_recontructed)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)

            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                for x in self.testdataloader:
                    x = x.permute(0, 3, 1, 2)
                    x = x.to(self.device)
                    mean, logvar, z = self.encoder(x)
                    x_reconstructed = self.decoder(z)

                    loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                    loss = loss_fn(x, x_reconstructed)
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(self.testdataloader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")


        print("Training complete!")

