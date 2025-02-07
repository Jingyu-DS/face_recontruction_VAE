import os
import imageio
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoading import DataLoad
from EncoderArch import Encoder
from DecoderArch import Decoder
from Loss import get_loss
from TrainerConstruct import Trainer


DATASET_PATH = "lfw-deepfunneled/"
ATTRIBUTES_PATH = "lfw_attributes.txt"
dx = 70
dy = 70
dimx = 45
dimy = 45
LATENT_SPACE_SIZE = 100
batch_size = 64


dataload = DataLoad(DATASET_PATH, ATTRIBUTES_PATH)
all_photos, all_attrs = dataload.fetch_dataset(dx, dy, dimx, dimy)
all_photos = np.array(all_photos / 255, dtype='float32')
X_train, X_val = train_test_split(all_photos, test_size=0.2, random_state=365)
train_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=X_val, batch_size=batch_size, shuffle=False)

encoder = Encoder(LATENT_SPACE_SIZE)
decoder = Decoder(LATENT_SPACE_SIZE)
trainer = Trainer(train_loader, test_loader, encoder, decoder, LATENT_SPACE_SIZE)

if __name__ == "__main__":
    trainer.train()
    torch.save(encoder.state_dict(), "vae_encoder.pth")
    torch.save(decoder.state_dict(), "vae_decoder.pth")

