#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:13:46 2018

@author: jagtarsingh
"""







import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
import matplotlib 
from random import randint
from matplotlib import pyplot as plt

plt.switch_backend('agg')
from IPython.display import Image
from IPython.core.display import Image, display
import pdb




# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




bs = 128



# Load Data
dataset = datasets.ImageFolder(root='/home/jagtar_singh_upenn/CIS_680_HW3/celeba', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(), 
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
print(len(dataset.imgs), len(dataloader))





# Fixed input for debugging
fixed_x, _ = next(iter(dataloader))
save_image(fixed_x, 'real_image.png')

Image('real_image.png')




class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)





class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)





class VAE(nn.Module):
    def __init__(self, channels=3, h_dim=512, z_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
#             nn.Conv2d(512, 1024, kernel_size=3, stride=2),
#             nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=6, stride=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        h = self.encoder(x)
        mu, log_sig = self.fc1(h), self.fc2(h)
        std = log_sig.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        z = self.fc3(z)
        z = self.decoder(z)
        return z, mu, log_sig
    
    def decode_random(self):
        z = torch.randn([32,64]).cuda()
        z = self.fc3(z)
        z = self.decoder(z)
        return z
        





channels = fixed_x.size(1)
print(channels)





model = VAE(channels=channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002) 

def loss_fn(recon_x, x, mu, log_sig):
    recon_loss = F.binary_cross_entropy(recon_x, x, size_average=False)
    # recon_loss = F.mse_loss(recon_x, x, size_average=False)
    KL_loss = -0.5 * torch.mean(1 + log_sig - mu.pow(2) - log_sig.exp())

    return recon_loss + KL_loss, recon_loss, KL_loss

epochs = 20


itera = 0
loss_all = []
for epoch in range(epochs):
    for idx, (images, _) in enumerate(dataloader):
        itera+=1
        recon_images, mu, log_sig = model(images.cuda())
        loss, recon_loss, kl_div_loss = loss_fn(recon_images, images.cuda(), mu, log_sig)
        batch_s = images.size(0)
        loss_all.append(loss.data[0]/batch_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        
        if itera%1000 == 0:
            n = min(images.size(0), 16)
            fixed_out,_,_ = model(fixed_x.cuda())
            fixed_x_new = fixed_x.cuda()
            comp = torch.cat([fixed_x_new[:n],fixed_out[:n]])
            comparison = torch.cat([images.cuda()[:n],
                                          recon_images[:n]])
#            save_image(comparison.data.cpu(),
#                         './reconstructed_celeba/reconstruction_' + str(epoch) + '.png', nrow=n)
            save_image(comp.data.cpu(),
                         './reconstructed_celeba/reconstruction_' + str(epoch) + '.png', nrow=n)
            random_out_img = model.decode_random()
            
            save_image(random_out_img.data.cpu(),
                         './reconstructed_celeba_rand/random_' + str(epoch) + '.png', nrow=n)
        print("Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                epochs, loss.data[0]/bs, recon_loss.data[0]/bs, kl_div_loss.data[0]/bs))


plt.plot(loss_all)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Vs Iteration for CELEBA dataset')
plt.savefig('celeba_loss.png')
torch.save(model.state_dict(), 'vae.torch')


