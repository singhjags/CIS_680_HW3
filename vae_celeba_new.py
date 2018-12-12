#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:13:46 2018

@author: jagtarsingh
"""


# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils

from random import randint
from matplotlib import pyplot as plt

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
    def __init__(self, image_channels=3, h_dim=512, z_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
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
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=1),
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





image_channels = fixed_x.size(1)
print(image_channels)





model = VAE(image_channels=image_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002) 

def loss_fn(recon_x, x, mu, log_sig):
    recon_loss = F.binary_cross_entropy(recon_x, x, size_average=False)
    # recon_loss = F.mse_loss(recon_x, x, size_average=False)

    
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
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
        loss_all.append(loss.data[0]/bs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        
        if itera%1000 == 0:
            n = min(images.size(0), 16)
            comparison = torch.cat([images.cuda()[:n],
                                          recon_images[:n]])
            save_image(comparison.data.cpu(),
                         './reconstructed/reconstruction_' + str(epoch) + '.png', nrow=n)
        print("Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                epochs, loss.data[0]/bs, recon_loss.data[0]/bs, kl_div_loss.data[0]/bs))


plt.plot(loss_all)
plt.imshow()
plt.savefig('loss_plot.png')
torch.save(model.state_dict(), 'vae.torch')


# def compare(x):
#     recon_x, _, _ = model(x)
#     return torch.cat([x, recon_x])


# def compare2(x):
#     recon_x, _, _ = model(x)
#     return recon_x


# # In[186]:


# # sample = torch.randn(bs, 1024)
# # compare_x = vae.decoder(sample)

# # fixed_x, _ = next(iter(dataloader))
# # fixed_x = fixed_x[:8]
# import pdb
# real_img = dataset[randint(1, 150)][0].unsqueeze(0)

# compare_x = compare2(real_img)
# for i in range(31):
    
#     fixed_x = dataset[randint(1, 150)][0].unsqueeze(0)
# #     pdb.set_trace()
#     real_img = torch.cat([real_img, fixed_x])

#     new_compare_x = compare2(fixed_x)
    
#     compare_x = torch.cat([compare_x, new_compare_x])
    
    
    

# save_image(compare_x.data.cpu(), 'fake_image.png')
# save_image(real_img.data.cpu(), 'real_image.png')
# print("Generated Images ")
# display(Image('fake_image.png', width=700, unconfined=True))
# print("Real Images ")
# display(Image('real_image.png', width=700, unconfined=True))


# # In[187]:


# fixed_x = dataset[randint(2, 180)][0].unsqueeze(0)
# compare_x = compare(fixed_x)

# save_image(compare_x.data.cpu(), 'sample_image.png')
# display(Image('sample_image.png', width=700, unconfined=True))

