
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
# from torchsummary import summary

# from pushover import notify
# from utils import makegif
from random import randint

from IPython.display import Image
from IPython.core.display import Image, display
import pdb
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


bs = 128


# In[4]:


# Load Data
dataset = datasets.ImageFolder(root='/home/jagtar_singh_upenn/CIS_680_HW3/celeba', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(), 
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
print(len(dataset.imgs), len(dataloader))


# In[5]:


# Fixed input for debugging
fixed_x, _ = next(iter(dataloader))
save_image(fixed_x, 'real_image.png')

Image('real_image.png')


# In[6]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# In[7]:


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)


# In[8]:


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=512, z_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
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
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
#         pdb.set_trace()
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


# In[9]:


image_channels = fixed_x.size(1)
print(image_channels)


# In[10]:


model = VAE(image_channels=image_channels).to(device)
# model.load_state_dict(torch.load('vae.torch', map_location='cpu'))


# In[11]:


# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002) 


# In[12]:


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# In[13]:


epochs = 20


# In[ ]:

itera = 0
for epoch in range(epochs):
    for idx, (images, _) in enumerate(dataloader):
        recon_images, mu, logvar = model(images.cuda())
        loss, bce, kld = loss_fn(recon_images, images.cuda(), mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                epochs, loss.data[0]/bs, bce.data[0]/bs, kld.data[0]/bs)
        print(to_print)
        if itera%10 == 0:
            n = min(images.size(0), 8)
            comparison = torch.cat([images.cuda()[:n],
                                          recon_images[:n]])
            save_image(comparison.data.cpu(),
                         './reconstructed/reconstruction_' + str(epoch) + '.png', nrow=n)

# # notify to android when finished training
# notify(to_print, priority=1)

torch.save(model.state_dict(), 'vae.torch')


# In[155]:


def compare(x):
    recon_x, _, _ = model(x)
    return torch.cat([x, recon_x])


# In[156]:


def compare2(x):
    recon_x, _, _ = model(x)
    return recon_x


# In[186]:


# sample = torch.randn(bs, 1024)
# compare_x = vae.decoder(sample)

# fixed_x, _ = next(iter(dataloader))
# fixed_x = fixed_x[:8]
import pdb
real_img = dataset[randint(1, 150)][0].unsqueeze(0)

compare_x = compare2(real_img)
for i in range(31):
    
    fixed_x = dataset[randint(1, 150)][0].unsqueeze(0)
#     pdb.set_trace()
    real_img = torch.cat([real_img, fixed_x])

    new_compare_x = compare2(fixed_x)
    
    compare_x = torch.cat([compare_x, new_compare_x])
    
    
    

save_image(compare_x.data.cpu(), 'fake_image.png')
save_image(real_img.data.cpu(), 'real_image.png')
print("Generated Images ")
display(Image('fake_image.png', width=700, unconfined=True))
print("Real Images ")
display(Image('real_image.png', width=700, unconfined=True))


# In[187]:


fixed_x = dataset[randint(2, 180)][0].unsqueeze(0)
compare_x = compare(fixed_x)

save_image(compare_x.data.cpu(), 'sample_image.png')
display(Image('sample_image.png', width=700, unconfined=True))

