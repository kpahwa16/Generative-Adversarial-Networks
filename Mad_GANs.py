#!/usr/bin/env python
# coding: utf-8

# ## MAD GAN

# In[1]:


# Initialization of libraries
import torch
import torch.nn
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import random
from random import randint
from sklearn.mixture import GMM
device = torch.device('cuda')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# defining parameters for the training
mb_size = 128 # Batch Size
Z_dim = 64  # Length of noise vector
X_dim = 1  # Input Length
h_dim = 128  # Hidden Dimension
lr = 1e-4    # Learning Rate
num_gen = 4


# In[3]:


np.random.seed(1)
gmm = GMM(5)
gmm.means_ = np.array([[10], [20], [60], [80], [110]])
gmm.covars_ = np.array([[3], [3], [2], [2], [1]]) ** 2
gmm.weights_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

X = gmm.sample(200000)
data = X
data = (data - X.min())/(X.max()-X.min())
plt.hist(data, 200000, normed=False, histtype='stepfilled', alpha=1)


# In[4]:


G = []
for i in range(num_gen):
    G.append(torch.nn.Sequential(
        torch.nn.Linear(Z_dim, h_dim),
        torch.nn.PReLU(),
        torch.nn.Linear(h_dim, h_dim),
        torch.nn.PReLU(),
        torch.nn.Linear(h_dim, X_dim),
        torch.nn.Sigmoid()
    ).cuda())

D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(h_dim, h_dim),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(h_dim, num_gen + 1),
    torch.nn.Softmax()
).cuda()


# In[5]:


G_solver = []
for i in range(num_gen):
    G_solver.append(optim.Adam(G[i].parameters(), lr))
D_solver = optim.Adam(D.parameters(), lr)
###
loss = nn.CrossEntropyLoss()
label_G = Variable(torch.LongTensor(mb_size))
label_G = label_G.to(device)
label_D = Variable(torch.LongTensor(mb_size))
label_D = label_D.to(device)


# In[6]:


# Reset the gradients to zero
params = [G[0], G[1], G[2], G[3], D]
def reset_grad():
    for net in params:
        net.zero_grad()
reset_grad()


# In[7]:


data_index = 0
for it in range(198000):
    if ((data_index + 1)*mb_size>len(data)):
        data_index = 0

    X.view(mb_size, 1)
    X = X.type(torch.FloatTensor)
    X = X. X = torch.from_numpy(np.array(data[data_index*mb_size : (data_index + 1)*mb_size]))
    X =to(device)
    Total_D_loss = 0
    for i in range(num_gen):
        # Dicriminator forward-loss-backward-update
        #forward pass
        z = torch.FloatTensor(mb_size, Z_dim).uniform_(-1, 1)
        z = z.to(device)
        G_sample = G[i](z)
        D_real = D(X)
        D_fake = D(G_sample)
        # Calculate the loss
        D_loss_real = loss(D_real,label_D.fill_(num_gen + 0.1*randint(-1,1)))
        D_loss_fake = loss(D_fake, label_G.fill_(i + 0.1*randint(-1,1)))
        D_loss = D_loss_real + D_loss_fake
        Total_D_loss = D_loss + Total_D_loss
        # Calulate and update gradients of discriminator
        D_loss.backward()
        D_solver.step()

        # reset gradient
        reset_grad()

    # Generator forward-loss-backward-update
    
    Total_G_loss = 0
    for i in range(num_gen):
        
        z = torch.FloatTensor(mb_size, Z_dim).uniform_(-1, 1)
        z = z.to(device)
        G_sample = G[i](z)
        D_fake = D(G_sample)

        G_loss = loss(D_fake, label_D.fill_(num_gen + 0.1*randint(-1,1)))
        Total_G_loss = G_loss + Total_G_loss
        G_loss.backward()
        G_solver[i].step()

        # reset gradient
        reset_grad()
        
    data_index = data_index + 1
    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, Total_D_loss.data.cpu().numpy(), Total_G_loss.data.cpu().numpy()))


# In[11]:


import numpy as np
final = np.zeros(1500*mb_size, dtype = float)
for i in range(1500):
    z = torch.FloatTensor(64, Z_dim).uniform_(-1, 1)
    z = z.to(device)
    l = G[randint(0,num_gen-1)](z).cpu().detach().numpy()
    final[i*mb_size : ((i+ 1)*mb_size -1)] = l[0]
p1 = plt.hist(final, 200, normed=True, histtype='bar', alpha=0.5)
p2 = plt.hist(data, 200, normed=True, histtype='bar', alpha=0.5)


# ## Points to ponder
# 1. What happens if we reduce the number of generator?
# 2. What happens if we change the learning rate and other parameters?
# 3. What happens if we change the noise distribution?
