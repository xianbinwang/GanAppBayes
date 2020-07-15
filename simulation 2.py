import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
x1 = torch.FloatTensor([0.5,1]).view(-1,1)
x2 = torch.FloatTensor([-0.5,1]).view(-1,1)
x3 = torch.FloatTensor([1,1]).view(-1,1)

BATCH_SIZE = 1000
LR = 0.0001
N_IDEA = 3
def prior_sample():
    a = torch.randn(BATCH_SIZE,2) * np.sqrt(2)
    return a

G = nn.Sequential(
        nn.Linear(N_IDEA,40),

        nn.Linear(40,2),
        )

D = nn.Sequential(
        nn.Linear(2,20),
        nn.Tanh(),
        nn.Linear(20,40),
        nn.Tanh(),
        nn.Linear(40,1),
        nn.Sigmoid(),
        )

opt_G = torch.optim.Adam(G.parameters(),lr=LR)
opt_D = torch.optim.Adam(D.parameters(),lr=LR)
plt.ion()
for epoch in range(10000):
    sample_prior = prior_sample()
    idea = torch.randn(BATCH_SIZE,N_IDEA)
    sample_generated = G(idea)
    prob_true = D(sample_generated)
    prob_false = D(sample_prior)
    
    D_loss = -torch.mean(torch.log(prob_true)) - torch.mean(torch.log(1-prob_false))
    G_loss = torch.mean(torch.log(prob_true/(1-prob_true)))  + torch.mean((0.5-torch.mm(sample_generated,x1))**2) + torch.mean((-1-torch.mm(sample_generated,x2))**2) +torch.mean((1-torch.mm(sample_generated,x3))**2)
    opt_D.zero_grad()
    D_loss.backward(retain_graph = True)
    opt_D.step()
    
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    if epoch % 100 == 0:
        plt.cla()
        idea = torch.randn(1000,N_IDEA)
        sample_generated = G(idea)
        x = np.linspace(-1,3,100)
        y = np.linspace(-2,2,100)
        X,Y = np.meshgrid(x,y)
        z = -1/4*(X**2+Y**2) - (0.5-0.5*X-Y)**2 - (-1+0.5*X-Y)**2 - (1-X-Y)**2
        plt.contour(X,Y,z,20,colors ='k')
        plt.scatter(sample_generated[:,0].data.numpy(),sample_generated[:,1].data.numpy())
        plt.pause(0.05)
plt.show() 
        


