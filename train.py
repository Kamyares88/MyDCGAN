import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

from model import Generator, Discriminator
from config import Config
from utils import weights_init
import torchvision.utils as vutils


def train_one_epoch(config, epoch, netG, netD, criterion, optimizerG, optimizerD, dataloader): 
    g_losses = []
    d_losses = []
    # iterate in dataloader
    for d, data in enumerate(dataloader, 0):
        #### 
        # Update D network: maximizing log(D(x)) + log(1- D(G(z)))
        ## Train with all-real batch
        netD.zero_grad()
        real_cpu = data[0].to(config.device)
        label = data[1].to(config.device)
        ## forward pass the real batch through D
        output = netD(real_cpu).view(-1)
        ## calculating loss on real batch
        errD_real = criterion(output, label.float())
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        ## Generating batch of latent vectors
        noise = torch.randn(real_cpu.size(0), config.nz, 1, 1, device=config.device)
        ## Generate fake image with G
        fake = netG(noise)
        label.fill_(0) # fake_label = 0
        ## classify the fake batch with D
        output = netD(fake.detach()).view(-1)
        ## calculate D's loss on all_fake batch
        errD_fake = criterion(output, label.float())
        # calculating the gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # sum the gradients
        errD = errD_real + errD_fake
        # update D
        optimizerD.step()

        #### Step2 maximizing log(D(G(z))) >> minimizing log(1-D(G(z)))
        netG.zero_grad()
        label.fill_(1)
        # since D is just updated, performing another forward pass into D
        output = netD(fake).view(-1)
        # G's loss
        errG = criterion(output, label.float())
        # calc. gradients
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # output trainin stats
        if d % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, config.num_epochs, d, len(dataloader),
                  errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
            )
        # Save Losses for plotting later
        g_losses.append(errG.item())
        d_losses.append(errD.item())
        #
        #if d==10:
        #    break
    return g_losses, d_losses

def train(config, netG, netD, criterion, optimizerG, optimizerD, dataloader, fixed_noise):
    img_list = []
    G_losses = []
    D_losses = []
    
    # Instantiate the Generator and Discriminator
    netG = netG.to(config.device)
    netD = netD.to(config.device)

    for epoch in range(config.num_epochs):
        g_losses, d_losses = train_one_epoch(config, epoch, netG, netD, criterion, optimizerG, optimizerD, dataloader)
        G_losses += g_losses
        D_losses += d_losses

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # plot curves
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        

        # plot image generation
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

    return netD, netG, G_losses, D_losses