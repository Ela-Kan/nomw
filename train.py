import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.autograd import Variable
from generator import Generator
from Discriminator import Discriminator
from utils import wasserstein_loss, transform_V
from Data_Loader import Dataset, prepare_data
import matplotlib.pyplot as plt

batch_size = 16
num_subjects = 1 

generator = Generator(num_subjects=num_subjects)
discriminator = Discriminator(num_subjects=num_subjects)

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x, subject_id):
        generated_images = self.generator(x, subject_id)
        discriminator_output = self.discriminator(generated_images, subject_id)
        return generated_images, discriminator_output

gan = GAN(generator, discriminator)

# Define the optimizers
optimizer_G = optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define your transformations accordingly

# transform = v2.Compose([
#     # v2.RandomApply([v2.GaussianNoise(var_limit=(0, 0.1))], p=0.5),
#     v2.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=(0.5, 1.5))], p=0.5),
#     # v2.RandomApply([v2.RandomAdjustContrast(contrast_factor=(0.5, 1.5))], p=0.5),
#     # v2.RandomApply([v2.RandomAdjustBrightness(brightness_factor=(0.5, 1.5))], p=0.5),
#     # v2.RandomApply([v2.RandomGamma()], p=0.5),
#     v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 5))], p=0.5),
#  ])

# Create DataLoader
subject_ids = prepare_data('data/original')
dataset = Dataset(subject_ids, 'data/original')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (filtered_images, unfiltered_images, subject_ids) in enumerate(dataloader):
        
        print(filtered_images.shape)       
        # Adversarial ground truths
        is_real = Variable(torch.ones(len(dataset), 1))
        is_fake = Variable(torch.zeros(len(dataset), 1))

        # Optional: Apply FT
        
        # Train Generator
        optimizer_G.zero_grad()
        gen_images = gan.generator(filtered_images, subject_ids)
        g_loss = -wasserstein_loss(gan.discriminator(gen_images, subject_ids), is_real) # negative for real loss, since aiming to minimise
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = -wasserstein_loss(gan.discriminator(unfiltered_images, subject_ids), is_real) # negative for real loss, since aiming to minimise
        fake_loss = wasserstein_loss(gan.discriminator(gen_images.detach(), subject_ids), is_fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Clip discriminator weights (important for stability in WGAN)
        for param in gan.discriminator.parameters():
            param.data.clamp_(-0.01, 0.01)

        if i % 10 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
