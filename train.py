import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch.autograd import Variable
from generator import Generator
from Discriminator import Discriminator
from utils import wasserstein_loss

batch_size = 16
image_size = (64,64,64)
num_subjects = 1 

generator = Generator(image_size=image_size, num_subjects=num_subjects)
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

gan = GAN(image_size)

# Define the optimizers
optimizer_G = optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Assuming you have loaded your data into filtered_data and unfiltered_data tensors
# You need to organize your data into batches using DataLoader
# Define your transformations accordingly

# Example:
transform = transforms.Compose([
    
    # Add other transformations as needed
])

# Create DataLoader
dataset = TensorDataset(filtered_data, unfiltered_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (filtered_images, unfiltered_images, subject_ids) in enumerate(dataloader):
                            
        # Adversarial ground truths
        is_real = Variable(torch.ones(len(dataset), 1))
        is_fake = Variable(torch.zeros(len(dataset), 1))

        # Optional: Apply FT
        
        # Train Generator
        optimizer_G.zero_grad()
        gen_images = gan.generator(filtered_images)
        g_loss = -wasserstein_loss(gan.discriminator(gen_images), is_real) # negative for real loss, since aiming to minimise
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = -wasserstein_loss(gan.discriminator(unfiltered_images), is_real) # negative for real loss, since aiming to minimise
        fake_loss = wasserstein_loss(gan.discriminator(gen_images.detach()), is_fake)
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
