import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
from torch.autograd import Variable
from generator import Generator
from Discriminator import Discriminator

# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define the loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for GANs
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define your transformations accordingly

transform = v2.Compose([
    v2.ToTensor(),
    v2.RandomApply([v2.GaussianNoise(var_limit=(0, 0.1))], p=0.5),
    v2.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=(0.5, 1.5))], p=0.5),
    v2.RandomApply([v2.RandomAdjustContrast(contrast_factor=(0.5, 1.5))], p=0.5),
    v2.RandomApply([v2.RandomAdjustBrightness(brightness_factor=(0.5, 1.5))], p=0.5),
    v2.RandomApply([v2.RandomGamma()], p=0.5),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 5))], p=0.5),
 ])

# Create DataLoader
# dataset = TensorDataset(filtered_data, unfiltered_data)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (filtered_images, unfiltered_images) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(torch.ones(filtered_images.size(0), 1))
        fake = Variable(torch.zeros(filtered_images.size(0), 1))

        # Train Generator
        optimizer_G.zero_grad()
        gen_images = generator(filtered_images)
        g_loss = criterion(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(unfiltered_images), valid)
        fake_loss = criterion(discriminator(gen_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        if i % 10 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
