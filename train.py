import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torch.autograd import Variable
from generator import Generator
from Discriminator import Discriminator
from utils import wasserstein_loss, transform_V
from Data_Loader import Dataset, prepare_data
import matplotlib.pyplot as plt
from monai.transforms import Compose,RandShiftIntensity, RandBiasField, RandScaleIntensity, RandAdjustContrast

batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define your transformations accordingly

"""NOTE: The following transformations must be applied to a np array, but they output tensors. adjust prob
intensity_transform = Compose([
    RandShiftIntensity(offsets=100, prob = 0.2),  # Adjust intensity by scaling with a factor of 1.5
    RandBiasField(degree = 2, prob = 0.2),
    RandScaleIntensity(factors=0.5, prob = 0.2),
    RandAdjustContrast(gamma = 2, prob = 0.2)
])

"""
# Create DataLoader
subject_ids = prepare_data('data\mni')
dataset = Dataset([subject_ids[:10]], 'data\mni', is_motion_corrected=True)




# =============================================================================
# Split into (training and validation datasets
# =============================================================================
generator = torch.Generator()
generator.manual_seed(0)

train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


num_subjects = len(subject_ids)
generator = Generator(num_subjects=num_subjects).to(device)
discriminator = Discriminator(num_subjects=num_subjects).to(device)

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x, subject_id):
        generated_images = self.generator(x, subject_id)
        discriminator_output = self.discriminator(generated_images, subject_id)
        return generated_images, discriminator_output

gan = GAN(generator, discriminator).to(device)

# Define the optimizers
optimizer_G = optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, (filtered_images, unfiltered_images, subject_ids) in enumerate(train_dataloader):
        
        # move all data to deice
        filtered_images = filtered_images.to(device)
        unfiltered_images = unfiltered_images.to(device)
        subject_ids = subject_ids.to(device)
        
        # Adversarial ground truths
        is_real = Variable(torch.ones(len(unfiltered_images), 1))
        is_fake = Variable(torch.zeros(len(unfiltered_images), 1))

        # Optional: Apply FT
        
        # Train Generator
        optimizer_G.zero_grad()
        gen_images = gan.generator(filtered_images, subject_ids)
        # print(gen_images.shape)
        # print(plt.imshow(gen_images[0,:,:,:,45].detach().numpy()[0], cmap='gray'))
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

    if epoch % 2 == 0:
        plt.imsave(f'gen_epoch_{epoch:02d}.png', gen_images[0,:,:,:,45].detach().numpy()[0], cmap='gray')
    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, num_epochs, d_loss.item(), g_loss.item())
    )

