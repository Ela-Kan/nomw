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
from monai.transforms import Compose,RandShiftIntensity, RandBiasField, RandScaleIntensity, RandAdjustContrast, ToNumpy

batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define your transformations accordingly

"""NOTE: The following transformations must be applied to a np array, but they output tensors
"""

intensity_transform = Compose([
    ToNumpy(),
    RandShiftIntensity(offsets=100, prob = 1),  # Adjust intensity by scaling with a factor of 1.5
    RandBiasField(degree = 2, prob = 1),
    RandScaleIntensity(factors=0.5, prob = 1),
    RandAdjustContrast(gamma = 2, prob = 1)])

# Create DataLoader
subject_ids = prepare_data('data\mni')
dataset = Dataset(subject_ids[:10], 'data\mni', is_motion_corrected=True)

# =============================================================================
# Split into (training and validation datasets
# =============================================================================
generator = torch.Generator()
generator.manual_seed(0)

num_training_subjects = int(0.8*len(dataset))
num_val_subjects = len(dataset) - num_training_subjects
train_set, val_set = random_split(dataset, [num_training_subjects, num_val_subjects], generator=generator)

print(f"Number of training subjects: {len(train_set)}. Number of validation subjects: {len(val_set)}.")

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size)

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
     
        # Validation loop
        gan.eval()  # Set model to evaluation mode
        total_d_val_loss = 0.0

        with torch.no_grad():  # Disable gradient computation during validation
            for (val_filtered_images, val_unfiltered_images, val_subject_ids) in val_dataloader:
                
                # move to device
                val_filtered_images = val_filtered_images.to(device)
                val_unfiltered_images = val_unfiltered_images.to(device)
                val_subject_ids = val_subject_ids.to(device)
                        
                val_is_real = Variable(torch.ones(len(val_unfiltered_images), 1))
                val_is_fake = Variable(torch.zeros(len(val_unfiltered_images), 1))
                val_gen_images = gan.generator(val_filtered_images, val_subject_ids)
                
                val_real_loss = -wasserstein_loss(gan.discriminator(val_unfiltered_images, val_subject_ids), val_subject_ids) # negative for real loss, since aiming to minimise
                val_fake_loss = wasserstein_loss(gan.discriminator(val_gen_images.detach(), val_subject_ids), val_is_fake)
                val_d_loss = (val_real_loss + val_fake_loss) / 2
                total_d_val_loss += val_d_loss.item()
                
        # Calculate and print average validation loss
        val_avg_d_loss = total_d_val_loss / len(val_dataloader)
    
    if epoch % 2 == 0:
        plt.imsave(f'generated_images/sub_{subject_ids[0]}_epoch_{epoch:02d}.png', gen_images[0,:,:,:,45].detach().numpy()[0], cmap='gray')
    print(
        "[Epoch %d/%d] [train - D loss: %f] [train - G loss: %f] [val - D loss: %f]"
        % (epoch, num_epochs, d_loss.item(), g_loss.item(), val_avg_d_loss.item())
    )