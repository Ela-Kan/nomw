# This is an alternate version of train2.py on the main branch of this GitHub Repository https://github.com/Ela-Kan/nomw
# created as an attempt to optimise the U-Net architecture (found in utils.py)
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.autograd import Variable
from generator import Generator
from Discriminator import Discriminator
from utils import wasserstein_loss, transform_Vtensor, itransform_Vtensor, ResUNet3D
from Data_Loader import Dataset, prepare_data
import matplotlib.pyplot as plt
from monai.transforms import Compose,RandShiftIntensity, RandBiasField, RandScaleIntensity, RandAdjustContrast, ToNumpy
import numpy
import gc
import math
import nibabel as nib

gc.enable()

flag_FT = False
flag_augmentation = False
batch_size = 4

flag_load_weights = True
path_weights = 'resunet3d_weights.pth'

val_img = nib.load('PATH/TO/IMAGE') # Used to extract affine data for validation later

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    print(f"Number of available GPUs: {num_gpus}")

    # Print information about each GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("CUDA is not available on this machine.")

# Create train and validation datasets and dataloaders
train_set = Dataset(prepare_data('PATH/TO/TRAIN_DATA_FOLDER'), 'PATH/TO/TRAIN_DATA_FOLDER', is_motion_corrected=True)
val_set = Dataset(prepare_data('PATH/TO/VAL_DATA_FOLDER'), 'PATH/TO/VAL_DATA_FOLDER', is_motion_corrected=True)
print(f"Number of training volumes: {len(train_set)}. Number of validation volumes: {len(val_set)}.") # Sanity check tot. volumes

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False) # Shuffle not needed, no training done on val.

# Apply augmentation to training set

"""NOTE: The following transformations must be applied to a np array, but they output tensors
"""
# intensity_transform = Compose([
#     ToNumpy(),
#     RandShiftIntensity(offsets=20, prob = 1),  # Adjust intensity by scaling with a factor of 1.5
#     RandAdjustContrast(gamma = 1.05, prob = 1)
# ])

num_subjects = len(train_set)


# Training loop

train_loss_arr = [] # Empty train loss array
val_loss_arr = [] # Empty validation loss array

trainvol_len = math.ceil(len(train_set)/batch_size)
trainvol_len = int(trainvol_len)
print("Number of volumes used in training:", trainvol_len) # Actual training volumes used accounting for bsize

valvol_len = math.ceil(len(val_set)/batch_size)
valvol_len = int(valvol_len)
print("Number of volumes used in validation:", valvol_len) # Actual validation volumes used accounting for bsize

num_epochs = 300


# Instantiate the UNet model
in_channels = 1  # Number of input channels (e.g., grayscale image)
out_channels = 1  # Number of output channels (e.g., denoised image)
# unet = UNet(in_channels, out_channels)
unet = ResUNet3D(in_channels, out_channels)
#unet = torch.nn.DataParallel(unet) Does nothing to share memories between GPUs
unet.to(device)

if flag_load_weights:
    if os.path.exists(path_weights):
        print(f"Loading the weights file '{path_weights}'.")
        unet.load_state_dict(torch.load(path_weights))
    else:
        print(f"The weights file '{path_weights}' does not exist.")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=1e-5)
#scheduler = StepLR(optimizer, step_size=20, gamma=0.25) # Learning rate scheduler (UNUSED.)

# Assuming train_dataloader is an iterable
for epoch in range(num_epochs):
    start_time = time.time()

    # Create a tqdm progress bar for the training dataloader
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

    for i, (noisy_images, clean_images, train_subject_ids) in enumerate(train_dataloader):
        # # Assuming clean_images is the target size
        # target_size = clean_images.size()[2:]  # Exclude batch size and channel dimensions
        
        # # Resize denoised_images to match the target size
        # denoised_images = F.interpolate(denoised_images, size=target_size, mode='trilinear', align_corners=False)


        # Move data to device
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # Print sizes for debugging
        # print("Input size:", noisy_images.size())
        # print("Target size:", clean_images.size())
        
        if flag_FT:
            # start_time_ft = time.time()
            noisy_images = transform_Vtensor(noisy_images)
            clean_images = transform_Vtensor(clean_images)
            # elapsed_time_ft = time.time() - start_time_ft
            # print(f"Filtering took {elapsed_time_ft:.2f} seconds.")
    
        # Forward pass
        denoised_images = unet(noisy_images)
        #Should comment out for GPU memory, .png saving function for training
        #if False:
            #plt.imsave(f'generated_images/resunet_noisy_images_sub_epoch_{epoch:02d}.png', noisy_images[0,:,:,:,45].detach().cpu().numpy()[0], cmap='gray')
            #plt.imsave(f'generated_images/resunet_clean_images_sub_epoch_{epoch:02d}.png', clean_images[0,:,:,:,45].detach().cpu().numpy()[0], cmap='gray')
            #plt.imsave(f'generated_images/resunet_denoised_images_sub_epoch_{epoch:02d}.png', denoised_images[0,:,:,:,45].detach().cpu().numpy()[0], cmap='gray')
        
        # print("Output size:", denoised_images.size())

        # Calculate loss
        loss = criterion(denoised_images, clean_images)
        train_loss_arr.append(loss.item()) # Save train loss output

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training statistics
        print("[Epoch %d/%d] [Batch %d/%d] [Loss: %f]"
              % (epoch + 1, num_epochs, i + 1, len(train_dataloader), loss.item()))

    # Validation loop

    unet.eval()
    with torch.no_grad():
        
            
        for (val_noisy_images, val_clean_images, val_subject_ids) in val_dataloader:
            val_noisy_images = val_noisy_images.to(device)
            val_clean_images = val_clean_images.to(device)
            
            
            # Apply FT
            if flag_FT:
                val_noisy_images = transform_Vtensor(val_noisy_images)
                val_clean_images = transform_Vtensor(val_clean_images)
            
            val_denoised_images = unet(val_noisy_images)

            # Apply FT
            if flag_FT:
                val_denoised_images = itransform_Vtensor(val_denoised_images)
                val_clean_images = transform_Vtensor(val_clean_images)

            val_loss = criterion(val_denoised_images, val_clean_images)

            val_loss_arr.append(val_loss.item()) # Save val loss output

            print("[Validation] [Loss: %f]" % val_loss.item())

            # Nifti image saving functionality

            denoised_image = val_denoised_images[0,:,:,:,:].squeeze(dim=0) #Zeroth batch chosen, squeeze from fake 4D to real 3D
            denoised_image = denoised_image.cpu().detach().numpy() # Convert torch tensor to numpy array
            nifti_val_img = nib.Nifti1Image(denoised_image, val_img.affine) # Convert numpy array into 3D nifti slice
            nib.save(nifti_val_img, os.path.join(r'/home/d2009158/nifti_output', f'resunet_sub_{val_subject_ids[0]}_epoch_{epoch:02d}_slice_{len(val_loss_arr)}.nii.gz'))
            
            del denoised_image, nifti_val_img # Delete large objects to save memory

            # Only saves validation denoised image. Beware: only axial z=45 slice, 1st val_denoised_images dimension is batchsize desired
            plt.imsave(f'generated_images/resunet_sub_{val_subject_ids[0]}_epoch_{epoch:02d}.png', val_denoised_images[0,:,:,:,45].detach().cpu().numpy()[0], cmap='gray')
    
    #scheduler.step()

    #Adapted from https://www.geeksforgeeks.org/averaging-over-every-n-elements-of-a-numpy-array/
    # Saves space separated average training and validation losses for each epoch
    print("Average training loss thus far:", numpy.average(numpy.array(train_loss_arr).reshape(-1,trainvol_len), axis=1))
    print("Average validation loss thus far:", numpy.average(numpy.array(val_loss_arr).reshape(-1,valvol_len), axis=1))
    
    unet.train()

    # Memory monitoring + management below
    print(torch.cuda.memory_summary(device=None, abbreviated=False)) # Memory monitor before GC + cache clear
    torch.cuda.empty_cache()
    gc.collect()
    print("Cache cleared, gc.collect() used")
    # Close the tqdm progress bar
    progress_bar.close()
    print(torch.cuda.memory_summary(device=None, abbreviated=False)) # Memory monitor after GC + cache clear

    # Calculate and print the time taken for the epoch
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch + 1} took {elapsed_time:.2f} seconds.")

    # Commented out for GPU saving
    #if False:
        #plt.imsave(f'generated_images/resunet_noisy_images_sub_epoch_{epoch:02d}.png', noisy_images[0,:,:,:,45].detach().cpu().numpy()[0], cmap='gray')
        #plt.imsave(f'generated_images/resunet_clean_images_sub_epoch_{epoch:02d}.png', clean_images[0,:,:,:,45].detach().cpu().numpy()[0], cmap='gray')
        #plt.imsave(f'generated_images/resunet_denoised_images_sub_epoch_{epoch:02d}.png', denoised_images[0,:,:,:,45].detach().cpu().numpy()[0], cmap='gray')

def convert(s):
    """Convert space separated string numbers into array of floats. Expects string list as argument."""
    a = []
    for i in s.split():
        a.append(float(i))

    return numpy.array(a)

# Final average training and validation loss arrays in numpy array format
print("Training loss array:", convert(numpy.average(numpy.array(train_loss_arr).reshape(-1,trainvol_len), axis=1)))
print("Validation loss array:", convert(numpy.average(numpy.array(val_loss_arr).reshape(-1,valvol_len), axis=1)))

# Save the model weights
#torch.save(unet.state_dict(), 'resunet3d_weights.pth') Commented out for now