import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torch.autograd import Variable
import argparse, os
from generator import Generator
from Discriminator import Discriminator
from utils import wasserstein_loss, transform_Vtensor, itransform_Vtensor
from Data_Loader import Dataset, prepare_data
import matplotlib.pyplot as plt
from monai.transforms import Compose,RandShiftIntensity, RandBiasField, RandScaleIntensity, RandAdjustContrast, ToNumpy, Transform
from monai.transforms.spatial.functional import resize

def main(args):
    
    data_dir = args.data_dir
    flag_FT = args.fourier_transform
    flag_augmentation = args.aug

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
    ## Define training variables
    batch_size = 2
     
    # Create train and validation datasets and dataloaders
    train_set = Dataset(prepare_data(os.path.join(data_dir, 'mni_train'))[:20], os.path.join(data_dir, 'mni_train'), is_motion_corrected=True)
    val_set = Dataset(prepare_data(os.path.join(data_dir, 'mni_val')), os.path.join(data_dir, 'mni_val'), is_motion_corrected=True)
    print(f"Number of training volumes: {len(train_set)}. Number of validation volumes: {len(val_set)}.")

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        
    """NOTE: The following transformations must be applied to a np array, but they output tensors
    """
    intensity_transform = Compose([
        ToNumpy(),
        RandShiftIntensity(offsets=20, prob = 0.5),  # Adjust intensity by scaling with a factor of 1.5
        RandAdjustContrast(gamma = 1.05, prob = 0.5)
    ])

    generator = Generator(num_subjects=len(train_set)+len(val_set)).to(device)
    discriminator = Discriminator(num_subjects=len(train_set)+len(val_set)).to(device)

    # Define the optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'min')
    scheduler_D = ReduceLROnPlateau(optimizer_D, 'min')

    # Training loop
    num_epochs = 1
    best_val_loss = float('inf')
    early_stop_patience = 10
    early_stop_counter = 0

    lambda_mse = 0.4
    criterion_GAN = nn.BCELoss()  # Your GAN loss criterion
    criterion_MSE = nn.MSELoss()  # Mean Squared Error loss

    for epoch in range(num_epochs):
        
        generator.train()
        discriminator.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_MSE = 0.0
        
        for i, (filtered_images, unfiltered_images, train_subject_ids) in enumerate(train_dataloader):
            
            if flag_augmentation:
                # if augmentation is enabled, apply it to the filtered images
                filtered_images = intensity_transform(filtered_images)
            
            # Optional: Apply FT
            if flag_FT:
                # for aux_batch in range(filtered_images.shape[0]):
                filtered_images= transform_Vtensor(filtered_images)
                unfiltered_images = transform_Vtensor(unfiltered_images)
        
            # move all data to device
            filtered_images = filtered_images.to(device)
            unfiltered_images = unfiltered_images.to(device)
            train_subject_ids = train_subject_ids.to(device)

            # Adversarial ground truths
            is_real = Variable(torch.ones(len(unfiltered_images), 1)).to(device)
            is_fake = Variable(torch.zeros(len(unfiltered_images), 1)).to(device)
        
            # ### Train Generator ###
            optimizer_G.zero_grad()
            gen_images = generator(filtered_images, train_subject_ids)

            # wasserstein loss
            # g_loss = -wasserstein_loss(discriminator(gen_images, train_subject_ids), is_real) # negative for real loss, since aiming to minimise
            # bce loss
            g_loss = criterion_GAN(discriminator(gen_images, train_subject_ids), is_real) 
            mse_loss = criterion_MSE(gen_images, unfiltered_images)
            g_loss+= lambda_mse * mse_loss  # lambda_mse is a weighting factor
        
            g_loss.backward()
            optimizer_G.step()
            
            running_loss_G += g_loss.item()
            running_loss_MSE += mse_loss.item()
            
            ## Train Discriminator ##
            optimizer_D.zero_grad()
            
            # wasserstein loss
            # d_loss_real = -wasserstein_loss(discriminator(unfiltered_images, train_subject_ids), is_real) # negative for real loss, since aiming to minimise
            # d_loss_fake = wasserstein_loss(discriminator(gen_images.detach(), train_subject_ids), is_fake)
            
            # bce loss
            d_loss_real = criterion_GAN(discriminator(unfiltered_images, train_subject_ids), is_real)
            d_loss_fake = criterion_GAN(discriminator(gen_images.detach(),train_subject_ids), is_fake)
            
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            optimizer_D.step()
            running_loss_D += d_loss.item()

            # Clip discriminator weights (important for stability in WGAN)
            for param in discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)
        
            # Validation loop
            generator.eval()  # Set model to evaluation mode
            discriminator.eval()
            val_running_loss_G = 0.0
            val_running_loss_MSE = 0.0
            val_running_loss_D = 0.0
            
            with torch.no_grad():  # Disable gradient computation during validation
                for (val_filtered_images, val_unfiltered_images, val_subject_ids) in val_dataloader:
                    
                    # Apply FT
                    if flag_FT:
                        # for aux_batch in range(filtered_images.shape[0]):
                        val_filtered_images= transform_Vtensor(val_filtered_images)
                        val_unfiltered_images = transform_Vtensor(val_unfiltered_images)
                
                    # move to device
                    val_filtered_images = val_filtered_images.to(device)
                    val_unfiltered_images = val_unfiltered_images.to(device)
                    val_subject_ids = val_subject_ids.to(device)
                        
                    val_is_real = Variable(torch.ones(len(val_unfiltered_images), 1)).to(device)
                    val_is_fake = Variable(torch.zeros(len(val_unfiltered_images), 1)).to(device)
                    
                    val_gen_images = generator(val_filtered_images, val_subject_ids)
                    
                    # compute generator loss
                    val_g_loss = criterion_GAN(discriminator(val_gen_images, val_subject_ids), val_is_real) 
                    val_mse_loss = criterion_MSE(val_gen_images, val_unfiltered_images)
                    val_g_loss+= lambda_mse * val_mse_loss  # lambda_mse is a weighting factor
                    val_running_loss_G += val_g_loss.item()
                    val_running_loss_MSE += val_mse_loss.item()

                    # compute discriminator loss
                    # val_d_loss_real = -wasserstein_loss(discriminator(val_unfiltered_images, val_subject_ids), val_is_real) # negative for real loss, since aiming to minimise
                    # val_d_loss_fake = wasserstein_loss(discriminator(val_gen_images.detach(), val_subject_ids), val_is_fake)
                    
                    val_d_loss_real = criterion_GAN(discriminator(val_unfiltered_images, val_subject_ids), val_is_real)
                    val_d_loss_fake = criterion_GAN(discriminator(val_gen_images.detach(),val_subject_ids), val_is_fake)
            
                    val_d_loss = 0.5 * (val_d_loss_real + val_d_loss_fake)
                    val_running_loss_D += val_d_loss.item()
                    
                    
            # Calculate and print average validation loss
            val_average_loss_D = val_running_loss_D / len(val_dataloader)
            val_average_loss_G = val_running_loss_G / len(val_dataloader)
            val_average_loss_MSE = val_running_loss_MSE / len(val_dataloader)

        if epoch % 2 == 0:
                
            # apply inverse fourier transform to generated validation images in image space
            if flag_FT:
                val_gen_images = itransform_Vtensor(val_gen_images)
  
            plt.imsave(f'generated_images/sub_{val_subject_ids[0]}_epoch_{epoch:02d}.png', val_gen_images[0,:,:,:,45].detach().cpu().numpy()[0] if device.type == 'cuda' else val_gen_images[0,:,:,:,45].detach().numpy()[0], cmap='gray')
        
        average_loss_G = running_loss_G / len(train_dataloader)
        average_loss_D = running_loss_D / len(train_dataloader)
        average_loss_MSE = running_loss_MSE / len(train_dataloader)
        
        # early stopping
        if val_average_loss_D < best_val_loss:
            best_val_loss = val_average_loss_D
            # save model
            torch.save(generator.state_dict(), 'generator.pth')
            torch.save(discriminator.state_dict(), 'discriminator.pth')
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1  # Increment early stopping counter if no improvement
    
        scheduler_G.step(val_average_loss_G)
        scheduler_D.step(val_average_loss_D)

        print("[Epoch %d/%d] [train - D loss: %f] [train - G loss: %f] [train - MSE loss: %f] [val - D loss: %f] [val - G loss: %f] [val - MSE loss: %f]"
            % (epoch, num_epochs, average_loss_D.item(), average_loss_G.item(), average_loss_MSE.item(), val_average_loss_D, val_average_loss_G, val_average_loss_MSE), flush=True)
        
        # Check for early stopping
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered! No improvement for {} epochs.".format(early_stop_patience))
            break  # Stop training

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a GAN for cmic_hacks project. Example usage: python3 train.py /path/to/data/directory --aug [OPTIONAL] --fourier_transform [OPTIONAL]')
    
    parser.add_argument('data_dir', help='path to data directory.')
    parser.add_argument('--cpu', action='store_true', default=False, 
                        help='Force to use cpu.')  
    parser.add_argument('--aug', action='store_true', default=False, 
                        help='Turn on data augmentaion.')  
    parser.add_argument('--fourier_transform', action='store_true', default=False, 
                        help='Turn on fourier transform.')  
    
    args = parser.parse_args()
    main(args)

