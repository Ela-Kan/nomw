# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:12:44 2023

Utilities for the CMICHACKATHON 2023 project B
"""
import torch
import cv2
import numpy as np
# import matplotlib.pyplot as plt

def transform_I(volume):
    # applies 2DFT of the log of the image
    
    # Iterate over slices
    for slice_idx in range(volume.shape[2]):
        # Extract a 2D slice from the volume
        slice_image = volume[:, :, slice_idx]
    
        # Perform 2D Fourier Transform of the log of the image
        f_transform = np.fft.fft2(np.log1p(slice_image))
        f_transform_shifted = np.fft.fftshift(f_transform)
    
        # Calculate the magnitude spectrum
        magnitude_spectrum = np.abs(f_transform_shifted)
    
        # # Display the original slice and Fourier-transformed slice
        # plt.subplot(121), plt.imshow(slice_image, cmap='gray')
        # plt.title(f'Original Slice {slice_idx + 1}'), plt.xticks([]), plt.yticks([])
    
        # plt.subplot(122), plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
        # plt.title(f'2D Fourier Transform Slice {slice_idx + 1}'), plt.xticks([]), plt.yticks([])
    
        # plt.show()
        
        return magnitude_spectrum

def itransform_I(volume, magnitude_spectrum):
    # Iterate over slices
    for slice_idx in range(volume.shape[2]):
        # restores the volume to the original image space
        # Inverse Fourier Transform
        f_transform_shifted_inverse = np.fft.ifftshift(magnitude_spectrum)
        f_transform_inverse = np.fft.ifft2(f_transform_shifted_inverse)
        
        # Exponential to undo the logarithm
        reconstructed_image = np.expm1(np.real(f_transform_inverse))
    return reconstructed_image



def transform_V(volume):
    # applies 3DFT of the log of the volume

    # Perform 3D Fourier Transform of the log of the volume
    f_transform = np.fft.fftn(np.log1p(volume))
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    # # # Display the original and Fourier-transformed volumes
    # # Note: You may need to adjust the visualization based on the 3D nature
    # plt.subplot(121), plt.imshow(volume[:, :, volume.shape[2] // 2], cmap='gray')
    # plt.title('Original Volume'), plt.xticks([]), plt.yticks([])

    # plt.subplot(122), plt.imshow(np.log1p(magnitude_spectrum[:, :, volume.shape[2] // 2]), cmap='gray')
    # plt.title('3D Fourier Transform Volume'), plt.xticks([]), plt.yticks([])

    # plt.show()

    return magnitude_spectrum

def transform_Vtensor(volume):
    # Apply 3D Fourier Transform of the log of the volume
    f_transform = torch.fft.fftn(torch.log1p(volume))

    # Shift the zero frequency component to the center
    f_transform_shifted = torch.fft.fftshift(f_transform)

    # Calculate the magnitude spectrum
    magnitude_spectrum = torch.abs(f_transform_shifted)

    return magnitude_spectrum





def itransform_V(magnitude_spectrum):
    # restores the volume to the original image space
    # Inverse 3D Fourier Transform
    f_transform_shifted_inverse = np.fft.ifftshift(magnitude_spectrum)
    f_transform_inverse = np.fft.ifftn(f_transform_shifted_inverse)

    # Exponential to undo the logarithm
    reconstructed_volume = np.expm1(np.real(f_transform_inverse))
    return reconstructed_volume


def itransform_Vtensor(magnitude_spectrum):

    # Inverse Fourier Transform
    f_transform_shifted_inverse = torch.fft.ifftshift(magnitude_spectrum)
    f_transform_inverse = torch.fft.ifftn(f_transform_shifted_inverse)

    # Exponential to undo the logarithm
    reconstructed_volume = torch.expm1(f_transform_inverse.real)

    return reconstructed_volume

def wasserstein_loss(output, target):
    """
    Wasserstein Loss (EMD) for WGAN
    output: Discriminator output
    target: Real (1) or Fake (0) labels
    """
    return torch.mean(output * target)









