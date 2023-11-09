# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:12:44 2023

Utilities for the CMICHACKATHON 2023 project B
"""




import cv2
import numpy as np
# import matplotlib.pyplot as plt

# Load the 3D brain volume
volume = cv2.imread('path_to_your_3d_volume.nii', cv2.IMREAD_GRAYSCALE)

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

def itransform_I(magnitude_spectrum):
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

def itransform_V(magnitude_spectrum):
    # restores the volume to the original image space
    # Inverse 3D Fourier Transform
    f_transform_shifted_inverse = np.fft.ifftshift(magnitude_spectrum)
    f_transform_inverse = np.fft.ifftn(f_transform_shifted_inverse)

    # Exponential to undo the logarithm
    reconstructed_volume = np.expm1(np.real(f_transform_inverse))

    # # Display the original and reconstructed volumes
    # # Note: You may need to adjust the visualization based on the 3D nature
    # plt.subplot(121), plt.imshow(np.log1p(magnitude_spectrum[:, :, volume.shape[2] // 2]), cmap='gray')
    # plt.title('3D Fourier Transform Volume'), plt.xticks([]), plt.yticks([])

    # plt.subplot(122), plt.imshow(reconstructed_volume[:, :, volume.shape[2] // 2], cmap='gray')
    # plt.title('Reconstructed Volume'), plt.xticks([]), plt.yticks([])

    # plt.show()

    return reconstructed_volume









