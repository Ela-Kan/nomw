# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:12:44 2023

Utilities for the CMICHACKATHON 2023 project B
"""
import torch
# import cv2
import numpy as np
import torch.nn as nn
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


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet3D, self).__init__()

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv3D(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv3D(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec3 = DoubleConv3D(512, 256)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec2 = DoubleConv3D(256, 128)
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dec1 = DoubleConv3D(128, 64)

        # Output layer
        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        self.pad = nn.ReplicationPad3d([2,1,3,2,2,1])  # Adjust the padding as needed

    def forward(self, x):
        
        #apply padding
        # x = self.pad(x)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((enc3[:, :, :dec3.size(2), :dec3.size(3), :dec3.size(4)], dec3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((enc2[:, :, :dec2.size(2), :dec2.size(3), :dec2.size(4)], dec2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((enc1[:, :, :dec1.size(2), :dec1.size(3), :dec1.size(4)], dec1), dim=1)
        dec1 = self.dec1(dec1)
        
        

        # Output layer
        out = self.out_conv(dec1)
        out = self.pad(out)  # Apply padding

        return out
