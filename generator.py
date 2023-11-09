# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn

class Generator(nn.Module):
    r""" It is mainly based on the UNET network as a backbone network generator.

    Args:
        image_size (int): The size of the image. (Default: 28)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """

    def __init__(self, 
                 image_size: tuple=(64,64,64), 
                 in_channels: int = 1, 
                 embedding_dim: int=16,
                 out_channels: int=1,
                 num_subjects: int = 10) -> None:
        
        super(Generator, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels

        self.label_embedding = nn.Embedding(num_subjects, embedding_dim)

        self.encoder = nn.Sequential(
                    nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                    # Add more layers as needed
                )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            # Add more layers as needed
        )

        # Decoder (Expansive Path)
        self.decoder = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, out_channels, kernel_size=2, stride=2)
            # Add more layers as needed
        )
  
        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
        """
        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.

        Returns:
            A four-dimensional vector (N*C*H*W).
        """

        conditional_inputs = torch.cat([inputs, self.label_embedding(labels)], dim=-1)
        out = self.model(conditional_inputs)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)