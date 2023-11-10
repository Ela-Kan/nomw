import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, 
                 #image_size: tuple=(64,64,64), 
                 img_channels: int = 1, 
                 num_subjects: int = 15) -> None:
        
        super(Discriminator, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Conv3d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, kernel_size=4, stride=2, padding=1),
        )
        self.condition_embedding = nn.Embedding(num_subjects, 256)
        self.condition_linear = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, subject_ids):
        x = self.image_branch(x)
        condition = self.condition_embedding(subject_ids)
        condition = self.condition_linear(condition)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, condition], dim=1)
        x = self.sigmoid(x)
        return x