import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, 
                 img_channels: int = 1, 
                 num_subjects: int = 15) -> None:
        
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(img_channels + num_subjects, 32, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=1), 
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.Conv3d(128, 1, kernel_size=5, stride=2, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, subject_ids):
        subject_ids = subject_ids.view(subject_ids.size(0), subject_ids.size(1), 1, 1, 1)
        subject_ids = subject_ids.expand(-1, -1, x.size(2), x.size(3), x.size(4))

        # Concatenate subject_ids with the input x
        x = torch.cat([x, subject_ids], dim=1)

        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(x)
        return x