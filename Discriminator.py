import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, 
                 img_shape = (91,109,91),
                 img_channels: int = 1, 
                 num_subjects: int = 10,
                 embed_dim = 1) -> None:
        
        super(Discriminator, self).__init__()
        
        self.embedding = nn.Embedding(num_subjects, embed_dim)
        
        self.model = nn.Sequential(
            nn.Conv3d(img_channels + embed_dim, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
            nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.25),
        )
        self.num_output_features = self._calculate_output_features((img_channels + embed_dim, img_shape[0], img_shape[1], img_shape[2]))

        self.fc = nn.Sequential(
            nn.Linear(self.num_output_features , 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
            )

    def _calculate_output_features(self, input_shape):
        # Function to calculate the number of features after convolutions
        with torch.no_grad():
            return self._forward_conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def _forward_conv(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
    def forward(self, inputs, subject_ids):
        
        subject_embedding = self.embedding(subject_ids)
        # Expand and concatenate subject_ids with the input x
        subject_embedding = subject_embedding.view(subject_embedding.size(0), subject_embedding.size(1), 1, 1, 1)
        subject_embedding = subject_embedding.expand(-1, -1, inputs.size(2), inputs.size(3), inputs.size(4))
        
        # Concatenate subject_ids with the input x
        x = torch.cat([inputs, subject_embedding], dim=1)
        x = self.model(x)
        x = x.view(x.size(0), -1) # flatten before fully connected layer
        x = self.fc(x)
        return x