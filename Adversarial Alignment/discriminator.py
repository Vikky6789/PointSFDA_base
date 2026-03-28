import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarsePointDiscriminator(nn.Module):
    def __init__(self):
        super(CoarsePointDiscriminator, self).__init__()
        # SnowflakeNet ka coarse output normally 256 points ka hota hai: (Batch, 3, 256)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) # LSGAN ke liye last me sigmoid nahi lagate

    def forward(self, x):
        # PointNet Conv1D ke liye format (Batch, Channels, Points) chahiye
        if x.shape[-1] == 3:
            x = x.transpose(1, 2)
            
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        # Max pool karke global feature nikal rahe hain
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return x