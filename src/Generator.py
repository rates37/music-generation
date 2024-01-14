
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from util import *
from typing import Tuple


class Generator(nn.Module):
    """Implements the Generator network from the MidiNet Paper
    """
    
    def __init__(self) -> None:
        """Generator constructor
        """
        # call parent class' constructor:
        super(Generator, self).__init__()
        
        # hyper-parameters:
        self.leakyReluSlope = 0.2
    
        # layers for generator:
        self.gConv1: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=141, # 128 + 13 -> range of pitch notes + 13-dimensional chord encoding
            out_channels=128,
            kernel_size=(2,1),
            stride=(2,2)
        )
        
        self.gConv2: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=141,
            out_channels=128,
            kernel_size=(2,1),
            stride=(2,2)
        )
        
        self.gConv3: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=141,
            out_channels=128,
            kernel_size=(2,1),
            stride=(2,2)
        )
        
        self.gConv4: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=157,
            out_channels=1,
            kernel_size=(1, 128),
            stride=(1,2)
        )
        
        self.gFc1: nn.Linear = nn.Linear(
            in_features=113,
            out_features=1024
        )

        self.gFc2: nn.Linear = nn.Linear(
            in_features=1037,  # 1024 + 13
            out_features=256
        )

        self.batchNorm1D1: nn.BatchNorm1d = nn.BatchNorm1d(num_features=1024)
        
        self.batchNorm1D2: nn.BatchNorm1d = nn.BatchNorm1d(num_features=256)

        self.gBatchNorm2D: nn.BatchNorm2d = nn.BatchNorm2d(num_features=128)

        
        # layers for conditioner CNN (for Minimax approach):
        self.cConv1: nn.Conv2d = nn.Conv2d(
            in_channels=1, 
            out_channels=16,
            kernel_size=(1,128),
            stride=(1,2)
        )
                
        self.cConv2: nn.Conv2d = nn.Conv2d(
            in_channels=16, 
            out_channels=16,
            kernel_size=(2,1),
            stride=(2,2)
        )
    
        self.cConv3: nn.Conv2d = nn.Conv2d(
            in_channels=16, 
            out_channels=16,
            kernel_size=(2,1),
            stride=(2,2)
        )
        
        self.cConv4: nn.Conv2d = nn.Conv2d(
            in_channels=16, 
            out_channels=16,
            kernel_size=(2,1),
            stride=(2,2)
        )

        self.cBatchNorm2D: nn.BatchNorm2d = nn.BatchNorm2d(num_features=16)
    
    def forward(self, z: torch.Tensor, previousX: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generator forward method

        Args:
            z (torch.Tensor): A 1x100 tensor of random noise from which we will generate the output
            previousX (torch.Tensor): A tensor of shape [batchSize, 1, 16, 128], representing the previous bars of the song
            y (torch.Tensor): A 13-dimensional chord encoding (y.shape = [batchSize, 13])

        Returns:
            torch.Tensor: The next generated bars
        """
        
        
        # compute conditioner CNN outputs:
        conditionerOutput1: torch.Tensor = self.cConv1(previousX)  # conditionerOutput1.shape is [batchSize, 16, 16, 1]
        conditionerOutput1 = self.cBatchNorm2D(conditionerOutput1)  # conditionerOutput1.shape is now: [batchSize, 16, 16, 1]
        conditionerOutput1 = F.leaky_relu(conditionerOutput1, negative_slope=self.leakyReluSlope)  # conditionerOutput1.shape is now: [batchSize, 16, 16, 1]
        
        conditionerOutput2: torch.Tensor = self.cConv2(conditionerOutput1)  # conditionerOutput2.shape is [batchSize, 16, 8, 1]
        conditionerOutput2 = self.cBatchNorm2D(conditionerOutput2)  # conditionerOutput2.shape is now: [batchSize, 16, 8, 1]
        conditionerOutput2 = F.leaky_relu(conditionerOutput2, negative_slope=self.leakyReluSlope)  # conditionerOutput2.shape is now: [batchSize, 16, 8, 1]
        
        conditionerOutput3: torch.Tensor = self.cConv3(conditionerOutput2)  # conditionerOutput3.shape is [batchSize, 16, 4, 1]
        conditionerOutput3 = self.cBatchNorm2D(conditionerOutput3)  # conditionerOutput3.shape is now: [batchSize, 16, 4, 1]
        conditionerOutput3 = F.leaky_relu(conditionerOutput3, negative_slope=self.leakyReluSlope)  # conditionerOutput3.shape is now: [batchSize, 16, 4, 1]
        
        conditionerOutput4: torch.Tensor = self.cConv4(conditionerOutput3)  # conditionerOutput4.shape is [batchSize, 16, 2, 1]
        conditionerOutput4 = self.cBatchNorm2D(conditionerOutput4)  # conditionerOutput4.shape is now: [batchSize, 16, 2, 1]
        conditionerOutput4 = F.leaky_relu(conditionerOutput4, negative_slope=self.leakyReluSlope)  # conditionerOutput4.shape is now: [batchSize, 16, 2, 1]
        
        
        # compute generator output:
        yReshaped: torch.Tensor = y.view(z.shape[0], 13, 1, 1)  # yReshaped.shape is [batchSize, 13, 1, 1]
        
        # concat z and y together (noise + chord encoding)
        z = torch.cat((z,y), 1)  # z.shape is now: [batchSize, 100+13] = [batchSize, 100+13]
        
        
        # pass through the first linear layer:
        z = self.gFc1(z)  # z.shape is now: [batchSize, 1024]
        # apply batchnorm:
        z = self.batchNorm1D1(z)  # z.shape is now: [batchSize, 1024]
        # apply relu:
        z = F.relu(z)  # z.shape is now: [batchSize, 1024]
        # concat z with y:
        z = torch.cat((z, y), 1)  # z.shape is now: [batchSize, 1037]
        
        # pass through the second linear layer:
        z = self.gFc2(z)  # z.shape is now: [batchSize, 256]
        # apply batchnorm:
        z = self.batchNorm1D2(z)  # z.shape is now: [batchSize, 256]
        # apply relu:
        z = F.relu(z)  # z.shape is now: [batchSize, 256]
        
        
        # reshape into a 4-dimensional tensor for use with the Transpose Conv Layers:
        z = z.view(z.shape[0], 128, 2, 1)  # z.shape is now: [batchSize, 128, 2, 1]
        
        # concat with yReshaped:
        z = concat_features(z, yReshaped)  # z.shape is now: [batchSize, 141, 2, 1]
        
        
        # pass through the first transpose conv layer:
        z = self.gConv1(z)  # z.shape is now: [batchSize, 128, 4, 1]
        # apply batchnorm:
        z = self.gBatchNorm2D(z)  # z.shape is now: [batchSize, 128, 4, 1]
        # apply relu:
        z = F.relu(z)  # z.shape is now: [batchSize, 128, 4, 1]
        # concat with yReshaped:
        z = concat_features(z, yReshaped)  # z.shape is now: [batchSize, 141, 4, 1]
        
        
        # pass through the second transpose conv layer:
        z = self.gConv2(z)  # z.shape is now: [batchSize, 128, 8, 1]
        # apply batchnorm:
        z = self.gBatchNorm2D(z)  # z.shape is now: [batchSize, 128, 8, 1]
        # apply relu:
        z = F.relu(z)  # z.shape is now: [batchSize, 128, 8, 1]
        # concat with yReshaped:
        z = concat_features(z, yReshaped)  # z.shape is now: [batchSize, 141, 8, 1]
        
        
        # pass through the third transpose conv layer:
        z = self.gConv3(z)  # z.shape is now: [batchSize, 128, 16, 1]
        # apply batchnorm:
        z = self.gBatchNorm2D(z)  # z.shape is now: [batchSize, 128, 16, 1]
        # apply relu:
        z = F.relu(z)  # z.shape is now: [batchSize, 128, 16, 1]
        # concat with yReshaped:
        z = concat_features(z, yReshaped)  # z.shape is now: [batchSize, 141, 16, 1]
        # concat with first hidden output of conditioner CNN:
        z = concat_features(z, conditionerOutput1)  # z.shape is now: [batchSize, 157, 16, 1]
        
        # pass through the fourth transpose conv layer:
        z = self.gConv4(z)  # z.shape is now: [batchSize, 1, 16, 128]
        
        # return sigmoid of z:
        return F.sigmoid(z)


if __name__ == "__main__":
    # testing:
    g = Generator()
    z = torch.ones([3, 100])  # batchSize 3
    y = torch.ones([3, 13])
    previousX = torch.ones([3, 1, 16, 128])
    preds = g(z, previousX, y)