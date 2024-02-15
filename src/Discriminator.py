
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from util import *
from typing import Tuple

class Discriminator(nn.Module):
    """Implements the Discriminator network from the MidiNet Paper
    """
    
    def __init__(self) -> None:
        """Discriminator Constructor
        """
        # call parent class' constructor:
        super(Discriminator, self).__init__()
        
        # hyper-parameters:
        self.leakyReluSlope: float = 0.2
        
        # layers: (conv filter shapes from page 5 of MidiNet paper)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=14, 
            out_channels=14, 
            kernel_size=(2, 128),  #  range of different possible pitches is 128
            stride=(2,2)
        )
        
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=27,   # output of conv1 concatenated feature axis-wise with y
            out_channels=77,
            kernel_size=(4,1),
            stride=(2,2)
        )
        
        self.fc1 : nn.Linear = nn.Linear(
            in_features=244,
            out_features=1024
        )
        
        self.fc2 : nn.Linear = nn.Linear(
            in_features=1037,  # 1024 + shape of y since y is concatenated with output of fc1
            out_features=1
        )

        self.batchNorm1D: nn.BatchNorm1d = nn.BatchNorm1d(num_features=1024)

        self.batchNorm2D: nn.BatchNorm2d = nn.BatchNorm2d(num_features=77)
    

    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
        """Passes fake/real data as well as the chord representation 

        Args:
            x (torch.Tensor): The real or fake song data
            y (torch.Tensor): The 13-dimensional chord representation

        Returns:
            Tuple[torch.Tensor]: A tuple of:
                - Probability vector of predictions
                - Raw predictions
                - A copy of the first conv layer's outputs to use for feature mapping
        """
        
        # x.shape is: [batchSize, 1, 16, 128]
        # y.shape is: [batchSize, 13]

        # expand y to be the same number of dimensions as x:
        yReshaped: torch.Tensor = y.view(x.shape[0], 13, 1, 1)  # since 13-dimensional chord encoding is used
        
        # concat y and x together:
        x = concat_features(x, yReshaped)  # x.shape is now: [batchSize, 14, 16 ,128]
        
        
        # pass through first conv layer:
        x = self.conv1(x)  # x.shape is now: [batchSize, 14, 8, 1]
        # apply leaky relu
        x = F.leaky_relu(x, negative_slope=self.leakyReluSlope)
        
        # save x, to implement feature matching in later passes: (refer to MidiNet paper, page 2, second paragraph from bottom on left)
        matchedFeatures = x
        
        # concat y with the output of conv1:
        x = concat_features(x, yReshaped)  # x.shape is now: [batchSize, 27, 16 ,128]
        

        # pass through the second conv layer:
        x = self.conv2(x)  # x.shape is now: [batchSize, 77, 3, 1]
        # pass through batchnorm:
        x = self.batchNorm2D(x)  # x.shape is now: [batchSize, 77, 3, 1]
        # apply leaky relu:
        x = F.leaky_relu(x, negative_slope=self.leakyReluSlope)  # x.shape is now: [batchSize, 77, 3, 1]
        
        
        # convert to 1D tensor:
        x = x.view(x.shape[0], -1)  # x.shape is now [batchSize, 77*3*1] = [batchSize, 231]
        
        # concat y onto x:
        x = torch.cat((x, y), 1)  # x.shape is now [batchSize, 231+13] = [batchSize, 244]
        
        
        # pass through first fully-connected layer:
        x = self.fc1(x)  # x.shape is now [batchSize, 1024]
        # apply batchnorm:
        x = self.batchNorm1D(x)  # x.shape is now [batchSize, 1024]
        # apply leaky relu:
        x = F.leaky_relu(x, self.leakyReluSlope)  # x.shape is now [batchSize, 1024]
        
        # concat y onto x:
        x = torch.cat((x, y), 1)  # x.shape is now [batchSize, 1024+13] = [batchSize, 1037]
        
        
        # pass through second fully-connected layer:
        x = self.fc2(x)  # x.shape is now [batchSize, 1]
        
        
        # return sigmoid of preds, preds, and featureMatches:
        return F.sigmoid(x), x, matchedFeatures


if __name__ == "__main__":
    d = Discriminator()
    x = torch.ones([3, 1, 16 ,128])
    y = torch.ones([3, 13])
    probs, preds, fm = d(x, y)
    print(probs.shape)
    print(preds.shape)
    print(fm.shape)