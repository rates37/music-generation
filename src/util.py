
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



def concat_features(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concats x and y along the feature map axis.  Refer to Figure 1 from MidiNet paper
        and the extract of paper below:
        
        " Assuming that the conditioning vector has length n, to add it to an intermediate
            layer of shape a-by-b we can duplicate the values ab times to get a tensor of 
            shape a-by-b-by-n, and then concatenate it with the intermediate layer in the
            feature map axis. "

    Args:
        x (torch.Tensor): A tensor of shape [batchSize, m, a, b]
        y (torch.Tensor): A tensor of shape [batchSize, n, 1, 1]

    Returns:
        torch.Tensor: The concatenation of x and y, duplicating the values of y along the feature map axis,
                        of shape [batchSize, m+n, a, b]
    """
    # x is [batchSize, m, a, b]
    # y is [batchSize, n, 1, 1]
    
    # reshape y to [batchSize, n, a, b] 
    expandedY = y.expand(x.shape[0], y.shape[1], x.shape[2], x.shape[3])
    print(expandedY.shape)
    # concat expaded y with x along the feature map axis (axis 1)
    return torch.cat((x, expandedY), 1)


if __name__ == "__main__":
    # testing some stuff:
    x = torch.ones([3,9,2,7])
    y = torch.ones([3,8,1,1])
    z = concat_features(x,y)
    print(z.shape)