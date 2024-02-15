import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

from util import *
from typing import Tuple

from Generator_Model_1 import GeneratorModel1
from Discriminator_Model_1 import DiscriminatorModel1


class MusicDataSet(Dataset):
    def __init__(self, dataPath: str, prevDataPath: str) -> None:
        self.data = torch.from_numpy(np.load(dataPath)).float()
        self.prevData = torch.from_numpy(np.load(prevDataPath)).float()
        
    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index], self.prevData[index]

def train(disc: DiscriminatorModel1, gen: GeneratorModel1, epochs: int = 10, lr: float=0.0002) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    disc.train()
    disc.to(device)
    gen.train()
    gen.to(device)
    discOptimiser = torch.optim.Adam(disc.parameters(), lr=lr)
    genOptimiser = torch.optim.Adam(gen.parameters(), lr=lr)
    
    batchSize = 72  # not mentioned in paper (?)
    
    trainLoader = DataLoader(MusicDataSet('data/data_x_augmented.npy', 'data/prev_x_augmented.npy'), batch_size=batchSize, shuffle=True)
    
    
    sumReductionLoss = nn.MSELoss(reduction='sum')
    bceLoss = nn.BCEWithLogitsLoss()
    l2Loss = lambda x, y: sumReductionLoss(x,y)/2
    lambda1 = 0.1
    lambda2 = 1
    
    
    for epoch in range(epochs):
        for i, (data, prevData) in enumerate(trainLoader):
            # print(data.shape, prevData.shape)
            data = torch.reshape(data, [data.shape[0], 1, data.shape[1], data.shape[2]])
            prevData = torch.reshape(prevData, [prevData.shape[0], 1, prevData.shape[1], prevData.shape[2]])
            # print(data.shape, prevData.shape)
            # exit()
            #! 1. Update Discriminator network
            batchSize = data.shape[0]
            # with real data:
            disc.zero_grad()
            data = data.to(device)
            prevData = prevData.to(device)
            labels = torch.ones((batchSize), device=device)
            
            probsReal, predsReal, matchedFeaturesReal = disc.forward(data)
            
            # compute loss:
            discLossReal = torch.mean(torch.mean(bceLoss(probsReal, torch.ones(probsReal.shape).to(device)), 0), -1)
            discLossReal.backward(retain_graph=True)
            
            # with generator output:
            noise = torch.randn((batchSize, 100)).to(device)
            generatorOutput = gen.forward(noise, prevData)
            labels = torch.zeros((batchSize), device=device)
            probsFake, predsFake, matchedFeaturesFake = disc.forward(generatorOutput.detach())
            discLossFake = torch.mean(torch.mean(bceLoss(probsFake, torch.zeros(probsFake.shape)).to(device), 0), -1)
            if epoch == 0:  #only train disc for 1 epoch
                discLossFake.backward(retain_graph=True)
                
                # update discriminator weights:
                discOptimiser.step()

            #! 2. Update Generator network
            gen.zero_grad()
            probsFake, predsFake, matchedFeaturesFake = disc.forward(generatorOutput.clone())  # changed from detach to clone
            generatorLoss = torch.mean(torch.mean(bceLoss(predsFake, torch.ones(predsFake.shape).to(device)), 0), -1)
            
            # implement feature matching
            featuresFromGenerator = torch.mean(matchedFeaturesFake, 0)
            featuresFromReal = torch.mean(matchedFeaturesReal.detach(), 0)
            featureMatchedLoss = torch.mul(l2Loss(featuresFromGenerator, featuresFromReal), lambda1)

            averageFromGenerator = torch.mean(generatorOutput, 0)
            averageFromReal = torch.mean(data)
            averageFeatureMatchedLoss = torch.mul(l2Loss(averageFromGenerator, averageFromReal), lambda2)
            
            generatorError = generatorLoss + featureMatchedLoss + averageFeatureMatchedLoss
            generatorError.backward(retain_graph=True)
            genOptimiser.step()

            #! 3. Update Generator network again
            gen.zero_grad()
            generatorOutputClone = generatorOutput.detach().clone()
            probsFake, predsFake, matchedFeaturesFake = disc.forward(generatorOutputClone)
            generatorLoss = torch.mean(torch.mean(bceLoss(predsFake, torch.ones(predsFake.shape).to(device)), 0), -1)
            
            # implement feature matching
            featuresFromGenerator = torch.mean(matchedFeaturesFake, 0)
            featuresFromReal = torch.mean(matchedFeaturesReal.detach(), 0)
            featureMatchedLoss = torch.mul(l2Loss(featuresFromGenerator, featuresFromReal), lambda1)

            averageFromGenerator = torch.mean(generatorOutputClone, 0)
            averageFromReal = torch.mean(data)
            averageFeatureMatchedLoss = torch.mul(l2Loss(averageFromGenerator, averageFromReal.detach()), lambda2)
            
            generatorError = generatorLoss + featureMatchedLoss + averageFeatureMatchedLoss
            generatorError.backward(retain_graph=True)
            genOptimiser.step()
            
            
        # print losses:
        if epoch % 1 == 0:
            print(f"Epoch: [{epoch+1}]\n\tGenerator Loss: [{generatorLoss}]\n\tDiscriminator Loss: [{discLossFake+discLossReal}]")    
        
        # save model:
        torch.save(disc.state_dict(), f"model_checkpoints/discriminator_epoch_{epoch}.pth")
        torch.save(gen.state_dict(), f"model_checkpoints/generator_epoch_{epoch}.pth")
            
            




if __name__ == "__main__":
    
    # instantiate models:
    generator = GeneratorModel1()
    
    discriminator = DiscriminatorModel1()
    # with torch.autograd.set_detect_anomaly(True):
    train(discriminator, generator)
    