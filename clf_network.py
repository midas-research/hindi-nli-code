import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn

class Classifier_Net(nn.Module):
    def __init__(self, n_classes):
        super(Classifier_Net, self).__init__()
        self.n_classes = n_classes
        self.input_dim = 1024

        self.classifier = nn.Sequential(
            
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_classes),
            )

    def forward(self,vector):
        output = self.classifier(vector)
        return output
    
    
        
    
        
        
        
        
        
        
        
        
