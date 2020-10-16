import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class NLI_HYPOTHS_Net(nn.Module):
    def __init__(self):
        super(NLI_HYPOTHS_Net, self).__init__()

        self.n_classes = 1

        self.classifier = nn.Sequential(
            nn.Linear(4*1024, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, self.n_classes),
            )

    def forward(self, c_emb, h_emb):
        c1 = c_emb
        c2 = h_emb
        
        c3 = torch.abs(c_emb-h_emb)
        c4 = c_emb*h_emb

        features = torch.cat([c1, c2], dim=1)
        features = torch.cat([features, c3], dim=1)
        features = torch.cat([features, c4], dim=1)
        features = features.view(features.shape[0], -1)
        output = self.classifier(features)
        
        return output

class NLI_Reverse_Net(nn.Module):
    def __init__(self, n_classes):
        super(NLI_Reverse_Net, self).__init__()

        self.n_classes = n_classes
        self.nli_classes = 2

        self.reverse_classifier = nn.Sequential(
            nn.Linear(self.nli_classes*self.n_classes*2, 8),
            nn.Linear(8, self.n_classes)
            )

    def forward(self, e_emb):
        output = self.reverse_classifier(e_emb)
        return output

class NLI_Classification_Net(nn.Module):
    def __init__(self, n_classes):
        super(NLI_Classification_Net, self).__init__()

        self.nli_classes = 2
        self.sentiment_classes = n_classes

        self.softmax = nn.Softmax(dim=1)

        self.nli_classifier = nn.Sequential(
            nn.Linear(4*768, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.nli_classes),
            )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(self.nli_classes*self.sentiment_classes*2 + 768, 8),
            nn.Linear(8, self.sentiment_classes)
            )


    def forward(self, c_emb, h_emb):
        c1 = c_emb
        c2 = h_emb
        
        c3 = torch.abs(c_emb-h_emb)
        c4 =c_emb*h_emb

        features = torch.cat([c1, c2], dim=1)
        features = torch.cat([features, c3], dim=1)
        features = torch.cat([features, c4], dim=1)
        features = features.view(features.shape[0], -1)
        
        nli_output = self.nli_classifier(features)
        nli_probs = self.softmax(nli_output)
        nli_probs_1 = nli_probs.view(nli_probs.shape[0] // (self.nli_classes * self.sentiment_classes), (self.nli_classes * self.sentiment_classes) * nli_probs.shape[1])
        
        premise_reduced_batch = torch.zeros(nli_probs_1.shape[0], 768)
        k = 0
        for i in range(c_emb.shape[0]):
            if(i%(self.nli_classes * self.sentiment_classes)==0):
                premise_reduced_batch[k, :] = c_emb[i, :]
                k += 1
        concat_inpput = torch.cat([premise_reduced_batch.cuda(), nli_probs_1], dim=1)
        clf_output = self.sentiment_classifier(concat_input)
        return nli_output, clf_output

class NLI_Reverse_Net_Exact(nn.Module):
    def __init__(self, n_classes):
        super(NLI_Reverse_Net_Exact, self).__init__()

        self.n_classes = n_classes
        self.nli_classes = 2

        self.reverse_classifier = nn.Sequential(
            nn.Linear(self.nli_classes*self.n_classes, 8),
            nn.Linear(8, self.n_classes)
            )

    def forward(self, e_emb):
        output = self.reverse_classifier(e_emb)
        return output
