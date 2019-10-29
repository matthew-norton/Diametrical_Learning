import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Simple_Net(nn.Module):
    def __init__(self,num_classes=10,input_shape=(28,28)):
        super(Simple_Net_Test, self).__init__()

        self.input_flat_size = int(np.prod(input_shape))
        self.fc0 = nn.Linear(self.input_flat_size, 320 , bias = False )

        self.fc1 = nn.Linear(320, 320, bias = False)
        self.fc2 = nn.Linear(320, 200, bias = False)

        self.fc3 = nn.Linear(200, num_classes, bias = False)



    def forward(self,x):

        x = x.view(-1, self.input_flat_size)

        x = F.relu(self.fc0(x) )
        x = F.relu(self.fc1(x) )
        x = F.relu(self.fc2(x) )

        x = self.fc3(x)

        return x
