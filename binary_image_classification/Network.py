import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, USE_CUDA, DROP_OUT=0):
        super(Network, self).__init__()
        self.USE_CUDA = USE_CUDA
        self.DROP_OUT = DROP_OUT
        self.cnn1 = nn.Conv2d(3, 8, kernel_size=(3, 3)) #3,128
        self.norm1 = nn.BatchNorm2d(8) #128
        self.cnn2 = nn.Conv2d(8, 8, 3) #128,128
        self.norm2 = nn.BatchNorm2d(8) #128
        self.cnn3 = nn.Conv2d(8, 16, 3)#128,64
        self.norm3 = nn.BatchNorm2d(16) #64
        self.fc1 = nn.Linear(16 * 4 * 10, 128) #64*4*10
        self.fc2 = nn.Linear(128, 1) 
        self.dropout = nn.Dropout2d(p=DROP_OUT)


    def net_forward(self, x):
        output = f.max_pool2d(f.relu(self.cnn1(x)), 3)
        output = f.max_pool2d(f.relu(self.cnn2(output)), 3)
        output = self.norm2(output)
        output = f.max_pool2d(f.relu(self.cnn3(output)), 3)

        output = output.view(output.size()[0], -1)

        if self.DROP_OUT > 0:
            output = self.dropout(output)

        output = f.relu(self.fc1(output))

        if self.DROP_OUT > 0:
            output = self.dropout(output)

        output = f.sigmoid(self.fc2(output))
        return output
    
    
