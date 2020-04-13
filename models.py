## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_dropout = nn.Dropout(0.2)
        self.dropout_d1 = nn.Dropout(0.3)
        self.dropout_d2 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(73728, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 136)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_dropout(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_dropout(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv_dropout(x)
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_d1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_d2(x)
        x = self.fc3(x)        
        return x