## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()   
        
        # Define the first convolutional layer with 16 features
        self.conv1 = nn.Conv2d(1, 4, 5) # 1 x 224 x 224 => 4 x 220 x 220
        
        # Define the first pooling layer kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2) # 4 x 220 x 220 => 4 x 110 x 110

        # Define the first dropout layer
        self.drop_1 = nn.Dropout(0.8)

        #################################################################

        # Define the second convolutional layer with 16 features
        self.conv2 = nn.Conv2d(4, 8, 3) # 4 x 110 x 110 => 8 x 108 x 108
        
        # Define the second pooling layer kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2) # 8 x 108 x 108 => 8 x 54 x 54
        
        # Define the second dropout layer
        self.drop_2 = nn.Dropout(0.85)
        
        #################################################################
        
        # Define the third convolutional layer with 16 features
        self.conv3 = nn.Conv2d(8, 16, 3) # 8 x 54 x 54 => 16 x 52 x 52
        
        # Define the third pooling layer kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(2, 2) # 16 x 52 x 52 => 16 x 26 x 26
        
        # Define the third dropout layer
        self.drop_3 = nn.Dropout(0.9)
        
        #################################################################
        
        # Define the fourth convolutional layer with 16 features
        self.conv4 = nn.Conv2d(16, 32, 3) # 16 x 26 x 26 => 32 x 24 x 24
        
        # Define the fourth pooling layer kernel_size=2, stride=2
        self.pool4 = nn.MaxPool2d(2, 2) # 32 x 24 x 24 => 32 x 12 x 12
        
        # Define the fourth dropout layer
        self.drop_4 = nn.Dropout(0.92)
        
        #################################################################
        
        # Define the first full connected layer
        self.fc1 = nn.Linear(32*12*12, 512)
        
        # Define the fifith dropout layer
        self.drop_5 = nn.Dropout(0.49)
        
        # Define the second full connected layer
        self.fc2 = nn.Linear(512, 136)
        
        
    # Define the feedforward behavior    
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop_1(x)
        
        ##################
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop_2(x)
        
        ##################

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop_3(x)
        
        ##################
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.drop_4(x)
        
        ##################
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.drop_5(x)
        x = self.fc2(x)
        
        return x
