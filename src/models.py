from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F

# Logistic Regression
def logistic_regression():
    return LogisticRegression()

# Support Vector Machine
def svm_classifier():
    return SVC()

# Feed-Forward Neural Network
import torch.nn.functional as F
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.3)  # Dropout rate of 30%

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))  # Output layer with sigmoid for binary classification
        return x
