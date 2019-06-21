import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.nn import functional as F
import sys
import cv2

# LeNet gained from Github, traind from scratch on MNIST
# Achieved the accuracy of 99.03%


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# hyperparameters
batch_size = 128
epochs_upper_bound = 10
# data
normalize = transforms.Normalize((0.1307,), (0.3081,))
transform = transforms.Compose([transforms.ToTensor(),normalize])
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
model.load_state_dict(torch.load('LeNet_MNIST_params.pkl', map_location=torch.device('cpu')))


# Train
for epoch in range(1, epochs_upper_bound + 1):
        train(model, device, trainloader, optimizer, epoch)
        test(model, device, testloader)
torch.save(model.state_dict(),"LeNet_MNIST_params.pkl")

# Test
test(model,device,testloader)
    
