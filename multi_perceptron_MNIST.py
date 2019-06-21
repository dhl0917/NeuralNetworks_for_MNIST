import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import sys
os.environ['CUDA_VISIBLE_DIVICES'] = '0'

# Three layers neural network with one hidden layer
# trained on MNIST from scratch
# Train accuracy: 100%    Test accuracy: 97.8%

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Net(nn.Module):
    def __init__(self):
        # One hidden layer
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


max_epochs = 10
batch_size = 16
# data
normalize = transforms.Normalize((0.1307,), (0.3081,))
transform = transforms.Compose([transforms.ToTensor(), normalize])
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8)
# network
net = Net()
net.to(device)
net.load_state_dict(torch.load('multi_perceptron_MNIST_params.pkl', map_location=torch.device('cpu')))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001,
                      momentum=0.9, weight_decay=0.0005)
#lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#    optimizer, max_epochs*len(trainloader))

def test():
    hit, total = 0, 0
    for i, (data, label) in enumerate(testloader):
        with torch.no_grad():
            data = data.to(device).view(data.shape[0], -1)
            label = label.to(device)
            output = net(data)
            hit += (output.topk(k=1, dim=1)[1]
                    == label.view(-1, 1)).sum().item()
            total += data.shape[0]
    acc = hit/total
    print(
        'Test accuracy: %.3f\t' % (acc)
    )

def train():
    max_acc = -1.0
    for epoch in range(max_epochs):
        # train
        for i, (data, label) in enumerate(trainloader):
            #data = data.cuda(non_blocking=True).view(data.shape[0], -1)
            #label = label.cuda(non_blocking=True)
            data = data.to(device).view(data.shape[0], -1)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()

        acc = (output.topk(k=1, dim=1)[1] ==
            label.view(-1, 1)).float().mean().item()
        print('Epoch: %d\tlr:%f\tTrain ACC: %.3f'
            % (epoch+1, optimizer.param_groups[0]['lr'], acc))
        # validation
        hit, total = 0, 0
        for i, (data, label) in enumerate(testloader):
            with torch.no_grad():
                data = data.to(device).view(data.shape[0], -1)
                label = label.to(device)
                output = net(data)
                hit += (output.topk(k=1, dim=1)[1]
                        == label.view(-1, 1)).sum().item()
                total += data.shape[0]
        acc = hit/total
        if acc > max_acc:
            max_acc = acc
        print(
            'Epoch: %d\tValidation accuracy: %.3f\tmax_acc: %.3f' % (
                epoch+1, acc, max_acc)
        )

train()
torch.save(net.state_dict(),"multi_perceptron_MNIST_params.pkl")
test()