from nn import draw_conv_filters
import torch
import torchvision
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 50
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-2
NUM_EPOCHS = 2

class CovolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, in_channels2, conv2_width, flattened_dim,  fc1_width, class_count):
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels2, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(flattened_dim, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.relu(x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.relu(x)
        logits = self.fc_logits(x)
        return logits

    
    def train_model(self, train_dataloader, test_dataloader, optimizer, num_epochs):
        d_train, d_test = {}, {}
        for e in range(num_epochs):
            train_loss, test_loss = [], []
            
            self.train()
            for i, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()

                y_pred = self(x)
                loss = nn.functional.cross_entropy(y_pred, y)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
    
                # mozda ce radit
                draw_conv_filters(e, i*len(y), self.conv1.weight.detach().numpy(), './conv_filters')

            self.eval()            
            for x, y in test_dataloader:
                y_pred = self(x)
                loss = nn.functional.cross_entropy(y_pred, y)
                test_loss.append(loss.item())
            
            d_train[e] = np.mean(train_loss)
            d_test[e] = np.mean(test_loss)

            print(f'Epoch {e}: train_loss - {train_loss} test_loss - {test_loss}')
    
        self.d_train = d_train
        self.d_test = d_test

    def draw_loss(self):
        plt.plot(list(self.d_train.keys()), list(self.d_train.values()), '-o', label='Training loss')
        plt.plot(list(self.d_test.keys()), list(self.d_test.values()), '-o', label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss compared to epoch')

def main():
    train = torchvision.datasets.MNIST('/mnist/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    test = torchvision.datasets.MNIST('/mnist/', train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    model = CovolutionalModel()
    model.train_model(train_dataloader, test_dataloader, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, num_epochs=NUM_EPOCHS))
    model.draw_loss()


main()