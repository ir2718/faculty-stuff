import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.reshape((img.shape[0], -1))
        return feats


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(num_maps_in))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(
            in_channels=num_maps_in, 
            out_channels=num_maps_out, 
            kernel_size=k, 
            bias=bias)
        )

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.bnreluconv1 = _BNReluConv(input_channels, emb_size)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bnreluconv2 = _BNReluConv(emb_size, emb_size)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bnreluconv3 = _BNReluConv(emb_size, emb_size)

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        x = self.bnreluconv1(img)
        x = self.mp1(x)

        x = self.bnreluconv2(x)
        x = self.mp2(x)

        x = self.bnreluconv3(x)

        x = torch.mean(x, (2,3), True)
        x = torch.reshape(x, shape=(x.shape[0], self.emb_size))
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        margin = 1.0
        tmp = torch.linalg.norm(a_x-p_x, dim=1) - \
              torch.linalg.norm(a_x-n_x, dim=1) + margin
        loss = torch.sum(torch.maximum(tmp, torch.zeros(tmp.shape)))
        return loss


##### izmjereni rezultati ######
# Epoch 0: Test Accuracy: 97.42%
# Epoch 1: Test Accuracy: 97.81%
# Epoch 2: Test Accuracy: 98.17%

# Epoch 0: Test Accuracy: 82.16% <--- IdentityModel