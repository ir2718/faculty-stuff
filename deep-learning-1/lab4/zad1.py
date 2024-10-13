from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from random import choice
import torchvision
import torch

class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            # Filter out images with target class equal to remove_class
            to_remove = [i for i, t in enumerate(self.targets) if remove_class == t.item()]
            keep = list(range(len(self.images)))
            keep = [i for i in keep if i not in to_remove]
            self.images = self.images[keep, :, :]
            self.targets = self.targets[keep]
                    
        
        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        target = self.targets[index]
        other = [k for k in self.target2indices.keys() if k != target]
        new_target = choice(other)
        indices = self.target2indices[new_target]
        return choice(indices)

    def _sample_positive(self, index):
        target = self.targets[index].item()
        indices = self.target2indices[target]
        return choice(indices)

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)

### test ###
# d = MNISTMetricDataset()
# pos = d._sample_positive(0)
# neg = d._sample_negative(0)

# print(d.images[neg])
# print(d.targets[neg])

# print(d.images[pos])
# print(d.targets[pos])