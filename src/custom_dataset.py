from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, instances, labels):
        self.instances = instances
        self.labels = labels

    def __getitem__(self, index):
        return (self.instances[index], self.labels[index])

    def __len__(self):
        return len(self.instances)
