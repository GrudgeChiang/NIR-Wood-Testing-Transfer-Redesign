from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch


class dataset_load(Dataset):
    def __init__(self,NIRdata, label, transform=None):
        self.NIR_data = np.array(pd.read_excel(NIRdata, engine='openpyxl'))
        self.NIR_data_info = np.array(pd.read_excel(label, engine='openpyxl'))
        self.label_info = self.NIR_data_info[:, 4]
        self.label_name = self.NIR_data_info[:, 0]
        self.transform = transform

    def __getitem__(self, index):
        NIRdata = self.NIR_data[:, index * 1: index * 1 + 1]
        label = self.label_info[index]

        if self.transform is not None:
            NIRdata = self.transform(NIRdata)
        return NIRdata, label

    def __len__(self):
        return len(self.label_name)


def dataset_random_split(full_dataset, train_size = 0.7):
    train_size = int(train_size * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_dataset_loader(train_dataset, test_dataset, batch_size=4, num_workers=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=True, drop_last=True)
    return train_loader, test_loader

def dataset_to_loader(full_dataset, train_size = 0.7, batch_size=4, num_workers=4):
    train_dataset, test_dataset = dataset_random_split(full_dataset, train_size)
    train_loader, test_loader = get_dataset_loader(train_dataset, test_dataset,
                                                           batch_size, num_workers)
    return train_loader, test_loader