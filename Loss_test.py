import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import datasets
import torch.nn.functional as F
import params
from torch.autograd import Variable

import New_Loss
import ETD_model


class Extractor_src(nn.Module):
    def __init__(self):
        super(Extractor_src, self).__init__()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 20)
        self.bn3 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 20)
        self.bn4 = nn.BatchNorm1d(20)

    def forward(self, input):
        x = self.fc1(input)
        x = self.bn3(x)
        x = self.fc2(F.relu((self.conv2_drop(x))))
        x = self.bn4(x)
        return x


class Extractor_tgt(nn.Module):
    def __init__(self):
        super(Extractor_tgt, self).__init__()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(117, 20)
        self.bn3 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 20)
        self.bn4 = nn.BatchNorm1d(20)

    def forward(self, input):
        x = self.fc1(input)
        x = self.bn3(x)
        x = self.fc2(F.relu((self.conv2_drop(x))))
        x = self.bn4(x)
        return x


class Regression_net(nn.Module):
    def __init__(self):
        super(Regression_net, self).__init__()
        self.fc3 = nn.Linear(20, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, input):
        result = self.fc3(input)
        sigmod_result = self.sigmod(result)
        return result, sigmod_result


class Regression_mix_net(nn.Module):
    def __init__(self):
        super(Regression_mix_net, self).__init__()
        self.fc3 = nn.Linear(20, 20)

    def forward(self, input):
        logits_src = self.fc3(input[:, :, 0])
        logits_tgt = self.fc3(input[:, :, 1])
        return logits_src, logits_tgt


class tgt_dataset(Dataset):
    def __init__(self, transform=None):
        self.NIR_data = np.array(pd.read_excel('NIRtargetdataTensile.xlsx', engine='openpyxl'))
        self.NIR_data_info = np.array(pd.read_excel('NIRtragetlabelTensile.xlsx', engine='openpyxl'))
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


class src_dataset(Dataset):
    def __init__(self, transform=None):
        self.NIR_data = np.array(pd.read_excel('NIRdataTensile.xlsx', engine='openpyxl'))
        self.NIR_data_info = np.array(pd.read_excel('NIRlabelTensile.xlsx', engine='openpyxl'))
        self.label_info = np.repeat(self.NIR_data_info[:, 4], 4, axis=0)
        self.label_name = np.repeat(self.NIR_data_info[:, 0], 4, axis=0)
        self.transform = transform

    def __getitem__(self, index):
        NIRdata = self.NIR_data[:, index * 1: index * 1 + 1]
        label = self.label_info[index]

        if self.transform is not None:
            NIRdata = self.transform(NIRdata)
        return NIRdata, label

    def __len__(self):
        return len(self.label_name)


def dataset_random_split(full_dataset, train_size=0.7):
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


def dataset_to_loader(full_dataset, train_size=0.7, batch_size=4, num_workers=4):
    train_dataset, test_dataset = dataset_random_split(full_dataset, train_size)
    train_loader, test_loader = get_dataset_loader(train_dataset, test_dataset,
                                                   batch_size, num_workers)
    return train_loader, test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    full_src_dataset = src_dataset()
    src_train_loader, src_test_loader = dataset_to_loader(full_src_dataset, 0.7)
    print(len(src_train_loader), len(src_test_loader))

    full_tgt_dataset = tgt_dataset()
    tgt_train_loader, tgt_test_loader = dataset_to_loader(full_tgt_dataset, 0.7)
    print(len(tgt_train_loader), len(tgt_test_loader))

    common_src_net = Extractor_src().to(device)
    common_tgt_net = Extractor_tgt().to(device)
    reg_src_net = Regression_net().to(device)
    reg_tgt_net = Regression_net().to(device)
    reg_net = Regression_mix_net().to(device)
    reg_src_net_enhenced = Regression_net().to(device)
    reg_tgt_net_enhenced = Regression_net().to(device)
    reg_net_enhenced = Regression_mix_net().to(device)
    reg_net_enhenced_att = Regression_mix_net().to(device)

    # Extractor.train()
    mmd_optimizer = optim.RMSprop([{'params': common_src_net.parameters()},
                                   {'params': common_tgt_net.parameters()},
                                   {'params': reg_src_net.parameters()},
                                   {'params': reg_tgt_net.parameters()},
                                   {'params': reg_net.parameters()}],
                                  lr=params.lr, momentum=params.momentum)

    optimizer_ot = optim.Adam([{'params': reg_net_enhenced.parameters()},
                               {'params': reg_tgt_net_enhenced.parameters()},
                               {'params': reg_src_net_enhenced.parameters()},
                               {'params': reg_net_enhenced_att.parameters()}],
                              lr=params.lr, weight_decay=0.01)

    src_dataiter = iter(src_train_loader)
    src_imgs, src_labels = next(src_dataiter)
    tgt_dataiter = iter(tgt_train_loader)
    tgt_imgs, tgt_labels = next(tgt_dataiter)

    src_samples = Variable(src_imgs.view(4, -1).float().cuda())
    src_features = common_src_net(src_samples)
    tgt_samples = Variable(tgt_imgs.view(4, -1).float().cuda())
    tgt_features = common_tgt_net(tgt_samples)
    features = torch.stack((src_features, tgt_features), dim=2)
    src_mix_result, tgt_mix_result = reg_net(features)
    mmd_loss = MMD(src_mix_result, tgt_mix_result, "rbf")

    src_result, src_sig_result = reg_src_net(src_mix_result)
    src_sig_labels = torch.sigmoid(src_labels)
    tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
    tgt_sig_labels = torch.sigmoid(tgt_labels)

    src_mix_result_enhenced, tgt_mix_result_enhenced = reg_net_enhenced(features)
    features_att = torch.stack((src_mix_result_enhenced, tgt_mix_result_enhenced), dim=2)
    src_mix_result_att, tgt_mix_result_att = reg_net_enhenced_att(features_att)
    src_result_enhenced, src_sig_result_enhenced = reg_src_net_enhenced(src_mix_result_att)
    tgt_result_enhenced, tgt_sig_result_enhenced = reg_tgt_net_enhenced(tgt_mix_result_att)
    mmd_loss_enhenced = MMD(src_mix_result_enhenced, tgt_mix_result_enhenced, "rbf")

    # new_loss 各种新loss
    sym_loss = New_Loss.symmetric_mse_loss(src_mix_result_enhenced, tgt_mix_result_enhenced)
    cor_loss = New_Loss.coral_loss(src_mix_result_enhenced, tgt_mix_result_enhenced)
    log_cor_loss = New_Loss.log_coral_loss(src_mix_result_enhenced, tgt_mix_result_enhenced)
    l2dist = New_Loss.l2dist(src_mix_result_enhenced, tgt_mix_result_enhenced)
    print(l2dist)