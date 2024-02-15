import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import params
import Network
import Propress_Fun
import New_Loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(src_loader, tgt_loader, common_src_net, common_tgt_net, reg_src_net, reg_tgt_net, reg_net, optimizer,
                src_criterion, tgt_criterion, epoch):
    common_src_net.train()
    common_tgt_net.train()
    reg_src_net.train()
    reg_tgt_net.train()
    reg_net.train()

    for _ in range(epoch):
        train_mmd_loss = 0
        train_reg_loss = 0
        for idx, (src_imgs, src_labels) in enumerate(src_loader):
            src_samples = Variable(src_imgs.view(4, -1).float().to(device))
            src_features = common_src_net(src_samples)
            tgt_imgs, tgt_labels = next(iter(tgt_loader))
            tgt_samples = Variable(tgt_imgs.view(4, -1).float().to(device))
            tgt_features = common_tgt_net(tgt_samples)

            features = torch.stack((src_features, tgt_features), dim=2)
            src_mix_result, tgt_mix_result = reg_net(features)

            mmd_loss = New_Loss.MMD(src_mix_result, tgt_mix_result, "rbf")

            src_result, _ = reg_src_net(src_mix_result)
            tgt_result, _ = reg_tgt_net(tgt_mix_result)

            reg_src_loss = src_criterion(src_result, src_labels.view(4, 1).to(device))
            reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).to(device))

            loss = params.theta3 * mmd_loss + reg_src_loss + reg_tgt_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mmd_loss += params.theta3 * mmd_loss.item() * tgt_features.size(0)
            train_reg_loss += reg_src_loss.item() * src_result.size(0) + reg_tgt_loss.item() * tgt_result.size(0)
            index = len(src_loader) * epoch + idx

        train_loss = (train_reg_loss + train_mmd_loss) / len(tgt_loader.dataset)
        print(f'Epoch: {epoch}\tTraining Loss: {train_loss:.6f}')

def train_target_domain(src_train_loader, tgt_train_loader, common_src_net, common_tgt_net, reg_src_net, reg_tgt_net, reg_net, tgt_criterion, mmd_optimizer, epoch=1):
    # 关闭梯度计算
    with torch.no_grad():
        for param in common_src_net.parameters():
            param.requires_grad = False
        for param in reg_src_net.parameters():
            param.requires_grad = False

    train_reg_loss = 0

    for idx, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_imgs, src_labels = src_data
        src_samples = torch.autograd.Variable(src_imgs.view(4, -1).float().cuda())
        tgt_imgs, tgt_labels = tgt_data
        tgt_samples = torch.autograd.Variable(tgt_imgs.view(4, -1).float().cuda())

        src_features = common_src_net(src_samples)
        tgt_features = common_tgt_net(tgt_samples)
        features = torch.stack((src_features, tgt_features), dim=2)
        src_mix_result, tgt_mix_result = reg_net(features)

        tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
        reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).cuda())

        loss = reg_tgt_loss
        mmd_optimizer.zero_grad()
        loss.backward()
        mmd_optimizer.step()

        train_reg_loss += reg_tgt_loss.item() * tgt_result.size(0)
        index = len(src_train_loader) * epoch + idx

    train_loss = (train_reg_loss) / len(tgt_train_loader.dataset)
    print('Epoch: {} \t network pruning Training Loss: {:.6f}'.format(epoch, train_loss))

    # 恢复梯度计算
    for param in common_src_net.parameters():
        param.requires_grad = True
    for param in reg_src_net.parameters():
        param.requires_grad = True
    return train_loss

def main():
    full_src_dataset = Propress_Fun.dataset_load('NIRdataTensile.xlsx', 'NIRlabelTensile.xlsx')
    src_train_loader, src_test_loader = Propress_Fun.dataset_to_loader(full_src_dataset, 0.7)

    full_tgt_dataset = Propress_Fun.dataset_load('NIRtargetdataTensile.xlsx', 'NIRtragetlabelTensile.xlsx')
    tgt_train_loader, tgt_test_loader = Propress_Fun.dataset_to_loader(full_tgt_dataset, 0.7)

    common_src_net = Network.Extractor_src().to(device)
    common_tgt_net = Network.Extractor_tgt().to(device)
    reg_src_net = Network.Regression_net().to(device)
    reg_tgt_net = Network.Regression_net().to(device)
    reg_net = Network.Regression_mix_net().to(device)

    optimizer = optim.RMSprop([
        {'params': common_src_net.parameters()},
        {'params': common_tgt_net.parameters()},
        {'params': reg_src_net.parameters()},
        {'params': reg_tgt_net.parameters()},
        {'params': reg_net.parameters()}
    ], lr=params.lr, momentum=params.momentum)

    src_criterion = nn.MSELoss()
    tgt_criterion = nn.MSELoss()

    train_model(src_train_loader, tgt_train_loader, common_src_net, common_tgt_net, reg_src_net, reg_tgt_net,
                reg_net, optimizer, src_criterion, tgt_criterion, params.N_GEN_EPOCHS)
    train_target_domain(src_train_loader, tgt_train_loader, common_src_net, common_tgt_net, reg_src_net, reg_tgt_net,
                reg_net, tgt_criterion, optimizer, epoch=1)


if __name__ == '__main__':
    main()
