import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import datasets
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import params
from torch.autograd import Variable

import New_Loss
import ETD_model
import Network
import Propress_Fun

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('D:/onedrive_me/OneDrive/code/python_code/my_thesis_second_theme/runs')

def reset_grad(*nets):
    """Reset gradients of given networks."""
    for net in nets:
        net.zero_grad()

def load_datasets(src_file, src_label_file, tgt_file, tgt_label_file, split_ratio=0.7):
    """Load datasets from files and split into train and test sets."""
    full_src_dataset = Propress_Fun.dataset_load(src_file, src_label_file)
    src_train_loader, src_test_loader = Propress_Fun.dataset_to_loader(full_src_dataset, split_ratio)

    full_tgt_dataset = Propress_Fun.dataset_load(tgt_file, tgt_label_file)
    tgt_train_loader, tgt_test_loader = Propress_Fun.dataset_to_loader(full_tgt_dataset, split_ratio)

    return src_train_loader, src_test_loader, tgt_train_loader, tgt_test_loader

def optimize_mmd_model(common_src_net, common_tgt_net, reg_net_enhanced, reg_tgt_net_enhanced, reg_src_net_enhanced, reg_net_enhanced_att, optimizer_ot, src_train_loader, tgt_train_loader):
    """Optimize MMD model with given data loaders."""
    for epoch in range(params.N_ETD_CDAN_EPOCHS):
        for idx, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
            src_imgs, _ = src_data
            tgt_imgs, _ = tgt_data

            src_samples = Variable(src_imgs.view(4, -1).float().to(device))
            src_features = common_src_net(src_samples)
            tgt_samples = Variable(tgt_imgs.view(4, -1).float().to(device))
            tgt_features = common_tgt_net(tgt_samples)
            features = torch.stack((src_features, tgt_features), dim=2)
            src_mix_result, tgt_mix_result = reg_net(features)

            src_result, _ = reg_src_net(src_mix_result)
            tgt_result, _ = reg_tgt_net(tgt_mix_result)

            src_mix_result_enhanced, tgt_mix_result_enhanced = reg_net_enhanced(features.detach())
            features_att = torch.stack((src_mix_result_enhanced, tgt_mix_result_enhanced), dim=2)
            src_mix_result_att, tgt_mix_result_att = reg_net_enhanced_att(features_att)
            src_result_enhanced, _ = reg_src_net_enhanced(src_mix_result_att)
            tgt_result_enhanced, _ = reg_tgt_net_enhanced(tgt_mix_result_att)

            mmd_loss_enhanced = New_Loss.MMD(src_mix_result_enhanced, tgt_mix_result_enhanced, "rbf")
            sym_loss = src_criterion_enhanced(src_mix_result_enhanced, tgt_mix_result_enhanced)
            C = ETD_model.distance(src_mix_result_enhanced, tgt_mix_result_enhanced)
            C = Variable(C).to(device)
            att = ETD_model.Attention().to(device)
            weight = att(src_mix_result_enhanced, tgt_mix_result_enhanced)
            C = weight.mul(C)
            epsilon = 0.8
            f = torch.exp((src_result_enhanced.repeat(1, src_result_enhanced.size()[0]) +
                           tgt_result_enhanced.repeat(1, tgt_result_enhanced.size()[0]).t() - C) / epsilon)
            ot_Loss = -(torch.mean(src_result_enhanced) +
                        torch.mean(tgt_result_enhanced) - epsilon * torch.mean(f))
            optimizer_ot.zero_grad()
            loss_enhanced = mmd_loss_enhanced + sym_loss + ot_Loss
            loss_enhanced.backward()
            optimizer_ot.step()
            print('idx: {} \t loss_enhanced: {:.6f}'.format(idx, loss_enhanced))
    reset_grad(reg_net_enhanced, reg_tgt_net_enhanced, reg_src_net_enhanced, reg_net_enhanced_att)

def train_model_with_mmd(common_src_net, common_tgt_net, reg_src_net, reg_tgt_net, reg_net, reg_src_net_enhanced, reg_tgt_net_enhanced, reg_net_enhanced, reg_net_enhanced_att, mmd_optimizer, src_train_loader, tgt_train_loader):
    """Train model with MMD loss."""
    for epoch in range(params.N_ETD_EPOCHS):
        train_mmd_loss = 0
        train_reg_loss = 0
        for idx, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
            src_imgs, src_labels = src_data
            tgt_imgs, tgt_labels = tgt_data

            src_samples = Variable(src_imgs.view(4, -1).float().to(device))
            src_features = common_src_net(src_samples)
            tgt_samples = Variable(tgt_imgs.view(4, -1).float().to(device))
            tgt_features = common_tgt_net(tgt_samples)
            features = torch.stack((src_features, tgt_features), dim=2)
            src_mix_result, tgt_mix_result = reg_net(features)

            src_result, src_sig_result = reg_src_net(src_mix_result)
            tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
            reg_src_loss = src_criterion(src_result, src_labels.view(4, 1).to(device))
            reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).to(device))

            loss = reg_src_loss + reg_tgt_loss
            mmd_optimizer.zero_grad()
            loss.backward()
            mmd_optimizer.step()
            train_mmd_loss += reg_tgt_loss.item() * tgt_features.size(0)
            train_reg_loss += reg_src_loss.item() * src_result.size(0)
            index = len(src_train_loader) * epoch + idx
            writer.add_scalar("ETD_loss", loss, index)
        train_loss = (train_reg_loss + train_mmd_loss) / len(tgt_train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))



if __name__ == '__main__':
    src_train_loader, src_test_loader, tgt_train_loader, tgt_test_loader = load_datasets(
        'NIRdataTensile.xlsx', 'NIRlabelTensile.xlsx', 'NIRtargetdataTensile.xlsx', 'NIRtragetlabelTensile.xlsx')

    # Initialize networks
    common_src_net = Network.Extractor_src().to(device)
    common_tgt_net = Network.Extractor_tgt().to(device)
    reg_src_net = Network.Regression_net().to(device)
    reg_tgt_net = Network.Regression_net().to(device)
    reg_net = Network.Regression_mix_net().to(device)
    reg_src_net_enhanced = Network.Regression_net().to(device)
    reg_tgt_net_enhanced = Network.Regression_net().to(device)
    reg_net_enhanced = Network.Regression_mix_net().to(device)
    reg_net_enhanced_att = Network.Regression_mix_net().to(device)

    # Optimizers
    mmd_optimizer = optim.RMSprop([
        {'params': common_src_net.parameters()},
        {'params': common_tgt_net.parameters()},
        {'params': reg_src_net.parameters()},
        {'params': reg_tgt_net.parameters()},
        {'params': reg_net.parameters()}
    ], lr=params.lr, momentum=params.momentum)

    optimizer_ot = optim.Adam([
        {'params': reg_net_enhanced.parameters()},
        {'params': reg_tgt_net_enhanced.parameters()},
        {'params': reg_src_net_enhanced.parameters()},
        {'params': reg_net_enhanced_att.parameters()}
    ], lr=params.lr, weight_decay=0.01)

    # Loss functions
    src_criterion_enhanced = nn.MSELoss()
    src_criterion = nn.MSELoss()
    tgt_criterion = nn.MSELoss()

    optimize_mmd_model(common_src_net, common_tgt_net, reg_net_enhanced, reg_tgt_net_enhanced, reg_src_net_enhanced,
                       reg_net_enhanced_att, optimizer_ot, src_train_loader, tgt_train_loader)
    train_model_with_mmd(common_src_net, common_tgt_net, reg_src_net, reg_tgt_net, reg_net, reg_src_net_enhanced,
                         reg_tgt_net_enhanced, reg_net_enhanced, reg_net_enhanced_att, mmd_optimizer, src_train_loader,
                         tgt_train_loader)