import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torch.nn.functional as F
import params
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import Network
import Propress_Fun
import New_Loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('D:/onedrive_me/OneDrive/code/python_code/my_thesis_second_theme/runs')

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    full_src_dataset = Propress_Fun.dataset_load('NIRdataTensile.xlsx','NIRlabelTensile.xlsx')
    src_train_loader, src_test_loader = Propress_Fun.dataset_to_loader(full_src_dataset, 0.7)
    print(len(src_train_loader), len(src_test_loader))

    full_tgt_dataset = Propress_Fun.dataset_load('NIRtargetdataTensile.xlsx', 'NIRtragetlabelTensile.xlsx')
    tgt_train_loader, tgt_test_loader = Propress_Fun.dataset_to_loader(full_tgt_dataset, 0.7)
    print(len(tgt_train_loader), len(tgt_test_loader))

    common_src_net = Network.Extractor_src().to(device)
    common_tgt_net = Network.Extractor_tgt().to(device)
    reg_src_net    = Network.Regression_net().to(device)
    reg_tgt_net    = Network.Regression_net().to(device)
    reg_net        = Network.Regression_mix_net().to(device)

    common_src_net.train()
    common_tgt_net.train()
    reg_src_net.train()
    reg_tgt_net.train()
    reg_net.train()


    # Extractor.train()
    mmd_optimizer = optim.RMSprop([{'params':  common_src_net.parameters()},
                                   {'params':  common_tgt_net.parameters()},
                                   {'params':     reg_src_net.parameters()},
                                   {'params':     reg_tgt_net.parameters()},
                                   {'params':         reg_net.parameters()}],
                                   lr= params.lr, momentum= params.momentum)
    src_criterion = nn.MSELoss()
    tgt_criterion = nn.MSELoss()
    N_GEN_EPOCHS = 0

    for epoch in range(N_GEN_EPOCHS):
        train_mmd_loss = 0
        train_reg_loss = 0
        for idx in range(len(src_train_loader)):
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
            mmd_loss = New_Loss.MMD(src_mix_result, tgt_mix_result, "rbf")

            src_result, src_sig_result = reg_src_net(src_mix_result)
            src_sig_labels = src_labels
            tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
            tgt_sig_labels = tgt_labels

            reg_src_loss = src_criterion(src_result, src_labels.view(4, 1).cuda())
            reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).cuda())

            print('idx: {} \t mmd Loss: {}'.format(idx, params.theta3 * mmd_loss))
            print('idx: {} \t src Loss: {}'.format(idx, reg_src_loss))
            print('idx: {} \t tgt Loss: {}'.format(idx, reg_tgt_loss))

            loss = params.theta3 * mmd_loss + reg_src_loss + reg_tgt_loss
            mmd_optimizer.zero_grad()
            loss.backward()
            mmd_optimizer.step()

            train_mmd_loss += params.theta3 * mmd_loss.item() * tgt_features.size(0)
            train_reg_loss += reg_src_loss.item() * src_result.size(0) + reg_tgt_loss.item() * tgt_result.size(0)

            index = len(src_train_loader) * epoch + idx
            writer.add_scalar("mmd_loss", loss, index)

        train_loss = (train_reg_loss+train_mmd_loss) / len(tgt_train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    for param in common_src_net.parameters():
        param.requires_grad = False
    for param in reg_src_net.parameters():
        param.requires_grad = False

    # 目标域 单独训练
    src_dataiter = iter(src_train_loader)
    src_imgs, src_labels = next(src_dataiter)
    src_samples = Variable(src_imgs.view(4, -1).float().cuda())

    train_reg_loss = 0
    epoch = 1
    for idx in range(len(tgt_train_loader)):
        tgt_dataiter = iter(tgt_train_loader)
        tgt_imgs, tgt_labels = next(tgt_dataiter)
        tgt_samples = Variable(tgt_imgs.view(4, -1).float().cuda())

        src_features = common_src_net(src_samples)
        tgt_features = common_tgt_net(tgt_samples)
        features = torch.stack((src_features, tgt_features), dim=2)
        src_mix_result, tgt_mix_result = reg_net(features)

        src_result, src_sig_result = reg_src_net(src_mix_result)
        src_sig_labels = src_labels
        tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
        tgt_sig_labels = tgt_labels

        reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).cuda())

        print('idx: {} \t tgt Loss: {}'.format(idx, reg_tgt_loss))

        loss = reg_tgt_loss
        mmd_optimizer.zero_grad()
        loss.backward()
        mmd_optimizer.step()

        train_reg_loss += reg_tgt_loss.item() * tgt_result.size(0)

        index = len(src_train_loader) * epoch + idx
        writer.add_scalar("mmd_loss", loss, index)

    train_loss = (train_reg_loss) / len(tgt_train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    for param in common_src_net.parameters():
        param.requires_grad = True
    for param in reg_src_net.parameters():
        param.requires_grad = True

    # test
    common_src_net.eval()
    common_tgt_net.eval()
    reg_src_net.eval()
    reg_tgt_net.eval()
    reg_net.eval()

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
    mmd_loss = New_Loss.MMD(src_mix_result, tgt_mix_result, "rbf")

    src_result, src_sig_result = reg_src_net(src_mix_result)
    src_sig_labels = torch.sigmoid(src_labels)
    tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
    tgt_sig_labels = torch.sigmoid(tgt_labels)


    print('src output: ', src_result, '\t \n src data label: ', src_labels.view(4, 1).cuda())
    print('tgt output: ', tgt_result, '\t \n tgt data label: ', tgt_labels.view(4, 1).cuda())





    # for data, label in src_train_loader:
    #     data, label = data.cuda(), label.cuda()
