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

def reset_grad():
    reg_net_enhanced.zero_grad()
    reg_tgt_net_enhanced.zero_grad()
    reg_src_net_enhanced.zero_grad()
    common_src_net.zero_grad()
    common_tgt_net.zero_grad()
    reg_src_net.zero_grad()
    reg_tgt_net.zero_grad()
    reg_net.zero_grad()



if __name__ == '__main__':
    full_src_dataset = Propress_Fun.dataset_load('NIRdataTensile.xlsx','NIRlabelTensile.xlsx')
    src_train_loader, src_test_loader = Propress_Fun.dataset_to_loader(full_src_dataset, 0.7)

    full_tgt_dataset = Propress_Fun.dataset_load('NIRtargetdataTensile.xlsx', 'NIRtragetlabelTensile.xlsx')
    tgt_train_loader, tgt_test_loader = Propress_Fun.dataset_to_loader(full_tgt_dataset, 0.7)

    common_src_net = Network.Extractor_src().to(device)
    common_tgt_net = Network.Extractor_tgt().to(device)
    reg_src_net    = Network.Regression_net().to(device)
    reg_tgt_net    = Network.Regression_net().to(device)
    reg_net        = Network.Regression_mix_net().to(device)
    reg_src_net_enhanced    = Network.Regression_net().to(device)
    reg_tgt_net_enhanced    = Network.Regression_net().to(device)
    reg_net_enhanced        = Network.Regression_mix_net().to(device)
    reg_net_enhanced_att    = Network.Regression_mix_net().to(device)


    # Extractor.train()
    mmd_optimizer = optim.RMSprop([{'params':  common_src_net.parameters()},
                                   {'params':  common_tgt_net.parameters()},
                                   {'params':     reg_src_net.parameters()},
                                   {'params':     reg_tgt_net.parameters()},
                                   {'params':         reg_net.parameters()}],
                                   lr= params.lr, momentum= params.momentum)

    optimizer_ot = optim.Adam([{'params':            reg_net_enhanced.parameters()},
                                {'params':       reg_tgt_net_enhanced.parameters()},
                                {'params':       reg_src_net_enhanced.parameters()},
                                {'params':       reg_net_enhanced_att.parameters()}],
                                lr= params.lr, weight_decay=0.01)

    src_criterion_enhanced = nn.MSELoss()
    src_criterion = nn.MSELoss()
    tgt_criterion = nn.MSELoss()
    N_GEN_EPOCHS = 10
    N_CDAN_EPOCHS = 1

    for epoch in range(N_CDAN_EPOCHS):
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

            src_result, src_sig_result = reg_src_net(src_mix_result)
            tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)

            # 实现一个子网络梯度的 backward -> step 需要 features.detach() 复制一个tensor并保存梯度
            src_mix_result_enhanced, tgt_mix_result_enhanced = reg_net_enhanced(features.detach())
            features_att = torch.stack((src_mix_result_enhanced, tgt_mix_result_enhanced), dim=2)
            src_mix_result_att, tgt_mix_result_att = reg_net_enhanced_att(features_att)
            src_result_enhanced, src_sig_result_enhanced = reg_src_net_enhanced(src_mix_result_att)
            tgt_result_enhanced, tgt_sig_result_enhanced = reg_tgt_net_enhanced(tgt_mix_result_att)


            mmd_loss_enhanced = New_Loss.MMD(src_mix_result_enhanced, tgt_mix_result_enhanced, "rbf")
            sym_loss = src_criterion_enhanced(src_mix_result_enhanced, tgt_mix_result_enhanced)
            C = ETD_model.distance(src_mix_result_enhanced, tgt_mix_result_enhanced)
            C = Variable(C).cuda()
            att = ETD_model.Attention().cuda(device)
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
    reset_grad()

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

            src_result, src_sig_result = reg_src_net(src_mix_result)
            tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
            reg_src_loss = src_criterion(src_result, src_labels.view(4, 1).cuda())
            reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).cuda())

            loss = reg_src_loss + reg_tgt_loss
            mmd_optimizer.zero_grad()
            loss.backward()
            mmd_optimizer.step()
            train_mmd_loss += reg_tgt_loss.item() * tgt_features.size(0)
            train_reg_loss += reg_src_loss.item() * src_result.size(0)
            index = len(src_train_loader) * epoch + idx
            writer.add_scalar("ETD_loss",  loss, index)
        train_loss = (train_reg_loss+train_mmd_loss) / len(tgt_train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # test
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

    src_result, src_sig_result = reg_src_net(src_mix_result)
    tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)

    print('src output: ', src_result, '\t \n src data label: ', src_labels.view(4, 1).cuda())
    print('tgt output: ', tgt_result, '\t \n tgt data label: ', tgt_labels.view(4, 1).cuda())

