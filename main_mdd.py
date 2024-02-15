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


import Network
import Propress_Fun
import New_Loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    full_src_dataset = Propress_Fun.dataset_load('NIRdataTensile.xlsx','NIRlabelTensile.xlsx')
    src_train_loader, src_test_loader = Propress_Fun.dataset_to_loader(full_src_dataset, 0.7)


    full_tgt_dataset = Propress_Fun.dataset_load('NIRtargetdataTensile.xlsx', 'NIRtragetlabelTensile.xlsx')
    tgt_train_loader, tgt_test_loader = Propress_Fun.dataset_to_loader(full_tgt_dataset, 0.7)

    common_src_net = Network.Extractor_src().to(device)
    common_tgt_net = Network.Extractor_tgt().to(device)
    reg_src_net    = Network.Regression_net().to(device)
    reg_tgt_net    = Network.Regression_net().to(device)
    reg_net = Network.Regression_mix_mdd_net().to(device)
    reg_src_to_tgt_net = Network.Regression_net().to(device)
    # Extractor.train()
    # gmmd_optimizer = optim.RMSprop(common_net1.parameters(), lr=0.004)
    mmd_optimizer = optim.RMSprop([{'params': common_src_net.parameters()},
                                    {'params': common_tgt_net.parameters()},
                                   {'params':     reg_src_net.parameters()},
                                   {'params':     reg_tgt_net.parameters()},
                                   {'params':         reg_net.parameters()},
                                   {'params': reg_src_to_tgt_net.parameters()}],
                                   lr= params.lr, momentum= params.momentum)

    src_criterion = nn.MSELoss()
    tgt_criterion = nn.MSELoss()

    N_GEN_EPOCHS = 10
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
            src_mix_result, tgt_mix_result, src_to_tgt_mix_result = reg_net(features)
            src_result, src_sig_result = reg_src_net(src_mix_result)
            tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
            src_to_tgt_result, src_to_tgt_sig_result = reg_src_to_tgt_net(src_to_tgt_mix_result)


            reg_src_loss = src_criterion(src_result, src_labels.view(4, 1).cuda())
            reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).cuda())
            src_to_tgt_loss = src_criterion(src_to_tgt_result, tgt_labels.view(4, 1).cuda())

            # mdd模块核心：source —> target
            loss = reg_src_loss + reg_tgt_loss + src_to_tgt_loss
            mmd_optimizer.zero_grad()
            loss.backward()
            mmd_optimizer.step()
            train_reg_loss += reg_src_loss.item() * src_result.size(0)
            print('idx: {} \t src Loss: {}'.format(idx, reg_src_loss))
            print('idx: {} \t tgt Loss: {}'.format(idx, reg_tgt_loss))
            index = len(src_train_loader) * epoch + idx
        train_loss = (train_reg_loss) / len(tgt_train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # text
    src_dataiter = iter(src_train_loader)
    src_imgs, src_labels = next(src_dataiter)
    tgt_dataiter = iter(tgt_train_loader)
    tgt_imgs, tgt_labels = next(tgt_dataiter)

    src_samples = Variable(src_imgs.view(4, -1).float().cuda())
    src_features = common_src_net(src_samples)
    tgt_samples = Variable(tgt_imgs.view(4, -1).float().cuda())
    tgt_features = common_tgt_net(tgt_samples)

    features = torch.stack((src_features, tgt_features), dim=2)
    src_mix_result, tgt_mix_result, src_to_tgt_mix_result = reg_net(features)
    src_result, src_sig_result = reg_src_net(src_mix_result)
    tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)

    print('src output: ', src_result, '\t \n src data label: ', src_labels.view(4, 1).cuda())
    print('tgt output: ', tgt_result, '\t \n tgt data label: ', tgt_labels.view(4, 1).cuda())
