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

import CDAN_model
import Network
import Propress_Fun
import New_Loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    full_src_dataset = Propress_Fun.dataset_load('NIRdataTensile.xlsx','NIRlabelTensile.xlsx')
    src_train_loader, src_test_loader = Propress_Fun.dataset_to_loader(full_src_dataset, 0.7)
    print(len(src_train_loader), len(src_test_loader))

    full_tgt_dataset = Propress_Fun.dataset_load('NIRtargetdataTensile.xlsx', 'NIRtragetlabelTensile.xlsx')
    tgt_train_loader, tgt_test_loader = Propress_Fun.dataset_to_loader(full_tgt_dataset, 0.7)
    print(len(tgt_train_loader), len(tgt_test_loader))


    NIR_tgt_data_info = np.array(pd.read_excel(
        "NIRtragetlabelTensile.xlsx",
        engine='openpyxl'))
    label_tgt_info = NIR_tgt_data_info[:, 4]
    tgt_hist, tgt_edges = np.histogram(label_tgt_info, bins=40)
    tgt_hist = ((tgt_hist - np.min(tgt_hist))/(np.max(tgt_hist) - np.min(tgt_hist)))

    NIR_src_data_info = np.array(pd.read_excel(
        "NIRlabelTensile.xlsx",
        engine='openpyxl'))
    label_src_info = NIR_src_data_info[:, 4]
    src_hist, src_edges = np.histogram(label_src_info, bins=40)
    src_hist = ((src_hist - np.min(src_hist))/(np.max(src_hist) - np.min(src_hist)))



    common_src_net = Network.Extractor_src().to(device)
    common_tgt_net = Network.Extractor_tgt().to(device)
    reg_src_net    = Network.Regression_net().to(device)
    reg_tgt_net    = Network.Regression_net().to(device)
    reg_net        = Network.Regression_mix_net().to(device)
    # Extractor.train()
    mmd_optimizer = optim.RMSprop([{'params':  common_src_net.parameters()},
                                   {'params':  common_tgt_net.parameters()},
                                   {'params':     reg_src_net.parameters()},
                                   {'params':     reg_tgt_net.parameters()},
                                   {'params':         reg_net.parameters()}],
                                   lr= params.lr, momentum= params.momentum)


    ad_net = CDAN_model.AdversarialNetwork(20 * 20, 20)
    ad_net = ad_net.to(device)
    ad_net.train(True)
    N_CDAN_EPOCHS = 1
    N_GEN_CDAN_EPOCHS = 2

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
            src_mix_result_copy, tgt_mix_result_copy = reg_net(features.detach())

            # CDAN模块 此处进行了一个修改，将原始二值代价函数修改为回归代价函数
            feature_f = torch.cat((src_features, tgt_features), dim=0)
            feature_g = torch.cat((src_mix_result_copy, tgt_mix_result_copy), dim=0)
            feature_g = nn.Sigmoid()(feature_g)
            entropy_g = CDAN_model.Entropy(feature_g)

            entropy_target = np.zeros(4)
            for i, _ in enumerate(entropy_target):
                try:
                    index = np.argwhere(tgt_edges > tgt_labels[i].numpy())[0] - 1
                except IndexError:
                    index = np.shape(tgt_hist)[0] - 1
                    print(index)
                if index < 0:
                    entropy_target[i] = tgt_hist[index+1]
                else:
                    entropy_target[i] = tgt_hist[index]
            entropy_source = np.zeros(4)
            for i, _ in enumerate(entropy_source):
                try:
                    index = np.argwhere(src_edges > src_labels[i].numpy())[0] - 1
                except IndexError:
                    index = np.shape(src_hist)[0] - 1
                    print(index)
                if index < 0:
                    entropy_source[i] = src_hist[index+1]
                else:
                    entropy_source[i] = src_hist[index]
            entropy = np.concatenate((entropy_source, entropy_target))
            entropy = torch.tensor(entropy)
            entropy = Variable(entropy.float().cuda())

            epsilon = 1e-5
            entropy = epsilon * entropy_g + 0.5 * entropy


            transfer_loss = CDAN_model.CDAN([feature_f, feature_g], ad_net,
                                            entropy=entropy, coeff=CDAN_model.calc_coeff(1), random_layer=None)
            # BSP模块
            sigma = CDAN_model.BSP(src_mix_result_copy, tgt_mix_result_copy)
            sigma_loss = 0.0001 * sigma

            CDANtrans_loss = transfer_loss + sigma_loss
            print('Epoch: {} \t CDANtrans_loss: {:.6f}'.format(idx, CDANtrans_loss))
            mmd_optimizer.zero_grad()
            CDANtrans_loss.backward()
            mmd_optimizer.step()
    for epoch in range(N_GEN_CDAN_EPOCHS):
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
            src_sig_labels = torch.sigmoid(src_labels)
            tgt_result, tgt_sig_result = reg_tgt_net(tgt_mix_result)
            tgt_sig_labels = torch.sigmoid(tgt_labels)

            # K-L散度
            src_criterion = nn.MSELoss()
            reg_src_loss = src_criterion(src_result, src_labels.view(4, 1).cuda())
            tgt_criterion = nn.MSELoss()
            reg_tgt_loss = tgt_criterion(tgt_result, tgt_labels.view(4, 1).cuda())

            loss = reg_src_loss + reg_tgt_loss
            mmd_optimizer.zero_grad()
            loss.backward()
            mmd_optimizer.step()
            train_mmd_loss += reg_tgt_loss.item() * tgt_result.size(0)
            train_reg_loss += reg_src_loss.item() * src_result.size(0)
            index = len(src_train_loader) * epoch + idx
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

    # for data, label in src_train_loader:
    #     data, label = data.cuda(), label.cuda()
