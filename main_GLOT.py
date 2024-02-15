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

import GLOT_model
import New_Loss
import Network
import Propress_Fun


def reset_grad():
    reg_net_enhenced.zero_grad()
    reg_tgt_net_enhenced.zero_grad()
    reg_src_net_enhenced.zero_grad()
    common_src_net.zero_grad()
    common_tgt_net.zero_grad()
    reg_src_net.zero_grad()
    reg_tgt_net.zero_grad()
    reg_net.zero_grad()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma3 = 0.02
gamma4 = 0.03


if __name__ == '__main__':
    full_src_dataset = Propress_Fun.dataset_load('NIRdataTensile.xlsx','NIRlabelTensile.xlsx')
    train_src_dataset, test_src_dataset  = Propress_Fun.dataset_random_split(full_src_dataset, train_size=0.7)
    train_src_dataset, unlab_src_dataset = Propress_Fun.dataset_random_split(train_src_dataset, train_size=0.7)
    src_train_loader = DataLoader(train_src_dataset, batch_size=4, num_workers=4,
                              shuffle=True, drop_last=True)
    src_test_loader  = DataLoader(test_src_dataset, batch_size=4, num_workers=4,
                              shuffle=True, drop_last=True)
    src_unlab_loader = DataLoader(unlab_src_dataset, batch_size=4, num_workers=4,
                              shuffle=True, drop_last=True)

    full_tgt_dataset = Propress_Fun.dataset_load('NIRtargetdataTensile.xlsx', 'NIRtragetlabelTensile.xlsx')
    train_tgt_dataset, test_tgt_dataset  = Propress_Fun.dataset_random_split(full_tgt_dataset, train_size=0.7)
    train_tgt_dataset, unlab_tgt_dataset = Propress_Fun.dataset_random_split(train_tgt_dataset, train_size=0.7)
    tgt_train_loader = DataLoader(train_tgt_dataset, batch_size=4, num_workers=4,
                              shuffle=True, drop_last=True)
    tgt_test_loader  = DataLoader(test_tgt_dataset, batch_size=4, num_workers=4,
                              shuffle=True, drop_last=True)
    tgt_unlab_loader = DataLoader(unlab_tgt_dataset, batch_size=4, num_workers=4,
                              shuffle=True, drop_last=True)


    common_src_net = Network.Extractor_src().to(device)
    common_tgt_net = Network.Extractor_tgt().to(device)
    reg_src_net    = Network.Regression_net().to(device)
    reg_tgt_net    = Network.Regression_net().to(device)
    reg_net        = Network.Regression_mix_net().to(device)
    reg_src_net_enhenced    = Network.Regression_net().to(device)
    reg_tgt_net_enhenced    = Network.Regression_net().to(device)
    reg_net_enhenced        = Network.Regression_mix_net().to(device)


    # Extractor.train()
    mmd_optimizer = optim.RMSprop([{'params':  common_src_net.parameters()},
                                   {'params':  common_tgt_net.parameters()},
                                   {'params':     reg_src_net.parameters()},
                                   {'params':     reg_tgt_net.parameters()},
                                   {'params':         reg_net.parameters()}],
                                   lr= params.lr, momentum= params.momentum)

    optimizer_ot = optim.Adam([{'params':           reg_net_enhenced.parameters()},
                               {'params':       reg_tgt_net_enhenced.parameters()},
                               {'params':       reg_src_net_enhenced.parameters()}],
                                lr= params.lr, weight_decay=0.01)

    N_CDAN_EPOCHS = 1
    for epoch in range(N_CDAN_EPOCHS):
        for idx in range(len(src_unlab_loader)):

            src_unlab_dataiter = iter(src_unlab_loader)
            src_unlab_imgs, src_unlab_labels = next(src_unlab_dataiter)
            tgt_unlab_dataiter = iter(tgt_unlab_loader)
            tgt_unlab_imgs, tgt_unlab_labels = next(tgt_unlab_dataiter)

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


            src_unlab_samples = Variable(src_unlab_imgs.view(4, -1).float().cuda())
            src_unlab_features = common_src_net(src_unlab_samples)
            tgt_unlab_samples = Variable(tgt_unlab_imgs.view(4, -1).float().cuda())
            tgt_unlab_features = common_tgt_net(tgt_unlab_samples)
            unlab_features = torch.stack((src_unlab_features, tgt_unlab_features), dim=2)

            src_mix_result_enhenced, tgt_mix_result_enhenced = reg_net(unlab_features)
            src_result_enhenced, src_sig_result_enhenced = reg_src_net(src_mix_result_enhenced)
            tgt_result_enhenced, tgt_sig_result_enhenced = reg_tgt_net(tgt_mix_result_enhenced)


            mmd_loss_enhenced = New_Loss.MMD(src_mix_result_enhenced, tgt_mix_result_enhenced, "rbf")
            sym_loss = New_Loss.symmetric_mse_loss(src_mix_result_enhenced, tgt_mix_result_enhenced)
            # loss_robust = GLOT_model.exp_rampup()(3) * 0.3
            KanNet_model = GLOT_model.KanNet().cuda(device)

            C_src = 0.5 * GLOT_model.pairwise_forward_kl(
                src_sig_result, src_sig_result_enhenced) + torch.cdist(src_mix_result, src_mix_result_enhenced, p=2)
            loss_ws_src = GLOT_model.EntropicWassersteinLoss(src_mix_result, C_src, KanNet_model, 0.1)

            C_tgt = 0.5 * GLOT_model.pairwise_forward_kl(
                tgt_sig_result, tgt_sig_result_enhenced) + torch.cdist(tgt_mix_result, tgt_mix_result_enhenced, p=2)
            loss_ws_tgt = GLOT_model.EntropicWassersteinLoss(tgt_mix_result, C_tgt, KanNet_model, 0.1)

            loss_ws = loss_ws_src + loss_ws_tgt
            loss_ws *= GLOT_model.exp_rampup()(epoch)

            optimizer_ot.zero_grad()
            loss_enhenced = mmd_loss_enhenced + loss_ws
            loss_enhenced.backward()
            optimizer_ot.step()

            print("loss_enhenced: ", loss_enhenced)

    reset_grad()



    kannet_optimizer = torch.optim.SGD(KanNet_model.parameters(), lr=0.1)
    KanNet_model.zero_grad()

    for epoch in range(N_CDAN_EPOCHS):
        train_mmd_loss = 0
        train_reg_loss = 0
        for idx in range(len(src_unlab_loader)):
            src_unlab_dataiter = iter(src_unlab_loader)
            src_unlab_imgs, src_unlab_labels = next(src_unlab_dataiter)
            tgt_unlab_dataiter = iter(tgt_unlab_loader)
            tgt_unlab_imgs, tgt_unlab_labels = next(tgt_unlab_dataiter)

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

            src_unlab_samples = Variable(src_unlab_imgs.view(4, -1).float().cuda())
            src_unlab_features = common_src_net(src_unlab_samples)
            tgt_unlab_samples = Variable(tgt_unlab_imgs.view(4, -1).float().cuda())
            tgt_unlab_features = common_tgt_net(tgt_unlab_samples)
            unlab_features = torch.stack((src_unlab_features, tgt_unlab_features), dim=2)

            src_mix_result_enhenced, tgt_mix_result_enhenced = reg_net(unlab_features)
            src_result_enhenced, src_sig_result_enhenced = reg_src_net(src_mix_result_enhenced)
            tgt_result_enhenced, tgt_sig_result_enhenced = reg_tgt_net(tgt_mix_result_enhenced)


            src_z = src_mix_result.detach().clone()
            src_unlab_z = src_mix_result_enhenced.detach().clone()
            src_outputs = src_sig_result.detach().clone()
            src_unlab_outputs = src_sig_result_enhenced.detach().clone()

            tgt_z = tgt_mix_result.detach().clone()
            tgt_unlab_z = tgt_mix_result_enhenced.detach().clone()
            tgt_outputs = tgt_sig_result.detach().clone()
            tgt_unlab_outputs = tgt_sig_result_enhenced.detach().clone()

            src_C = torch.cdist(src_z, src_unlab_z, p=2) + 0.5 * GLOT_model.pairwise_forward_kl(
                src_outputs, src_unlab_outputs)
            src_neg_loss_ws = -GLOT_model.EntropicWassersteinLoss(src_z, src_C, KanNet_model, 0.1)

            tgt_C = torch.cdist(tgt_z, tgt_unlab_z, p=2) + 0.5 * GLOT_model.pairwise_forward_kl(
                tgt_outputs, tgt_unlab_outputs)
            tgt_neg_loss_ws = -GLOT_model.EntropicWassersteinLoss(tgt_z, tgt_C, KanNet_model, 0.1)

            neg_loss_ws = - gamma4 * src_neg_loss_ws - gamma3 * tgt_neg_loss_ws
            neg_loss_ws.backward()
            kannet_optimizer.step()
            KanNet_model.zero_grad()


            train_mmd_loss += - gamma3 * tgt_neg_loss_ws.item() * tgt_features.size(0)
            train_reg_loss += - gamma4 * src_neg_loss_ws.item() * src_result.size(0)
        print("train_mmd_loss: ", train_mmd_loss)
        print("train_reg_loss: ", train_reg_loss)
        train_loss = (train_reg_loss + train_mmd_loss) / len(tgt_train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

