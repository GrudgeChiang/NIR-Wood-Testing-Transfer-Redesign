import torch.nn as nn
import torch.nn.functional as F


class Extractor_src(nn.Module):
    def __init__(self):
        super(Extractor_src, self).__init__()
        self.conv2_drop = nn.Dropout(0.5)
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
        self.conv2_drop = nn.Dropout(0.5)
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
        self.conv2_drop1 = nn.Dropout(0.5)
        self.conv2_drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(20, 20)
        self.bn4 = nn.BatchNorm1d(20)
        self.fc4 = nn.Linear(20, 20)
        self.bn5 = nn.BatchNorm1d(20)
        self.fc5 = nn.Linear(20, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, input):
        x = self.fc3(input)
        x = self.bn4(x)
        x = self.fc4(F.relu((self.conv2_drop1(x))))
        x = self.bn5(x)
        result = self.fc5(F.relu((self.conv2_drop2(x))))
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


class Regression_mix_mdd_net(nn.Module):

    def __init__(self):
        super(Regression_mix_mdd_net, self).__init__()
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)

    def forward(self, input):
        logits_src = self.fc3(input[:, :, 0])
        logits_tgt = self.fc3(input[:, :, 1])
        logits_src_to_tgt = self.fc4(input[:, :, 0])
        return logits_src, logits_tgt, logits_src_to_tgt