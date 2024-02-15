import torch
import torch.nn as nn


def distance(s_feature, t_feature):
    m = t_feature.size()[0]
    n = s_feature.size()[0]
    a = s_feature.pow(2).sum(1).unsqueeze(1).repeat(1, m)
    b = t_feature.pow(2).sum(1).repeat(n, 1)
    distance = (a + b - 2 * (s_feature.mm(t_feature.t()))).pow(0.5)
    return distance


class Attention(nn.Module):
    """ attention Layer"""

    def __init__(self, in_dim=20, out_dim=20):
        super(Attention, self).__init__()

        self.query_fc = nn.Linear(in_dim, out_dim)
        self.key_fc = nn.Linear(in_dim, out_dim)
        #        self.sigm = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B * F )
            returns :
                attention: B * B
        """
        #        batchsize, length = x.size()
        proj_query = self.query_fc(x)  # B * F
        proj_key = self.key_fc(y)  # B * F
        attention = torch.matmul(proj_query, proj_key.transpose(1, 0))  # B * B
        attention = torch.sigmoid(attention)
        return attention