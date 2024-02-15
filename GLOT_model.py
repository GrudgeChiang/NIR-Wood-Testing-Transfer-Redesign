import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np


def EntropicWassersteinLoss(
    z_s: torch.Tensor,
    ot_cost: torch.Tensor,
    kantorovich_pot: nn.Module,
    epsilon: float = 0.1,
):
    """
    Computes the stochastic entropic optimal transport cost moving points z_s to z_t.
    """
    source_size = ot_cost.size(0)
    target_size = ot_cost.size(1)
    kantorovich_val = kantorovich_pot(z_s)

    exp_term = (-ot_cost + kantorovich_val.repeat(1, target_size)) / epsilon
    kan_network_loss = kantorovich_val.mean()
    ot_loss = (
        torch.mean(
            -epsilon
            * (
                torch.log(torch.tensor(1.0 / source_size))
                + torch.logsumexp(exp_term, dim=0)
            )
        )
        + kan_network_loss
    )

    return ot_loss

class KanNet(nn.Module):
    """Kantorovich potential neural network"""

    def __init__(self, input_dim=20, bottleneck_dim=512):
        super(KanNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck_dim, 1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.fc(x)
        return x


def pairwise_forward_kl(y_s: torch.Tensor, y_t: torch.Tensor, from_logits=True):
    """
    Computes the pairwise forward KL divergence between two distributions.
    Args:
        y_s: source distribution
        y_t: target distribution
        from_logits: whether y_s and y_t are logits
    """
    if from_logits:
        y_s_prob = F.softmax(y_s, dim=1)
        y_t_prob = F.softmax(y_t, dim=1)
        return (y_t_prob * y_t_prob.log()).sum(dim=1) - torch.einsum(
            "ik, jk -> ij", y_s_prob.log(), y_t_prob
        )
    return (y_t * y_t.log()).sum(dim=1) - torch.einsum("ik, jk -> ij", y_s.log(), y_t)

def exp_rampup(rampup_length=30):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper
