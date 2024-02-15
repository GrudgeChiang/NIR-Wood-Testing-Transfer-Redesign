import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    return torch.mean(XX + YY - 2. * XY)

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax)


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    # num_classes = input1.size()[1]
    # return torch.sum((input1 - input2)**2) / num_classes
    batch_size = input1.size()[0]
    return torch.sum((input1 - input2) ** 2) / batch_size


def EMD_loss(input1, input2):
    """Simple Implementation Version of Wasserstain distance
    Used like in the WGAN

    Note:
    - Return the distance between two domains' center and take the Positive part
    - Sends gradients to both input1 and input2.
    """
    dist = torch.mean(input1) - torch.mean(input2)
    return dist.abs()


def entropy_loss(input_f):
    """Simple Implementation Version of target domain entropy

    Note:
    - Return the entropy of one domain, expect to reduce the variance
    - Sends gradients to both input1 and input2.
    """
    ans = torch.mul(input_f, torch.log(input_f + 0.001))
    mid = torch.sum(ans, dim=1)
    return -torch.mean(mid)


def coral_loss(input1, input2, gamma=1e-3):
    ''' Implementation of coral covariances (which is regularized)

    Note:
    - Return the covariances of all features. That's [f*f]
    - Send gradients to both input1 and input2
    '''
    # First: subtract the mean from the data matrix
    batch_size = float(input1.shape[0])
    h_src = input1 - torch.mean(input1, dim=0)
    h_trg = input2 - torch.mean(input2, dim=0)

    cov_src = (1. / (batch_size - 1)) * torch.mm(torch.transpose(h_src, 0, 1), h_src)
    cov_trg = (1. / (batch_size - 1)) * torch.mm(torch.transpose(h_trg, 0, 1), h_trg)
    # Returns the Frobenius norm
    # The mean account for the factor 1/d^2
    return torch.mean(torch.pow(cov_src - cov_trg, 2))


def log_coral_loss(input1, input2, gamma=1e-3):
    ''' Implementation of eig version coral covariances (which is regularized)

    Note:
    - Return the covariances of all features. That's [f*f]
    - Send gradients to both input1 and input2
    '''
    # First: subtract the mean from the data matrix
    batch_size = float(input1.shape[0])
    h_src = input1 - torch.mean(input1, dim=0)
    h_trg = input2 - torch.mean(input2, dim=0)

    cov_src = (1. / (batch_size - 1)) * torch.mm(torch.transpose(h_src, 0, 1), h_src)
    cov_trg = (1. / (batch_size - 1)) * torch.mm(torch.transpose(h_trg, 0, 1), h_trg)

    # eigen decomposition
    e_src, v_src = torch.symeig(cov_src, eigenvectors=True)
    e_trg, v_trg = torch.symeig(cov_trg, eigenvectors=True)

    # Returns the Frobenius norm
    log_cov_src = torch.mm(e_src.view(1, -1), torch.mm(torch.diag(torch.log(e_src + 0.0001)), torch.transpose(v_src, 0, 1)))
    log_cov_trg = torch.mm(e_trg.view(1, -1), torch.mm(torch.diag(torch.log(e_trg + 0.0001)), torch.transpose(v_trg, 0, 1)))
    return torch.mean(torch.pow(log_cov_src - log_cov_trg, 2))

def l2dist(source, target):
    """Computes pairwise Euclidean distances in torch."""
    def flatten_batch(x):
      dim = torch.prod(torch.tensor(x.shape[1:]))
      return x.reshape([-1, dim])
    def scale_batch(x):
      dim = torch.prod(torch.tensor(x.shape[1:]))
      return x/dim.double().sqrt()
    def prepare_batch(x):
      return scale_batch(flatten_batch(x))

    target_flat = prepare_batch(target)  # shape: [bs, nt]
    target_sqnorms = torch.sum(target_flat.pow(2),1).unsqueeze(1)
    target_sqnorms_t = target_sqnorms.transpose(0,1)  # shape: [bs, nt]

    source_flat = prepare_batch(source)  # shape: [bs, ns]
    source_sqnorms = torch.sum(source_flat.pow(2),1).unsqueeze(1)

    dotprod = source_flat.mm(target_flat.transpose(0,1))
    sqdist = source_sqnorms - 2*dotprod + target_sqnorms_t #broadcast
    dist = F.relu(sqdist).sqrt()  # potential tiny negatives are suppressed
    return dist  # shape: [ns, nt]


