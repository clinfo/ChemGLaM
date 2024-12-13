import numpy as np
import torch
import torch.nn.functional as F


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def edl_loss(func, y, alpha, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.tensor(1.0, dtype=torch.float32)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, num_classes, annealing_step, device=None
):
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, num_classes, annealing_step, device
        )
    )
    return loss

def edl_square_loss(
    output, target, num_classes, annealing_step, device=None
):
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    
    nll = loglikelihood_loss(target, alpha, device)
    
    annealing_coef = torch.tensor(1.0, dtype=torch.float32)

    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
        
    loss = torch.mean(nll + kl_div)
    
    return loss

class EvidentialLoss(torch.nn.Module):
    def __init__(self, num_classes, annealing_step, device=None):
        super(EvidentialLoss, self).__init__()
        
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device
    
    def forward(self, y_pred, y_true):
        return edl_digamma_loss(y_pred, y_true, self.num_classes, self.annealing_step, self.device)

class EvidentialSquareLoss(torch.nn.Module):
    def __init__(self, num_classes, annealing_step, device=None):
        super(EvidentialSquareLoss, self).__init__()
        
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device
    
    def forward(self, y_pred, y_true):
        return edl_square_loss(y_pred, y_true, self.num_classes, self.annealing_step, self.device)

def nig_nll(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (v + 1)

    nll = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return torch.mean(nll) if reduce else nll

def kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = (
        0.5 * (a1 - 1) / b1 * (v2 * torch.square(mu2 - mu1))
        + 0.5 * v2 / v1
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1))
        - 0.5
        + a2 * torch.log(b1 / b2)
        - (torch.lgamma(a1) - torch.lgamma(a2))
        + (a1 - a2) * torch.digamma(a1)
        - (b1 - b2) * a1 / b1
    )
    return KL

def nig_reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    error = (y - gamma).abs()

    if kl:
        kl = kl_nig(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + alpha
        reg = error * evi

    return torch.mean(reg) if reduce else reg

def edl_regression(output, y_true, coeff=1.0, device=None):
    output = output.view(-1, 4)
    gamma = output[:, 0].view(-1)
    evidential_output = softplus_evidence(output) # evidential_output > 0 
    
    targets = y_true.view(-1)
    v = evidential_output[:, 1].view(-1) + 1e-3 # v > 0
    alpha = evidential_output[:, 2].view(-1) + 1 + 1e-3 # alpha > 1
    beta = evidential_output[:, 3].view(-1) + 1e-3 # beta > 0 
    
    aleatoric = beta / (alpha - 1)
    epistemic = beta / (v*(alpha-1))
    
    print("aleatoric:", aleatoric)
    print("epistemic:", epistemic)
    
    loss_nll = nig_nll(targets, gamma, v, alpha, beta)
    loss_reg = nig_reg(targets, gamma, v, alpha, beta)
    
    return loss_nll + coeff * loss_reg

class EvidentialRegressionLoss(torch.nn.Module):
    def __init__(self, device=None, coeff=1.0):
        super(EvidentialRegressionLoss, self).__init__()

        self.device = device
        self.coeff = 1.0
    
    def forward(self, y_pred, y_true):
        return edl_regression(y_pred, y_true, coeff=self.coeff, device=self.device)