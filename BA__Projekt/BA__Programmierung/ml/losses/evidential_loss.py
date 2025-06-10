# BA__Projekt/BA__Programmierung/ml/losses/evidential_loss.py
import torch
import torch.nn.functional as F


def evidential_loss(y, mu, v, alpha, beta, lambda_coef=1.0):
    two_blambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(torch.pi / v)
        - alpha * torch.log(two_blambda)
        + (alpha + 0.5) * torch.log((y - mu) ** 2 * v + two_blambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    error = torch.abs(y - mu)
    reg = error * (2 * v + alpha)
    return (nll + lambda_coef * reg).mean()
