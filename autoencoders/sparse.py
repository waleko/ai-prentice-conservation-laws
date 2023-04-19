import torch
from torch import nn


def sparse_loss(model: nn.Module, X: torch.Tensor, act_fn=nn.Tanh()) -> torch.Tensor:
    loss = 0
    values = X
    model_children = list(model.children())
    enc_children, dec_children = list(model_children[0].children()), list(model_children[1].children())
    for i in range(0, len(enc_children), 2):
        values = act_fn(enc_children[i](values))
        loss += torch.mean(torch.abs(values))
    for i in range(0, len(dec_children), 2):
        values = act_fn(dec_children[i](values))
        loss += torch.mean(torch.abs(values))
    return loss


def l1_loss(model: nn.Module) -> torch.Tensor:
    loss = 0
    model_params = list(model.parameters())
    for param in model_params:
        loss += torch.mean(torch.abs(param))
    return loss


def sparse_and_l1(sparse_reg: float = 0, l1_lambda: float = 0):
    return lambda X, __, model: l1_lambda * l1_loss(model) + sparse_reg * sparse_loss(model, X)


def bottleneck_sparse_loss(model: nn.Module, X: torch.Tensor, threshold: float) -> int:
    values = model.encoder(X)
    loss = torch.mean(torch.abs(values), dim=0)
    print(loss)
    return torch.sum(loss > threshold).item()
