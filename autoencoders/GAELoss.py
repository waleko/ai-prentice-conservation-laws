import torch
from sklearn.neighbors import KDTree
from torch import nn


class GAELoss(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def __call__(self, X, X_dot):
        X_np = X.detach().cpu().numpy()
        tree = KDTree(X_np, metric='minkowski', p=2)
        neighbors = tree.query(X_np, k=self.k, return_distance=False)
        X_dot_tr = X_dot[:, None, :].repeat((1, self.k, 1))
        X_tr = X[:, None, :].repeat((1, self.k, 1))
        X_neighbors = X[neighbors,]

        A = torch.square(torch.norm(X_neighbors - X_tr, dim=-1, p=2))
        assert A.shape == (X.shape[0], self.k)
        t = torch.mean(A) + 1e-9
        S = torch.exp(-A / t)
        return torch.mean(S * torch.square(torch.norm(X_neighbors - X_dot_tr, dim=-1, p=2)))
