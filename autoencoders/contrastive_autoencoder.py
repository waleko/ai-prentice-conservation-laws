from torch import nn
import torch


class ContrastiveWrapper(nn.Module):
    def __init__(self, model: nn.Module, n_eff: int):
        super(ContrastiveWrapper, self).__init__()
        self.__contrastive_loss = torch.tensor([0])
        self.model = model
        self.n_eff = n_eff

    def forward(self, x):
        labels = x[:, 0]
        data = x[:, 1:]

        embedding = self.model.encoder(data)

        conserved = embedding[:, self.n_eff:]
        n_conserved = conserved.size(1)

        # contrastive loss calculation
        labels = labels.view(-1, 1).expand(-1, n_conserved).long()
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        n_labels = unique_labels.size(0)
        labels_dims = labels.repeat(1, n_conserved)

        N = labels_count.view(-1, 1)

        S = torch.zeros((n_labels, n_conserved), dtype=conserved.dtype).scatter_add_(0, labels_dims, conserved)
        S2 = torch.zeros((n_labels, n_conserved), dtype=conserved.dtype).scatter_add_(0, labels_dims, conserved ** 2)
        M = S / N
        MSE = (S2 - 2 * M * S) / N + (M ** 2)
        res = MSE.mean()
        self.__contrastive_loss = res

        return self.model.decoder(embedding)

    def contrastive_loss(self):
        return self.__contrastive_loss
