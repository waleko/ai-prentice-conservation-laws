from torch import nn
import torch


class ContrastiveWrapper(nn.Module):
    def __init__(self, model: nn.Module, n_eff: int, device):
        super(ContrastiveWrapper, self).__init__()
        self.model = model
        self.n_eff = n_eff
        self.device = device
        self.m = 0.01

    def forward(self, x):
        return self.calc(x, skip_loss=False)

    def calc(self, x, skip_loss: bool = False):
        labels = x[:, 0]
        data = x[:, 1:]

        embedding = self.model.encoder(data)
        decoded = self.model.decoder(embedding)

        if not skip_loss:
            conserved = embedding[:, self.n_eff:]
            n_conserved = conserved.size(1)

            # contrastive loss calculation
            labels = labels.view(-1, 1).expand(-1, n_conserved).long()
            _, labels_inverse = labels.unique(dim=0, return_inverse=True)

            lbl = labels_inverse.view(-1, 1)
            Y = (lbl != lbl.T)
            D2 = (conserved[:, None] - conserved[None, :]) ** 2
            D_mean = D2.mean(dim=-1)

            self.C_sim = torch.sum(Y * D_mean) / (Y.sum() + 1)
            self.C_diff = torch.sum((~Y) * torch.max(torch.zeros_like(D_mean, device=self.device), self.m - D_mean))\
                          / ((~Y).sum() + 1)

        return decoded

    def contrastive_loss(self, sim_lambda: float = 1, diff_lambda: float = 1):
        return sim_lambda * self.C_sim + diff_lambda * self.C_diff
