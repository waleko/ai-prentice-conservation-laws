import torch


def lifted_structured_loss(sim_lambda: float, diff_lambda: float, m: float):
    def helper(conserved: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_conserved = conserved.size(1)
        # contrastive loss calculation
        labels = labels.view(-1, 1).expand(-1, n_conserved).long()
        _, labels_inverse = labels.unique(dim=0, return_inverse=True)

        lbl = labels_inverse.view(-1, 1)
        Y = (lbl != lbl.T)
        D2 = (conserved[:, None] - conserved[None, :]) ** 2
        D_mean = D2.mean(dim=-1)

        C_sim = torch.sum(Y * D_mean) / (Y.sum() + 1)
        C_diff = torch.sum((~Y) * torch.max(torch.zeros_like(D_mean, device=labels.device), m - D_mean)) \
                 / ((~Y).sum() + 1)
        return sim_lambda * C_sim + diff_lambda * C_diff

    return helper
