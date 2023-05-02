from typing import Tuple

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


def get_centers(embedding: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the centers of the clusters for each label
    @param embedding: Contrastive loss embedding
    @param labels: Labels for each sample
    @return: Tuple of centers and indices of the centers for each sample
    """
    n_conserved = embedding.size(1)
    labels = labels.view(-1, 1).expand(-1, n_conserved).long()
    unique_labels, labels_inverse, labels_count = labels.unique(dim=0, return_inverse=True, return_counts=True)
    centers = torch.zeros_like(unique_labels,
                               dtype=torch.float,
                               device=embedding.device) \
        .scatter_add_(0, labels_inverse.view(-1, 1).repeat(1, n_conserved), embedding)
    centers /= labels_count.view(-1, 1)
    return centers, labels_inverse


def center_loss(embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the center loss. See https://ydwen.github.io/papers/WenECCV16.pdf
    @param embedding: Contrastive loss embedding
    @param labels: Labels for each sample
    @return: Center loss
    """
    centers, labels_inverse = get_centers(embedding, labels)
    return torch.sum((embedding - centers[labels_inverse, ]) ** 2) / embedding.size(0)


def diff_between_distances(embedding: torch.Tensor, labels: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    """
    Calculate the MSE between the distances of the centers and the desired Wasserstein distance
    @param embedding: Contrastive loss embedding
    @param labels: Labels for each sample
    @param wd: Wasserstein distance (as a matrix)
    @return: MSE
    """
    centers, labels_inverse = get_centers(embedding, labels)
    unique_labels = torch.unique(labels).long()
    dists = torch.cdist(centers, centers)
    return torch.sum((dists - wd[unique_labels, :][:, unique_labels]) ** 2) / (unique_labels.shape[0] ** 2)
