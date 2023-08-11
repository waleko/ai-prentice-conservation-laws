import torch
import torch.nn as nn
from torch.autograd.functional import jacobian


def default_loss(model, input, target):
    output = model(input)
    loss = nn.MSELoss()(output, target)
    return loss


def get_M(fisher_ae, mu, slice_idx=None):
    # TODO
    # is there a more sufficient way to compute the Jacobians?
    J = torch.stack([jacobian(fisher_ae.decoder, mu_i) for mu_i in mu])
    if slice_idx is not None:
        J = J[..., :slice_idx]
    M = torch.matmul(torch.swapaxes(J, 1, 2), J) * torch.exp(-fisher_ae.xi) + torch.eye(J.shape[-1])[None, ...]
    return M


def sample_gaussian_variables(mu, sigma):
    n = torch.randn_like(mu)
    L = torch.linalg.cholesky(sigma)
    z = torch.bmm(L, n[..., None]).reshape(mu.shape) + mu
    return z


def probabilistic_loss(fisher_ae, input, _):
    latent_mean = fisher_ae.encoder(input)
    M = get_M(fisher_ae, latent_mean)
    latent = sample_gaussian_variables(latent_mean, torch.inverse(M))
    output = fisher_ae.decoder(latent)

    s1 = (latent ** 2).sum()
    s2 = fisher_ae.xi ** 2 + input.shape[0] * input.shape[1] * fisher_ae.xi
    s3 = ((input - output) ** 2).sum() * torch.exp(-fisher_ae.xi)
    loss = (s1 + s2 + s3) / latent.shape[0] / latent.shape[1]
    
    return loss.reshape(())


def probabilistic_loss_joint_ae(joint_ae, input, _):
    latent_mean = joint_ae.projection(joint_ae.encoder(input))
    conserved = joint_ae.conserved(input)
    M = get_M(joint_ae, torch.cat((latent_mean, conserved), axis=-1), slice_idx=latent_mean.shape[-1])
    latent = sample_gaussian_variables(latent_mean, torch.inverse(M))
    output = joint_ae.decoder(torch.cat((latent, conserved), axis=-1))

    s1 = (latent ** 2).sum()
    s2 = joint_ae.xi ** 2 + input.shape[0] * input.shape[1] * joint_ae.xi
    s3 = ((input - output) ** 2).sum() / torch.exp(joint_ae.xi)
    loss = (s1 + s2 + s3) / latent.shape[0] / latent.shape[1]
    
    return loss.reshape(())


def equality_forcing_loss(model, input, target, lam=1):
    output = model(input)
    mse = nn.MSELoss()
    loss = mse(output, target)
    idx = {}
    elements, indices = torch.unique(target, return_inverse=True, dim=0)
    for i in range(len(elements)):
        mask = (indices == i)
        predictions = output[mask]
        loss += lam * torch.sum((predictions - torch.mean(predictions, axis=0)) ** 2)
    return loss
