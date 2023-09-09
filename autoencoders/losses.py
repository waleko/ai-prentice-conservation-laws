import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.autograd.functional import jacobian
from torch.autograd import grad
from typing import Iterable


class MSELoss(_Loss):
    def __init__(self, lam: float = 1.0):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lam = lam
        
    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor):
        loss = self.mse(outputs, target)
        return self.lam * loss


class RegularizationLoss(_Loss):
    def __init__(self, p: Iterable[float] = [1.0, 2.0], lam: Iterable[float] = [1e-5, 1e-5]):
        super(RegularizationLoss, self).__init__()

        self.p = p
        self.lam = lam

    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor):
        params = torch.abs(torch.concat([p.reshape(-1) for p in model.parameters()]))
        loss = sum([lam * (params ** p).mean() for p, lam in zip(self.p, self.lam)])
        return loss


class CombinedLoss(_Loss):
    def __init__(self, losses: Iterable[_Loss] = [MSELoss(), RegularizationLoss()]):
        super(CombinedLoss, self).__init__()

        self.losses = losses

    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor):
        total_loss = sum([loss(model, inputs, outputs, target) for loss in self.losses])
        return total_loss


# def default_loss(model, inputs, target):
#     outputs = model(inputs)
#     loss = nn.MSELoss()(outputs, target)
#     return loss


class ProbabilisticLoss(_Loss):
    def __init__(self, lam: float = 1.0):
        super(ProbabilisticLoss, self).__init__()

        self.lam = lam
    
    def get_M(fisher_ae, mu, slice_idx=None):
        J = compute_jacobians(mu, fisher_ae.decoder(mu))
        if slice_idx is not None:
            J = J[..., :slice_idx]
        M = torch.matmul(torch.swapaxes(J, 1, 2), J) * torch.exp(-fisher_ae.xi) + torch.eye(J.shape[-1])[None, ...]
        return M

    def sample_gaussian_variables(mu, sigma):
        n = torch.randn_like(mu)
        L = torch.linalg.cholesky(sigma)
        z = torch.bmm(L, n[..., None]).reshape(mu.shape) + mu
        return z

    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor):
        latent_mean = model.encoder(inputs)
        M = self.get_M(model, latent_mean)
        latent = self.sample_gaussian_variables(latent_mean, torch.inverse(M))
        outputs = model.decoder(latent)
    
        s1 = (latent ** 2).sum()
        s2 = model.xi ** 2 + inputs.shape[0] * inputs.shape[1] * model.xi
        s3 = ((inputs - outputs) ** 2).sum() * torch.exp(-model.xi)
        loss = (s1 + s2 + s3) / latent.shape[0] / latent.shape[1]
        
        return self.lam * loss.reshape(())
    

# def get_M(fisher_ae, mu, slice_idx=None):
#     # TODO
#     # is there a more sufficient way to compute the Jacobians?
#     J = torch.stack([jacobian(fisher_ae.decoder, mu_i) for mu_i in mu])
#     if slice_idx is not None:
#         J = J[..., :slice_idx]
#     M = torch.matmul(torch.swapaxes(J, 1, 2), J) * torch.exp(-fisher_ae.xi) + torch.eye(J.shape[-1])[None, ...]
#     return M


# def sample_gaussian_variables(mu, sigma):
#     n = torch.randn_like(mu)
#     L = torch.linalg.cholesky(sigma)
#     z = torch.bmm(L, n[..., None]).reshape(mu.shape) + mu
#     return z


# def probabilistic_loss(fisher_ae, inputs, _):
#     latent_mean = fisher_ae.encoder(inputs)
#     M = get_M(fisher_ae, latent_mean)
#     latent = sample_gaussian_variables(latent_mean, torch.inverse(M))
#     outputs = fisher_ae.decoder(latent)

#     s1 = (latent ** 2).sum()
#     s2 = fisher_ae.xi ** 2 + inputs.shape[0] * inputs.shape[1] * fisher_ae.xi
#     s3 = ((inputs - outputs) ** 2).sum() * torch.exp(-fisher_ae.xi)
#     loss = (s1 + s2 + s3) / latent.shape[0] / latent.shape[1]
    
#     return loss.reshape(())


# def probabilistic_loss_joint_ae(joint_ae, inputs, _):
#     latent_mean = joint_ae.projection(joint_ae.encoder(inputs))
#     conserved = joint_ae.conserved(inputs)
#     M = get_M(joint_ae, torch.cat((latent_mean, conserved), axis=-1), slice_idx=latent_mean.shape[-1])
#     latent = sample_gaussian_variables(latent_mean, torch.inverse(M))
#     outputs = joint_ae.decoder(torch.cat((latent, conserved), axis=-1))

#     s1 = (latent ** 2).sum()
#     s2 = joint_ae.xi ** 2 + inputs.shape[0] * inputs.shape[1] * joint_ae.xi
#     s3 = ((inputs - outputs) ** 2).sum() / torch.exp(joint_ae.xi)
#     loss = (s1 + s2 + s3) / latent.shape[0] / latent.shape[1]
    
#     return loss.reshape(())


class ConstantForcingLoss(_Loss):
    def __init__(self, mu: float = 1.0, lam: float = 1.0):
        super.__init__(ConstantForcingLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.mu = 0
        self.lam = lam

    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor):
        loss = self.mse(outputs, target)
        idx = {}
        elements, indices = torch.unique(target, return_inverse=True, dim=0)
        for i in range(len(elements)):
            mask = (indices == i)
            predictions = outputs[mask]
            loss += self.mu * torch.sum((predictions - torch.mean(predictions, axis=0)) ** 2)
        return self.lam * loss
        
        
# def equality_forcing_loss(model, inputs, target, lam=1):
#     outputs = model(inputs)
#     mse = nn.MSELoss()
#     loss = mse(outputs, target)
#     idx = {}
#     elements, indices = torch.unique(target, return_inverse=True, dim=0)
#     for i in range(len(elements)):
#         mask = (indices == i)
#         predictions = outputs[mask]
#         loss += lam * torch.sum((predictions - torch.mean(predictions, axis=0)) ** 2)
#     return loss


class OrthogonalityLoss(_Loss):
    def __init__(self, orthogonal_model: nn.Module, possible_inputs: torch.Tensor = None, use_encoder: bool = False, lam: float = 1e-3):
        super(OrthogonalityLoss, self).__init__()

        self.orthogonal_model = orthogonal_model
        
        self.idx = None
        self.precomputed_jacobians = None
        if possible_inputs is not None:
            self.idx = {tuple(value): idx for idx, value in enumerate(possible_inputs)}
            self.precomputed_jacobians = compute_jacobians(possible_inputs, orthogonal_model(possible_inputs))
            
        self.use_encoder = use_encoder
        self.lam = lam

    def get_idx(self, inputs: torch.Tensor):
        return torch.tensor([self.idx[tuple(single_input)] for single_input in inputs])

    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor):
        if self.use_encoder:
            J = compute_jacobians(inputs, model.encoder(inputs))
        J = compute_jacobians(inputs, outputs)
        if self.precomputed_jacobians is None:
            orthogonal_J = compute_jacobians(inputs, self.orthogonal_model(inputs))
        else:
            orthogonal_J = self.precomputed_jacobians[self.get_idx(inputs)]
        loss = (torch.matmul(J, torch.swapaxes(orthogonal_J, 1, 2)) ** 2).mean()
        return self.lam * loss


class JointAEOrthogonalityLoss(_Loss):
    def __init__(self, lam: float = 1e-3):
        super(JointAEOrthogonalityLoss, self).__init__()

        self.lam = lam

    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor):
        J = compute_jacobians(inputs, model.projection(model.encoder(inputs)))
        orthogonal_J = compute_jacobians(inputs, model.conserved(inputs))
        loss = (torch.matmul(J, torch.swapaxes(orthogonal_J, 1, 2)) ** 2).mean()
        return self.lam * loss
        

def compute_jacobians(inputs: torch.Tensor, outputs: torch.Tensor):
    return torch.stack([grad(c_sum, inputs, create_graph=True)[0] for c_sum in outputs.sum(axis=0)], axis=1)
