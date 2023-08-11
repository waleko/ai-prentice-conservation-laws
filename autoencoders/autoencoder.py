import torch
from torch import nn
from abc import abstractmethod


class RegressionNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, intermediate_dim: int=64, intermediate_layers: int=4, act_fn=nn.ReLU, last_act=False):
        super(RegressionNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if last_act:
            last_activation = (act_fn(),)
        else:
            last_activation = ()

        intermediate_layer = nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim),
                act_fn(),
            )
        self.f = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                act_fn(),
                intermediate_layer * (intermediate_layers - 1)
                nn.Linear(intermediate_dim, output_dim),
                *last_activation,
            )

    def forward(self, input):
        output = self.f(input)
        return output
    
    
class PeriodicEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, periodic_dim: int = 0, intermediate_dim: int = 64, intermediate_layers: int = 4, act_fn=nn.ReLU):
        super(PeriodicEncoder, self).__init__()
        self.periodic_dim = periodic_dim
        self.encoder = RegressionNN(input_dim, hidden_dim + periodic_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x_periodic, x_linear = x[..., :2 * self.periodic_dim], x[..., 2 * self.periodic_dim:]
        x_periodic = x_periodic.reshape((*x_periodic.shape[:-1], self.periodic_dim, 2))
        x_periodic = x_periodic / (x_periodic ** 2).sum(axis=-1)[..., None] ** 0.5
        x_periodic = x_periodic.reshape((*x_periodic.shape[:-2], -1))
        return torch.cat((x_periodic, x_linear), axis=-1)


class AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, intermediate_dim: int = 10, intermediate_layers: int = 5,
                 act_fn=nn.Tanh, last_act_encoder=False):
        super(AE, self).__init__()
        self.encoder = RegressionNN(input_dim, hidden_dim, intermediate_dim, intermediate_layers, act_fn, last_act_encoder)
        self.decoder = RegressionNN(hidden_dim, input_dim, intermediate_dim, intermediate_layers, act_fn)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Projection(nn.Module):
    def __init__(self, dim):
        super(Projection, self).__init__()

        self.dim = dim

    @abstractmethod
    def parametrization(self, x):
        pass


class CombinedProjection(Projection):
    def __init__(self, projections):
        dim = sum([projection.dim for projection in projections])
        
        super(CombinedProjection, self).__init__(dim)
        
        self.projections = projections
        
        self.slices = []
        i = 0
        for projection in projections:
            j = i + projection.dim
            self.slices.append((projection, i, j))
            i = j

    def forward(self, x):
        return torch.cat([projection(x[..., i:j]) for projection, i, j in self.slices], axis=-1)

    def parametrization(self, x):
        return torch.cat([projection.parametrization(x[..., i:j]) for projection, i, j in self.slices], axis=-1)


class SnProjection(Projection):
    def __init__(self, n):
        super(SnProjection, self).__init__(n + 1)

    def forward(self, x):
        return x / (x ** 2).sum(axis=-1)[..., None] ** 0.5

    def parametrization(self, x):
        sphere = self.forward(x)
        theta_lst = []
        while sphere.shape[-1] != 2:
            theta = torch.arccos(sphere[..., 0])[..., None]
            theta_lst.append(theta)
            sphere = sphere[..., 1:] / torch.sin(theta)
        phi = (torch.arccos(sphere[..., 0]) * torch.sign(sphere[..., 1]))[..., None]
        return torch.cat(theta_lst + [phi], axis=-1)


class IdProjection(Projection):
    def __init__(self, n):
        super(IdProjection, self).__init__(n)

    def forward(self, x):
        return x

    def parametrization(self, x):
        return x


def get_torus_projection():
    circle_projection = SnProjection(1)
    torus_projection = CombinedProjection([circle_projection, circle_projection])
    return torus_projection


class JointAE(nn.Module):
    def __init__(
        self,
        conserved: RegressionNN,
        input_dim: int,
        dyn_dim: int,
        cons_dim: int,
        projection: Projection=None,
        intermediate_dim: int=64,
        intermediate_layers: int=4,
        act_fn=nn.PReLU
    ):
        super(JointAE, self).__init__()

        self.input_dim = input_dim
        self.dyn_dim = dyn_dim
        self.cons_dim = conserved.output_dim

        if projection is None:
            projection = IdProjection(dyn_dim)
        
        self.encoder = RegressionNN(input_dim, projection.dim, intermediate_dim, intermediate_layers, act_fn)
        self.projection = projection
        self.conserved = conserved
        self.decoder = RegressionNN(self.projection.dim + self.cons_dim, input_dim, intermediate_dim, intermediate_layers, act_fn, last_act=False)

        self.xi = nn.Parameter(torch.randn(1))
        
    def forward(self, input):
        dyn = self.encoder(input)
        with torch.no_grad():
            cons = self.conserved(input)
        hidden = torch.cat((self.projection(dyn), cons), axis=-1)
        output = self.decoder(hidden)
        return output
    
    def extract_dynamics(self, x):
        dyn = self.encoder(x)
        return self.projection.parametrization(dyn)


class FisherAE(AE):
    def __init__(self, input_dim: int, hidden_dim: int, intermediate_dim: int=64, intermediate_layers: int=4, act_fn=nn.PReLU):
        super(FisherAE, self).__init__(input_dim, hidden_dim, intermediate_dim, intermediate_layers, act_fn)

        self.xi = nn.Parameter(torch.randn(1))
