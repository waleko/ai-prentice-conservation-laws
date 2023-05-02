from torch import nn


class AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, intermediate_dim: int = 10, intermediate_layers: int = 5,
                 act_fn=nn.Tanh):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            act_fn(),
            *[nn.Linear(intermediate_dim, intermediate_dim) if i % 2 == 0 else act_fn() for i in
              range(intermediate_layers * 2)],
            nn.Linear(intermediate_dim, hidden_dim),
            # act_fn(), # removed for contrastive learning with umap
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            act_fn(),
            *[nn.Linear(intermediate_dim, intermediate_dim) if i % 2 == 0 else act_fn() for i in
              range(intermediate_layers * 2)],
            nn.Linear(intermediate_dim, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
