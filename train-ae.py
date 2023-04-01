# %% [markdown]
# In this notebook we try to estimate the dimensionality of the dynamic trajectory subspace.
# 
# ### Helpful articles
# - [Supplementary material from AI Poincare](https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.126.180604/poincare_supplemental_materialv2.pdf)
# - [Interpretable conservation law estimation by deriving the symmetries of dynamics from trained deep neural networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.103.033303) cited in AI Poincare.

# %%
# %pip install -r requirements.txt --quiet

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from tqdm.notebook import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# %%
import utils
from experiments import *

# %%
import wandb

# %%
wandb.init(project="Autoencoders Trajectory (average, cluster, lr)")

# %%
class AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, intermediate_dim: int):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" # cpu is faster ??

# %%
def train_model(train_dataloader, valid_dataloader, input_dim: int, experiment_name: str, hidden_layer_dim: int, epochs=1000, intermediate_dim=20, batch_size=10):
    model = AE(input_dim, hidden_layer_dim, intermediate_dim).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)

    counter = 0
    for epoch in tqdm(range(epochs)):
        train_losses = []
        valid_losses = []

        model.train()
        for batch_pts in train_dataloader:
            inp = batch_pts.float().to(device)
            output = model(inp)
            loss = criterion(output, inp)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        for batch_pts in valid_dataloader:
            inp = batch_pts.float().to(device)
            output = model(inp)
            loss = criterion(output, inp)
            valid_losses.append(loss.item())
        
        train_loss = np.sqrt(np.average(train_losses))
        valid_loss = np.sqrt(np.average(valid_losses))

        wandb.log({f"{experiment_name}_{hidden_layer_dim}_train_loss": train_loss})
        wandb.log({f"{experiment_name}_{hidden_layer_dim}_val_loss": valid_loss})
        wandb.log({f"{experiment_name}_{hidden_layer_dim}_lr": optimizer.param_groups[0]['lr']})
        scheduler.step()

        if valid_loss < decision_threshold / 2:
            counter += 1
        else:
            counter = 0
        if counter > 50:
            wandb.alert(title="Early stopping", text=f"Early stopping for {experiment_name}{hidden_layer_dim} on epoch #{epoch}/{epochs}")
            print("Early stopping")
            break

    return model

# %%
def full_train(traj, experiment_name, max_hidden_size, epochs=1000, intermediate_dim=20, batch_size=32, start_embedding_size=1):
    traj_scaled = MaxAbsScaler().fit_transform(traj) # scale

    traj_train, traj_val, traj_test = random_split(traj_scaled, [0.8, 0.1, 0.1])

    train_dataloader = DataLoader(traj_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(traj_val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(traj_test, batch_size=batch_size, shuffle=True)

    models = []
    model_losses = []
    criterion = nn.MSELoss().to(device)
    
    for hidden_layer_size in tqdm(range(start_embedding_size, max_hidden_size + 1)):
        model = train_model(train_dataloader, valid_dataloader, traj.shape[1], experiment_name, hidden_layer_size, intermediate_dim=intermediate_dim, batch_size=batch_size, epochs=epochs)
        models.append(model)
        # test loss
        model.eval()
        test_losses = []
        for batch_pts in test_dataloader:
            inp = batch_pts.float().to(device)
            output = model(inp)
            loss = criterion(output, inp)
            test_losses.append(loss.item())
        test_loss = np.sqrt(np.average(test_losses))
        wandb.log({f"test_loss_{experiment_name}_{hidden_layer_size}": test_loss})
        model_losses.append(test_loss)
    return models, model_losses

# %%
decision_threshold = 0.05

# %%
def plot_losses_errorbars(experiment_name, n_eff, model_losses, max_hidden_size, start_embedding_size=1):
    loss_mean = np.mean(model_losses.T, axis=1)
    loss_std = np.std(model_losses.T, axis=1) / 3

    hidden_sizes = range(start_embedding_size, max_hidden_size + 1)
    colors = [1 if x < n_eff else 0 for x in hidden_sizes]

    plt.errorbar(hidden_sizes, loss_mean, yerr=loss_std, fmt='none')
    plt.scatter(hidden_sizes, loss_mean, c=colors)
    plt.axhline(y=decision_threshold, color='blue', linestyle='--')
    plt.xticks(hidden_sizes)
    title = f"{experiment_name} n_eff={n_eff}"
    plt.title(title)
    plt.xlabel("hidden layer size")
    plt.ylabel("r.m.s. loss after auto-encoder")
    plt.savefig(f"plot_{experiment_name}_2e-2")
    wandb.log({title: wandb.Image(f"plot_{experiment_name}_2e-2.png")})
    plt.show()
    plt.close()

# %%
def train_n_eff(exp: PhysExperiment, epochs=5000, intermediate_dim=20, batch_size=10):
    traj = exp.single_trajectory(42)
    n_eff = exp.n_eff
    model = train_model(traj, exp.experiment_name, n_eff, epochs, intermediate_dim, batch_size)
    test_traj = torch.Tensor(traj).to(device)
    with torch.no_grad():
        embedding = model.encoder(test_traj).detach().cpu().numpy()
        transformed = model(test_traj).detach().cpu().numpy()

        all_trajs = np.concatenate((traj, transformed))
        color = np.concatenate((np.zeros(shape=(traj.shape[0], 1)), np.ones(shape=(transformed.shape[0], 1))))
        exported = np.append(all_trajs, color, axis=1)
        table = wandb.Table(columns=exp.column_names + ["transformed"], data=exported)
        wandb.log({f"{exp.experiment_name} before/after": table})
    if n_eff == 1:
        # coloring
        traj_with_color = np.append(traj, embedding, axis=1)
        wandb.log({f"{exp.experiment_name} coloring for n_eff=1 embedding": wandb.Table(exp.column_names + ["color"], data=traj_with_color)})
    elif n_eff == 2:
        # 2d embedding
        wandb.log({f"{exp.experiment_name} 2d n_eff embedding": wandb.Table(["projection1", "projection2"], embedding)})
    elif n_eff == 3:
        # 3d embedding
        wandb.log({f"{exp.experiment_name} 3d n_eff embedding": wandb.Object3D(embedding)})
    else:
        wandb.alert(f"no visual representation for n_eff={n_eff} with experiment {exp.experiment_name}")

# %%
# old_device = device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for exp in [KdV]:
#     traj = exp.single_trajectory(42)
#     max_hidden_size = exp.n_eff + 2
#     min_hidden_size = exp.n_eff - 3
    
#     models, epochs = full_train(traj, exp.experiment_name, max_hidden_size, epochs=5000, batch_size=20, start_embedding_size=min_hidden_size, intermediate_dim=500)
#     plot_losses(exp.experiment_name, exp.n_eff, epochs, max_hidden_size, start_embedding_size=min_hidden_size)
# device = old_device

# %%
for exp in tqdm(common_experiments, position=0):
    max_hidden_size = exp.n_eff + 2
    losses_2d = []
    for _ in tqdm(range(3), position=1, leave=False):
        traj = exp.single_trajectory()
        models, epochs = full_train(traj, exp.experiment_name, max_hidden_size, epochs=5000, batch_size=20)
        losses_2d.append(epochs)
    losses_2d = np.array(losses_2d)
    print(losses_2d)
    # wandb.log({exp.experiment_name: losses_2d})
    # plot_losses_errorbars(exp.experiment_name, exp.n_eff, losses_2d, max_hidden_size)

# %%
wandb.finish()

# %%
# wandb.init(project="AE embedding & Turing")
# ll = [
#     # Pendulum,
#     # HarmonicOscillator,
#     # DoublePendulum,
#     # DoublePendulumSmallEnergy,
#     DoublePendulumLargeEnergy,
#     # CoupledOscillator,
#     # KeplerProblem,
#     # Sphere5
# ]
# device = "cpu"
# for exp in ll:
#     train_n_eff(exp, epochs=5000, batch_size=32)

# %%
# device = "cuda"
# for exp in [Turing]:
#     traj = exp.single_trajectory(42)
#     models, epochs = full_train(traj, exp.experiment_name, 103, epochs=5000, start_embedding_size=97, intermediate_dim=200)
#     plot_losses(exp.experiment_name, exp.n_eff, epochs, 103, start_embedding_size=97)
# wandb.finish()


