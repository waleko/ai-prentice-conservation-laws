from typing import Optional, Tuple, List, Iterable

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm.autonotebook import tqdm

from autoencoders.GAELoss import l1_loss
from autoencoders.autoencoder import AE
from autoencoders.external_metrics import mse_neighborhood_metric, ranks_metric
from utils import PhysExperiment


class TrajectoryAutoencoderSuite:
    def __init__(self,
                 experiment: PhysExperiment,
                 epochs: int = 5000,
                 criterion: nn.Module = nn.MSELoss(),
                 l1_lambda: float = 0,
                 ae_class=AE,
                 ae_args=None,
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 log_prefix: Optional[str] = None,
                 apply_scaling: bool = True,
                 train_val_test_split: List[int] = None
                 ):
        if ae_args is None:
            ae_args = {}
        if log_prefix is None:
            self.full_exp_name = experiment.experiment_name
        else:
            self.full_exp_name = f"{log_prefix}_{experiment.experiment_name}"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using {device} device...")

        self.experiment = experiment
        self.epochs = epochs

        self.ae_class = ae_class
        self.ae_args = ae_args
        self.device = device

        self.criterion = criterion.to(device)
        self.mse_criterion = nn.MSELoss().to(device)
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size

        if train_val_test_split is None:
            train_val_test_split = [0.8, 0.2, 0.2]
        self.train_val_test_split = train_val_test_split
        self.apply_scaling = apply_scaling

    def train_traj_num(self,
                       random_seed: Optional[int] = 42,
                       bottleneck_dim_range: Optional[Iterable[int]] = None):
        traj = self.experiment.single_trajectory(random_seed)
        return self.train_traj_data(traj, bottleneck_dim_range)

    def train_traj_data(self,
                        traj: np.ndarray,
                        bottleneck_dim_range: Optional[Iterable[int]] = None
                        ) -> Tuple[List[nn.Module], List[float]]:
        if self.apply_scaling:
            traj = MaxAbsScaler().fit_transform(traj)
        if bottleneck_dim_range is None:
            bottleneck_dim_range = range(1, self.experiment.n_eff + 3)

        a, b, c = self.train_val_test_split
        a = int(a * traj.shape[0])
        b = int(b * traj.shape[0])
        c = traj.shape[0] - a - b
        traj_train, traj_val, traj_test = random_split(traj, [a, b, c])

        train_dataloader = DataLoader(traj_train, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(traj_val, batch_size=self.batch_size, shuffle=True)

        models = []
        model_losses = []

        for bottleneck_dim in tqdm(bottleneck_dim_range):
            model = self.__train_single(train_dataloader, valid_dataloader, bottleneck_dim)
            models.append(model)

            # test
            loss_val, mse = self.test(model, traj_test, bottleneck_dim)
            model_losses.append(loss_val)

        return models, model_losses

    def __train_single(self, train_dataloader, valid_dataloader, bottleneck_dim: int):
        model = self.ae_class(self.experiment.pt_dim, bottleneck_dim, **self.ae_args).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs // 10, gamma=0.5)

        counter = 0
        for epoch in tqdm(range(self.epochs)):
            train_losses = []
            train_mse_losses = []

            valid_losses = []
            valid_mse_losses = []

            model.train()
            for batch_pts in train_dataloader:
                inp = batch_pts.float().to(self.device)
                output = model(inp)
                loss = self.criterion(output, inp) + l1_loss(model.parameters(), self.l1_lambda)
                train_losses.append(loss.item())
                train_mse_losses.append(self.mse_criterion(output, inp).item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            for batch_pts in valid_dataloader:
                inp = batch_pts.float().to(self.device)
                output = model(inp)
                loss = self.criterion(output, inp) + l1_loss(model.parameters(), self.l1_lambda)
                valid_losses.append(loss.item())
                valid_mse_losses.append(self.mse_criterion(output, inp).item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_mse_loss = np.average(train_mse_losses)
            valid_mse_loss = np.average(valid_mse_losses)

            wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_train_loss": train_loss})
            wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_val_loss": valid_loss})
            wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_train_mse_loss": train_mse_loss})
            wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_val_mse_loss": valid_mse_loss})
            wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_lr": optimizer.param_groups[0]['lr']})
            scheduler.step()
        return model

    def test(self, model, traj_test: torch.Tensor, bottleneck_dim: int) -> Tuple[float, float]:
        model.eval()
        traj_test_np = np.array(traj_test)
        output = model(traj_test)
        output_np = output.detach().cpu().numpy()

        test_mse = mean_squared_error(traj_test_np, output_np)
        test_loss = self.criterion(traj_test, output).item()
        test_metric_mse_neighborhood = mse_neighborhood_metric(traj_test_np, output_np)

        test_metric_rank = ranks_metric(traj_test_np, output_np)
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_loss": test_loss})
        wandb.log({f"{self.full_exp_name}_all_test_loss": test_loss})
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_mse": test_mse})
        wandb.log({f"{self.full_exp_name}_all_test_mse": test_mse})
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_metric_rank": test_metric_rank})
        wandb.log({f"{self.full_exp_name}_all_{bottleneck_dim}_test_metric_rank": test_metric_rank})
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_metric_mse_neighborhood": test_metric_mse_neighborhood})
        wandb.log({f"{self.full_exp_name}_all_test_metric_mse_neighborhood": test_metric_mse_neighborhood})

        return test_loss, test_mse

    def analyze_n_eff(self, random_seed: Optional[int] = 42):
        traj = self.experiment.single_trajectory(random_seed)
        n_eff = self.experiment.n_eff

        res = self.train_traj_data(traj, [n_eff])
        model, test_loss = res[0][0], res[1][0]

        model.eval()
        full_traj = torch.Tensor(traj).to(self.device)
        with torch.no_grad():
            embedding = model.encoder(full_traj).detach().cpu().numpy()
            transformed = model(full_traj).detach().cpu().numpy()

            all_trajs = np.concatenate((traj, transformed))
            color = np.concatenate((np.zeros(shape=(traj.shape[0], 1)), np.ones(shape=(transformed.shape[0], 1))))
            exported = np.append(all_trajs, color, axis=1)
            table = wandb.Table(columns=self.experiment.column_names + ["transformed"], data=exported)
            wandb.log({f"{self.experiment.experiment_name} before/after": table})
        if n_eff == 1:
            # coloring
            traj_with_color = np.append(traj, embedding, axis=1)
            wandb.log({f"{self.full_exp_name} coloring for n_eff=1 embedding": wandb.Table(
                self.experiment.column_names + ["color"], data=traj_with_color)})
        elif n_eff == 2:
            # 2d embedding
            wandb.log(
                {f"{self.full_exp_name} 2d n_eff embedding": wandb.Table(["projection1", "projection2"], embedding)})
        elif n_eff == 3:
            # 3d embedding
            wandb.log({f"{self.full_exp_name} 3d n_eff embedding": wandb.Object3D(embedding)})
        else:
            wandb.alert(f"no visual representation for n_eff={n_eff} with experiment {self.full_exp_name}")

    def plot_errorbars(self,
                       model_losses: np.ndarray,
                       bottleneck_dim_range: Iterable[int],
                       decision_threshold: float):
        loss_mean = np.mean(model_losses.T, axis=1)
        loss_std = np.std(model_losses.T, axis=1)

        colors = [1 if x < self.experiment.n_eff else 0 for x in bottleneck_dim_range]

        plt.errorbar(bottleneck_dim_range, loss_mean, yerr=loss_std, fmt='none', ecolor=colors)
        plt.scatter(bottleneck_dim_range, loss_mean, c=colors)
        plt.axhline(y=decision_threshold, color='blue', linestyle='--')
        plt.xticks(bottleneck_dim_range)
        title = f"{self.full_exp_name} n_eff={self.experiment.n_eff}"
        plt.title(title)
        plt.xlabel("hidden layer size")
        plt.ylabel("loss after auto-encoder")
        plt.savefig(f"plot_{self.full_exp_name}")
        wandb.log({title: wandb.Image(f"plot_{self.full_exp_name}.png")})
        plt.close()
