from typing import Optional, Tuple, List, Iterable, Callable

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

from autoencoders.autoencoder import AE
from autoencoders.external_metrics import mse_neighborhood_metric, ranks_metric
from experiments.animator import Animator
from utils import PhysExperiment


class TrajectoryAutoencoderSuite:
    """
    Autoencoder tool for training an autoencoder for trajectory embedding
    """

    def __init__(self,
                 experiment: PhysExperiment,
                 epochs: int = 5000,
                 criterion: nn.Module = nn.MSELoss(),
                 additional_loss: Optional[Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor]] = None,
                 ae_class=AE,
                 ae_args=None,
                 device: Optional[str] = None,
                 batch_size: int = 50,
                 log_prefix: Optional[str] = None,
                 apply_scaling: bool = True,
                 train_val_test_split: List[int] = None,
                 do_animate: bool = False,
                 early_stopping_threshold: Optional[float] = 1e-5,
                 init_lr: float = 0.01
                 ):
        """
        @param experiment: experiment with trajectory data and ground truth information
        @param epochs: number of epochs for training
        @param criterion: loss for training
        @param additional_loss: additional loss (like l1, sparse, e.g.) that is added to the [criterion].
        Inner model properties can be used.
        @param ae_class: Autoencoder class (see [AE])
        @param ae_args: Additional arguments to pass to the ae_class
        @param device: device to run the model on (default: auto)
        @param batch_size: batch size for learning
        @note batch_size should be a multiple of training size for GAE to work properly
        @param log_prefix: prefix for the suite that will appear in all wandb logs (default: none)
        @param apply_scaling: whether to apply min/max abs scaling to the trajectories (default: true)
        @param train_val_test_split: train, validation, test split proportions for the trajectories
        (default: [0.8, 0.1, 0.1]
        """
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

        if additional_loss is None:
            # constant zero
            additional_loss = lambda _, __, ___: torch.tensor([0.0]).to(self.device)
        self.additional_loss = additional_loss

        self.batch_size = batch_size

        if train_val_test_split is None:
            train_val_test_split = [0.8, 0.1, 0.1]
        self.train_val_test_split = train_val_test_split
        self.apply_scaling = apply_scaling

        self.animator = Animator(self.experiment, self.full_exp_name)
        self.do_animate = do_animate

        self.early_stopping_threshold = early_stopping_threshold
        self.init_lr = init_lr

    def train_traj_num(self,
                       random_seed: Optional[int] = 42,
                       bottleneck_dim_range: Optional[Iterable[int]] = None,
                       analyze_n_eff: bool = True):
        traj = self.experiment.single_trajectory(random_seed)
        return self.train_traj_data(traj, bottleneck_dim_range, analyze_n_eff)

    def train_traj_data(self,
                        traj: np.ndarray,
                        bottleneck_dim_range: Optional[Iterable[int]] = None,
                        analyze_n_eff: bool = True
                        ) -> Tuple[List[nn.Module], List[float]]:
        """
        Train models for the given trajectory
        @param bottleneck_dim_range: bottleneck layer sizes for models
        """
        # scale
        if self.apply_scaling:
            traj = MaxAbsScaler().fit_transform(traj)
        # default range
        if bottleneck_dim_range is None:
            bottleneck_dim_range = range(1, self.experiment.n_eff + 3)

        # split the trajectory
        a, b, c = self.train_val_test_split
        assert a + b + c == 1.0
        a = int(a * traj.shape[0])
        b = int(b * traj.shape[0])
        c = traj.shape[0] - a - b
        assert a > 0 and b > 0 and c > 0
        traj_train, traj_val, traj_test = random_split(traj, [a, b, c])

        train_dataloader = DataLoader(traj_train, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(traj_val, batch_size=self.batch_size, shuffle=True)

        models = []
        model_losses = []

        for bottleneck_dim in tqdm(bottleneck_dim_range):
            if self.do_animate:
                self.animator.start(torch.tensor(traj).to(self.device).float(), f"b{bottleneck_dim}")

            # train
            model = self.__train_single(train_dataloader, valid_dataloader, bottleneck_dim)

            if self.do_animate:
                fname = self.animator.save()
                wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_animation": wandb.Video(fname)})
            models.append(model)

            # test
            loss_val, mse = self.test(model, traj_test, bottleneck_dim)
            model_losses.append(loss_val)

            # analyze n_eff model if requested
            if bottleneck_dim == self.experiment.n_eff and analyze_n_eff:
                self.analyze_model(model, traj)

        return models, model_losses

    def __train_single(self, train_dataloader, valid_dataloader, bottleneck_dim: int):
        # get model
        model = self.ae_class(self.experiment.pt_dim, bottleneck_dim, **self.ae_args).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr)
        # scheduler for lr change
        # todo: make configurable
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
                loss = self.criterion(inp, output) + self.additional_loss(inp, output, model)
                train_losses.append(loss.item())
                train_mse_losses.append(self.mse_criterion(inp, output).item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            for batch_pts in valid_dataloader:
                inp = batch_pts.float().to(self.device)
                output = model(inp)
                loss = self.criterion(inp, output) + self.additional_loss(inp, output, model)
                valid_losses.append(loss.item())
                valid_mse_losses.append(self.mse_criterion(inp, output).item())

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

            # animate
            if self.do_animate:
                self.animator.forward_and_log(model)

            # early stopping
            if self.early_stopping_threshold is not None and valid_loss < self.early_stopping_threshold:
                counter += 1
            else:
                counter = 0
            if counter > 50:
                wandb.alert(title="Early stopping",
                            text=f"Early stopping for {self.full_exp_name}_{bottleneck_dim} on epoch #{epoch}/{self.epochs}")
                print("Early stopping")
                break
        return model

    def test(self, model, traj_test, bottleneck_dim: int) -> Tuple[float, float]:
        model.eval()
        traj_test = torch.tensor(np.array(traj_test)).to(self.device).float()
        output = model(traj_test)
        test_inp = traj_test
        traj_test_np = test_inp.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        test_mse = mean_squared_error(traj_test_np, output_np)
        test_loss = (self.criterion(test_inp, output)).item()  # + self.additional_loss(test_inp, output, model)
        test_metric_mse_neighborhood = mse_neighborhood_metric(traj_test_np, output_np)

        test_metric_rank = ranks_metric(traj_test_np, output_np)
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_loss": test_loss})
        wandb.log({f"{self.full_exp_name}_all_test_loss": test_loss})
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_mse": test_mse})
        wandb.log({f"{self.full_exp_name}_all_test_mse": test_mse})
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_metric_rank": test_metric_rank})
        wandb.log({f"{self.full_exp_name}_all_test_metric_rank": test_metric_rank})
        wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_metric_mse_neighborhood": test_metric_mse_neighborhood})
        wandb.log({f"{self.full_exp_name}_all_test_metric_mse_neighborhood": test_metric_mse_neighborhood})

        return test_loss, test_mse

    def analyze_n_eff(self, random_seed: Optional[int] = 42):
        """
        Trains the model with bottleneck layer size equal to n_eff.

        Logs additional information to wandb, like embeddings and before/after data.
        """
        traj = self.experiment.single_trajectory(random_seed)
        n_eff = self.experiment.n_eff

        self.train_traj_data(traj, [n_eff], analyze_n_eff=True)

    def analyze_model(self, model: nn.Module, traj: np.ndarray, bottleneck_dim: Optional[int] = None):
        if bottleneck_dim is None:
            bottleneck_dim = self.experiment.n_eff
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
        # coloring
        traj_with_color = np.append(traj, embedding, axis=1)
        wandb.log({f"{self.full_exp_name} colored by embedding projection": wandb.Table(
            self.experiment.column_names + [f"projection{idx}" for idx in range(bottleneck_dim)],
            data=traj_with_color)})

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
