from typing import Optional, Tuple, List, Callable

import numpy as np
import torch
import wandb
from sklearn.preprocessing import MaxAbsScaler
from torch import nn
from torch.utils.data import DataLoader, default_collate
from torch.utils.data import random_split
from tqdm.autonotebook import tqdm

from autoencoders.autoencoder import AE
from utils import PhysExperiment


class TrajectoryContrastiveSuite:
    def __init__(self,
                 experiment: PhysExperiment,
                 epochs: int = 5000,
                 criterion: nn.Module = nn.MSELoss(),
                 additional_loss: Optional[Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor]] = None,
                 contrastive_loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 ae_class=AE,
                 ae_args=None,
                 device: Optional[str] = None,
                 batch_size: int = 512,
                 log_prefix: Optional[str] = None,
                 apply_scaling: bool = True,
                 train_val_test_split: List[int] = None,
                 do_animate: bool = False,
                 early_stopping_threshold: Optional[float] = 1e-5,
                 init_lr: float = 0.01
                 ):
        """
        A suite for training an autoencoder on a physical experiment with contrastive loss.
        It contains methods for training, testing and inference.

        @param experiment: Physical experiment (with data and infos)
        @param epochs: Number of iterations during training
        @param criterion: Loss between input and output of the autoencoder (default: MSE)
        @param additional_loss: Additional loss between input and output with the model information (e.g. for L1)
        @param contrastive_loss: Contrastive loss used. Shall accept embedding and labels from a batch.
        @param ae_class: Base class for autoencoder model. See `AE`
        @param ae_args: Additional arguments passed to the ae_class
        @param device: Device for computations
        @param batch_size: Batch size for training
        @param log_prefix: Additional prefix for all wandb logs
        @param apply_scaling: Whether to apply scaling of training data
        @param train_val_test_split: Train, validation, test split ration (default: [0.8, 0.1, 0.1])
        @param do_animate: Whether to animate the training process (currently unsupported)
        @param early_stopping_threshold: Optional threshold for early stopping
        @param init_lr: Initial lr for the optimizer
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
        if contrastive_loss is None:
            contrastive_loss = lambda _, __: torch.tensor([0.0]).to(self.device)
        self.contrastive_loss = contrastive_loss

        self.batch_size = batch_size

        if train_val_test_split is None:
            train_val_test_split = [0.8, 0.1, 0.1]
        self.train_val_test_split = train_val_test_split
        self.apply_scaling = apply_scaling

        # self.animator = Animator(self.experiment, self.full_exp_name)
        self.animator = None
        self.do_animate = do_animate

        self.early_stopping_threshold = early_stopping_threshold
        self.init_lr = init_lr

    def train(self, traj_cnt: Optional[int] = None, traj_len: Optional[int] = None):
        """
        Trains the autoencoder with data from the experiment
        @param traj_cnt: Number of trajectories to use (default: all)
        @param traj_len: Number of points per trajectory (default: all)
        @return: Trained autoencoder
        """
        trajs = self.experiment.contrastive_data(traj_cnt, traj_len)
        return self.train_traj_data(trajs)

    def __parse_data(self, x):
        p = default_collate(x).type(dtype=torch.float32).to(self.device)
        return p[:, :1], p[:, 1:]

    def train_traj_data(self, trajs: np.ndarray) -> nn.Module:
        """
        Trains the autoencoder with given trajectory data
        @param trajs: Trajectory data
        @return: Trained autoencoder
        """
        # scale
        if self.apply_scaling:
            trajs = MaxAbsScaler().fit_transform(trajs)

        # split the trajectory
        a, b, c = self.train_val_test_split
        assert a + b + c == 1.0
        a = int(a * trajs.shape[0])
        b = int(b * trajs.shape[0])
        c = trajs.shape[0] - a - b
        assert a > 0 and b > 0 and c > 0
        traj_train, traj_val, traj_test = random_split(trajs, [a, b, c])

        train_dataloader = DataLoader(traj_train, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.__parse_data)
        valid_dataloader = DataLoader(traj_val, batch_size=self.batch_size, shuffle=True, collate_fn=self.__parse_data)
        test_dataloader = DataLoader(traj_test, batch_size=self.batch_size, shuffle=True, collate_fn=self.__parse_data)

        if self.do_animate:
            # todo
            # self.animator.start(torch.tensor(trajs).to(self.device).float(), f"b{bottleneck_dim}")
            pass

        model = self.__train_single(train_dataloader, valid_dataloader)

        if self.do_animate:
            fname = self.animator.save()
            wandb.log({f"contrastive_{self.full_exp_name}_animation": wandb.Video(fname)})

        # test
        self.test(model, test_dataloader)

        return model

    def __train_single(self, train_dataloader, valid_dataloader) -> nn.Module:
        # get model
        model = self.ae_class(self.experiment.pt_dim, self.experiment.pt_dim, **self.ae_args).to(self.device)
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
            for labels, inp in train_dataloader:
                embedding = model.encoder(inp)
                output = model.decoder(embedding)

                loss = self.criterion(inp, output) + \
                       self.additional_loss(inp, output, model) + \
                       self.contrastive_loss(embedding[:, self.experiment.n_eff:], labels)
                train_losses.append(loss.item())
                train_mse_losses.append(self.mse_criterion(inp, output).item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            for labels, inp in valid_dataloader:
                embedding = model.encoder(inp)
                output = model.decoder(embedding)

                loss = self.criterion(inp, output) + \
                       self.additional_loss(inp, output, model) + \
                       self.contrastive_loss(embedding[:, self.experiment.n_eff:], labels)
                valid_losses.append(loss.item())
                valid_mse_losses.append(self.mse_criterion(inp, output).item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_mse_loss = np.average(train_mse_losses)
            valid_mse_loss = np.average(valid_mse_losses)

            wandb.log({f"contrastive_{self.full_exp_name}_train_loss": train_loss})
            wandb.log({f"contrastive_{self.full_exp_name}_val_loss": valid_loss})
            wandb.log({f"contrastive_{self.full_exp_name}_train_mse_loss": train_mse_loss})
            wandb.log({f"contrastive_{self.full_exp_name}_val_mse_loss": valid_mse_loss})
            wandb.log({f"contrastive_{self.full_exp_name}_lr": optimizer.param_groups[0]['lr']})
            scheduler.step()

            # animate
            if self.do_animate:
                # self.animator.forward_and_log(model)
                pass

            # early stopping
            if self.early_stopping_threshold is not None and valid_loss < self.early_stopping_threshold:
                counter += 1
            else:
                counter = 0
            if counter > 50:
                wandb.alert(title="Early stopping",
                            text=f"Early stopping for {self.full_exp_name} on epoch #{epoch}/{self.epochs}")
                print("Early stopping")
                break
        return model

    def test(self, model, test_dataloader) -> Tuple[float, float]:
        """
        Tests the model on the given test data
        @param model: Trained autoencoder
        @param test_dataloader: Test data
        @return: Loss (used for training) and MSE loss
        """
        model.eval()

        test_losses = []
        test_mse_losses = []

        for labels, inp in test_dataloader:
            embedding = model.encoder(inp)
            output = model.decoder(embedding)

            loss = self.criterion(inp, output) + \
                   self.additional_loss(inp, output, model) + \
                   self.contrastive_loss(embedding[:, self.experiment.n_eff:], labels)
            test_losses.append(loss.item())
            test_mse_losses.append(self.mse_criterion(inp, output).item())

        # traj_test_np = test_inp.detach().cpu().numpy()
        # output_np = output.detach().cpu().numpy()

        test_loss = np.average(test_losses)
        test_mse = np.average(test_mse_losses)
        # test_metric_mse_neighborhood = mse_neighborhood_metric(traj_test_np, output_np)
        # test_metric_rank = ranks_metric(traj_test_np, output_np)

        wandb.log({f"contrastive_{self.full_exp_name}_test_loss": test_loss})
        wandb.log({f"contrastive_{self.full_exp_name}_test_mse": test_mse})
        # wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_metric_rank": test_metric_rank})
        # wandb.log({f"{self.full_exp_name}_{bottleneck_dim}_test_metric_mse_neighborhood": test_metric_mse_neighborhood})

        return test_loss, test_mse

    def try_transform(self, model: nn.Module, traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies trained autoencoder to given trajectory
        @param model: Trained autoencoder
        @param traj: Trajectory to transform
        @return: Tuple of decoded trajectory, dynamic components and conserved quantities
        """
        x = torch.tensor(traj, dtype=torch.float32).to(self.device)
        embedding = model.encoder(x)
        decoded = model.decoder(x)
        return decoded.detach().cpu().numpy(), \
            embedding[:, :self.experiment.n_eff].detach().cpu().numpy(), \
            embedding[:, self.experiment.n_eff:].detach().cpu().numpy()

    def try_transform_num(self, model: nn.Module, random_seed: Optional[int] = 42) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies trained autoencoder to trajectory from experiment data
        @param model: Trained autoencoder
        @param random_seed: Seed for trajectory generation
        @return: Tuple of trajectory, decoded trajectory, dynamic components and conserved quantities
        """
        traj = self.experiment.single_trajectory(random_seed)
        res = self.try_transform(model, traj)
        return traj, *res
