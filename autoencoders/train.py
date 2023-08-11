from typing import Optional, Tuple, List, Iterable, Callable

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from tqdm.autonotebook import tqdm
from tqdm import trange
import copy

from autoencoders.autoencoder import AE, JointAE, RegressionNN
from autoencoders.external_metrics import mse_neighborhood_metric, ranks_metric
from autoencoders.losses import default_loss
from experiments.animator import Animator
import utils
from utils import PhysExperiment, gen_dist_matrix
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        (default: [0.8, 0.1, 0.1])
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


class JointAutoencoderSuite:
    def __init__(self,
                 experiment: PhysExperiment,
                 traj_cnt: int = 200,
                 traj_len: int = 1000,
                 cons_epochs: int = 100,
                 ae_epochs: int = 200,
                 criterion_conserved = default_loss,
                 criterion_ae = default_loss,
                 conserved_class=RegressionNN,
                 conserved_args=None,
                 joint_ae_class=JointAE,
                 joint_ae_args=None,
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 train_val_test_split: List[int] = None,
                 early_stopping_threshold: Optional[float] = 1e-5,
                 init_lr_conserved: float = 0.001,
                 init_lr_joint_ae: float = 1e-4,
                 dtype=torch.float32,
                 ):
        """
        @param experiment: experiment with trajectory data and ground truth information
        @param epochs: number of epochs for training
        @param loss: loss for training
        @param ae_class: Autoencoder class (see [AE])
        @param ae_args: Additional arguments to pass to the ae_class
        @param device: device to run the model on (default: auto)
        @param batch_size: batch size for learning
        @note batch_size should be a multiple of training size for GAE to work properly
        @param log_prefix: prefix for the suite that will appear in all wandb logs (default: none)
        @param apply_scaling: whether to apply min/max abs scaling to the trajectories (default: true)
        @param train_val_test_split: train, validation, test split proportions for the trajectories
        (default: [0.8, 0.1, 0.1])
        """
        if conserved_args is None:
            conserved_args = {}
        if joint_ae_args is None:
            joint_ae_args = {}
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using {device} device...")

        self.experiment = experiment
        self.traj_cnt = traj_cnt
        self.traj_len = traj_len
        self.cons_epochs = cons_epochs
        self.ae_epochs = ae_epochs
        
        self.conserved_class = conserved_class
        self.conserved_args = conserved_args
        self.joint_ae_class = joint_ae_class
        self.joint_ae_args = joint_ae_args
        self.device = device

        self.criterion_conserved = criterion_conserved
        self.criterion_ae = criterion_ae
        self.mse_criterion = nn.MSELoss()

        self.batch_size = batch_size

        if train_val_test_split is None:
            train_val_test_split = [0.6, 0.2, 0.2]
        self.train_val_test_split = train_val_test_split

        self.early_stopping_threshold = early_stopping_threshold
        self.init_lr_conserved = init_lr_conserved
        self.init_lr_joint_ae = init_lr_joint_ae
        self.dtype = dtype

    def umap_embedding(self, data):
        m, n, k = data.shape
        data = StandardScaler().fit_transform(data.reshape(m * n, k)).reshape(m, n, k)
        dmat = gen_dist_matrix(data)
        n_components = self.experiment.n_conservation_laws
        epochs = 20000 if n_components == 1 else None
        umap_model = UMAP(
            n_components=n_components,
            n_neighbors=data.shape[0] // 2,
            n_epochs=epochs,
            metric="precomputed",
        )
        umap_embedding = umap_model.fit_transform(dmat)
        return umap_embedding

    def train_conservation(self, train_data, train_target):
        model = self.conserved_class(self.experiment.pt_dim, self.experiment.n_conservation_laws, **self.conserved_args)
        model = train_loop(model, train_data, train_target, self.batch_size,
            self.init_lr_conserved, self.cons_epochs, self.criterion_conserved)
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def train_ae(self, train_data, valid_data, conserved):
        inp_dim = self.experiment.pt_dim
        cons_dim = self.experiment.n_conservation_laws
        dyn_dim = inp_dim - cons_dim
        model = self.joint_ae_class(conserved, inp_dim, dyn_dim, cons_dim, **self.joint_ae_args)
        model = train_loop(model, train_data, train_data, self.batch_size, self.init_lr_joint_ae,
            self.ae_epochs, self.criterion_ae, valid_data=valid_data, valid_target=valid_data)
        return model

    def train_all(self, save=True, test_conserved=True, test_joint_ae=True, extract_dynamics=None):
        data = self.experiment.data[:self.traj_cnt, :self.traj_len, :]
        params = self.experiment.params[:self.traj_cnt]
        a, b, c = self.train_val_test_split
        train, test, _, test_params = train_test_split(data, params, test_size=c)
        train, val = train_test_split(train, test_size=b / (a + b))
        
        print("Constructing UMAP embedding")
        umap_embedding = self.umap_embedding(train)
        
        print("\nTraining conserved NN")
        train_data = torch.tensor(flatten_trajectories(train), dtype=self.dtype)
        cons_target = torch.tensor(np.repeat(StandardScaler().fit_transform(umap_embedding), train.shape[1], axis=0), dtype=self.dtype)
        conserved = self.train_conservation(train_data, cons_target)
        
        print("\nTraining joint AE")
        valid_data = torch.tensor(flatten_trajectories(val), dtype=self.dtype)
        joint_ae = self.train_ae(train_data, valid_data, conserved)
        
        test_data = torch.tensor(flatten_trajectories(test), dtype=self.dtype)

        if save:
            torch.onnx.export(conserved, test_data, self.get_conserved_NN_name())
            torch.onnx.export(joint_ae, test_data, self.get_joint_ae_name())
        
        if test_conserved:
            self.test_conserved(conserved, test, test_params)
        if extract_dynamics is not None:
            self.test_dynamics(joint_ae, test, extract_dynamics)
        if test_joint_ae:
            self.test_joint_ae()

        return conserved, joint_ae
                          
    def get_conserved_NN_name(self):
        return f"conserved_{self.experiment.experiment_name}.onnx"
    
    def get_joint_ae_name(self):
        return f"join_ae_{self.experiment.experiment_name}.onnx"
    
    def test_conserved(self, conserved, test, test_params):
        data = torch.tensor(flatten_trajectories(test), dtype=self.dtype)

        predicted_conserved = conserved(data).detach()
        true_conserved = np.repeat(test_params, test.shape[1], axis=0)

        if self.experiment.n_conservation_laws == 1:
            utils.plot_embedding_vs_conserved_quantity(plt.gca(), predicted_conserved, true_conserved.T[0], "energy")
        elif self.experiment.n_conservation_laws == 2:
            utils.plot_all_2d(*plt.subplots(1, 2, figsize=(10, 5)), predicted_conserved, true_conserved, ["E1", "E2"])
        elif self.experiment.n_conservation_laws == 3:
            utils.plot_all_3d(*plt.subplots(3, 3, figsize=(15, 15)), predicted_conserved, true_conserved, ["E", "L", "$\\phi$"])
        else:
            print("WHAT THE HECK?")
        plt.show()
        print("\n")
        
    def test_dynamics(self, joint_ae, test, extract_dynamics):
        idx = np.argsort(np.std(test, axis=(1, 2)))[len(test) // 5:]
        data = torch.tensor(flatten_trajectories(test[idx]), dtype=self.dtype)
        true_dynamics = extract_dynamics(data)
        predicted_dynamics = joint_ae.extract_dynamics(data).detach()
        utils.plot_dynamics(plt, data, true_dynamics, predicted_dynamics)
        plt.show()
        print("\n\n")
        
    def test_joint_ae(self):
        # TODO
        pass


class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        target = self.targets[idx]
        return sample, target


def train_loop(
        model,
        train_data,
        train_target,
        batch_size,
        initial_lr,
        epochs,
        criterion=default_loss,
        valid_criterion=default_loss,
        optimizer_class=torch.optim.Adam,
        valid_data=None,
        valid_target=None,
        lam_l1=1e-5,
        lam_l2=1e-5,
    ):
    dataloader = DataLoader(CustomDataset(train_data, train_target), batch_size=batch_size, shuffle=True)
    optimizer = optimizer_class(model.parameters(), lr=initial_lr)
    best_model = None
    best_loss = torch.inf
    if valid_data is None:
        valid_data = train_data
        valid_target = train_target
    for epoch in trange(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            params = torch.concat([p.reshape(-1) for p in model.parameters()])
            loss = criterion(model, inputs, targets) + lam_l1 * torch.abs(params).sum() + lam_l2 * (params ** 2).mean()
            loss.backward()
            optimizer.step()
        train_loss = criterion(model, train_data, train_target)
        valid_loss = valid_criterion(model, valid_data, valid_target)
        valid_loss_log = f", Validation loss: {valid_loss}"
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = copy.deepcopy(model)
        # if (epoch + 1) % 5 == 0:
        #     print(f"Epoch: {epoch + 1}/{epochs}, Train loss: {train_loss}" + valid_loss_log)
    print(f"Best loss: {best_loss}")
    return best_model


def flatten_trajectories(data):
    return data.reshape(data.shape[0] * data.shape[1], -1)
