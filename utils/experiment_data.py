import logging
from typing import Optional

from ipywidgets import widgets

from ai_prentice_wasserstein import DimensionalityPredictor
from .data_loader import *


class PhysExperiment:
    def __init__(self, experiment_name: str, n_conservation_laws: int, data: np.ndarray,
                 column_names: Union[List[str], None] = None, plot_config: Optional[List[Tuple[int, int]]] = None,
                 trajectory_animator: Optional[Callable] = None):
        assert len(data.shape) == 3
        self.experiment_name = experiment_name
        self.n_conservation_laws = n_conservation_laws
        self.data = data
        self.plot_config = plot_config
        self.traj_cnt = data.shape[0]
        self.traj_len = data.shape[1]
        self.pt_dim = data.shape[2]
        self.trajectory_animator = trajectory_animator

        if column_names is None:
            self.column_names = [f'Component #{x}' for x in range(self.pt_dim)]
        else:
            self.column_names = column_names

    def single_trajectory(self, random_seed=None):
        np.random.seed(random_seed)
        idx = np.random.randint(0, self.traj_cnt)
        return self.data[idx]

    def plot_trajectory(self, traj: np.ndarray):
        if self.plot_config is None:
            logging.warning(f"plot config for {self.experiment_name} is not specified")
            return

        # plot for every configuration
        N = len(self.plot_config)
        fig, axs = plt.subplots(N)
        fig.suptitle(f"Input trajectories: {self.experiment_name}")
        for idx in range(N):
            # components on the plot
            j, k = self.plot_config[idx]
            if N == 1:
                x = axs
            else:
                x = axs[idx]
            x.scatter(traj[:, j], traj[:, k])
            x.set_xlabel(f"Component #{j}")
            x.set_ylabel(f"Component #{k}")
        fig.show()

    @property
    def n_eff(self):
        return self.pt_dim - self.n_conservation_laws

    def animate_trajectories(self, count: int = 1):
        if self.trajectory_animator is None:
            print(f"No animator for {self.experiment_name}. Skipping...")
            return []
        trajs = self.data[np.random.randint(self.traj_cnt, size=count), :]
        res = []
        for traj in trajs:
            res.append(self.trajectory_animator(traj))
        return res

    def contrastive_data(self, traj_cnt: Optional[int] = None, traj_len: Optional[int] = None,
                         random_seed=None) -> np.ndarray:
        if traj_cnt is None:
            traj_cnt = self.traj_cnt
        if traj_len is None:
            traj_len = self.traj_len
        np.random.seed(random_seed)
        # indices = np.random.choice(self.traj_cnt, traj_cnt, replace=False) # fixme: bring back random
        indices = np.arange(traj_cnt)
        data = self.data[
            np.repeat(indices, traj_len),
            np.random.choice(self.traj_len, traj_cnt * traj_len)
        ]
        points = data.reshape(-1, self.pt_dim)
        indices = np.tile(indices[:, None], (1, traj_len)).reshape(-1, 1)
        return np.concatenate((indices, points), axis=1)

    def calc_umap_wd(self, traj_cnt: Optional[int] = None, traj_len: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Calculate UMAP embedding and Wasserstein distance matrix for the experiment
        @param traj_cnt: Number of trajectories to use
        @param traj_len: Number of points in each trajectory
        @return: Returns tuple of UMAP embedding and Wasserstein distance matrix
        """
        if traj_cnt is None:
            traj_cnt = self.traj_cnt
        if traj_len is None:
            traj_len = self.traj_len
        predictor = DimensionalityPredictor(self.experiment_name)
        predictor.fit(self.data[:traj_cnt, :traj_len])
        return predictor.embeddings[self.n_conservation_laws], predictor.ws_distance_matrix


class CsvPhysExperiment(PhysExperiment):
    def __init__(self, experiment_name: str, n_conservation_laws: int, traj_cnt: int = 200,
                 traj_len: int = 1000, column_names: Union[List[str], None] = None,
                 plot_config: Optional[List[Tuple[int, int]]] = None, start_index: int = 0,
                 trajectory_animator: Optional[Callable] = None):
        data = get_data(experiment_name, traj_cnt, traj_len, start_index=start_index)
        super().__init__(experiment_name, n_conservation_laws, data, column_names, plot_config, trajectory_animator)


def ipython_show_gif(filename: str):
    file = open(filename, "rb")
    image = file.read()
    return widgets.Image(
        value=image,
        format='gif'
    )


def get_npz_data(filename: str, preprocess, count: int = 200, sample_size: int = 1000):
    raw_data = np.load(f"trajectories/{filename}.npz")
    raw_data, params = raw_data["data"], raw_data["params"]
    if preprocess is None:
        data, raw_data = raw_data, raw_data
    else:
        data = preprocess(raw_data)
    flatten_data = data.reshape(data.shape[0], data.shape[1], -1)
    return flatten_data[:count, :sample_size], params[:count]


class NpzPhysExperiment(PhysExperiment):
    def __init__(self, experiment_name: str, n_conservation_laws: int, traj_cnt: int = 200,
                 traj_len: int = 1000, column_names: Union[List[str], None] = None,
                 plot_config: Optional[List[Tuple[int, int]]] = None, start_index: int = 0,
                 trajectory_animator: Optional[Callable] = None, filename: Optional[str] = None,
                 preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        if filename is None:
            filename = experiment_name
        data, params = get_npz_data(filename, preprocess, traj_cnt, traj_len)
        super().__init__(experiment_name, n_conservation_laws, data, column_names, plot_config, trajectory_animator)
        self.params = params
