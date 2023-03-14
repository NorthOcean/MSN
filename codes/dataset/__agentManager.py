"""
@Author: Conghao Wong
@Date: 2022-08-03 10:50:46
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-23 20:36:01
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..base import BaseManager
from ..basemodels.layers import _BaseTransformLayer, get_transform_layers
from ..constant import INPUT_TYPES
from ..utils import POOLING_BEFORE_SAVING
from .maps import SocialMapManager, TrajMapManager
from .trajectories import Agent, AnnotationManager


class AgentManager(BaseManager):
    """
    AgentManager
    ---
    Structure to manage several `Agent` objects.
    The `AgentManager` object is managed by the `DatasetsManager` object.

    Member Managers
    ---
    - Trajectory map manager (optional, dynamic): type is `TrajMapManager`;
    - Social map manager (optional, dynamic): type is `SocialMapManager`.

    Public Methods
    ---
    ```python
    # concat agents to this `AgentManager`
    (method) append: (self: Self@AgentManager, target: Any) -> None

    # set inputs and outputs
    (method) set: (self: Self@AgentManager, dimension: int, 
                   inputs_type: list[str],
                   labels_type: list[str]) -> None

    # get inputs
    (method) get_inputs: (self: Self@AgentManager) -> list[Tensor]

    # get labels
    (method) get_labels: (self: Self@AgentManager) -> list[Tensor]

    # make inputs and labels into a dataset object
    (method) make_dataset: (self: Self@AgentManager, 
                            shuffle: bool = False) -> DatasetV2

    # save all agents' data
    (method) save: (self: Self@AgentManager, save_dir: str) -> None

    # load from saved agents' data
    (method) load: (cls: Type[Self@AgentManager], path: str) -> AgentManager
    ```

    Context Map Methods
    ---
    ```python
    # init map managers that manage to make context maps
    (method) init_map_managers: (self: Self@AgentManager, 
                                 map_type: str, 
                                 base_path: str) -> None

    #  load context maps to `Agent` objects
    (method) load_maps: (self: Self@AgentManager) -> None
    ```
    """

    def __init__(self, manager: BaseManager, name='Agent Manager'):
        super().__init__(manager=manager, name=name)

        self._agents: list[Agent] = []
        self.model_inputs = None
        self.model_labels = None

        # Context map managers
        self.t_manager: TrajMapManager = None
        self.s_manager: SocialMapManager = None

        # file root paths
        self.base_path: str = None
        self.npz_path: str = None
        self.maps_dir: str = None

        # Transform layer
        self.t_layers: dict[str, _BaseTransformLayer] = {}

    @property
    def agents(self) -> list[Agent]:
        return self._agents

    @agents.setter
    def agents(self, value: list[Agent]) -> list[Agent]:
        self._agents = self.update_agents(value)

    @property
    def picker(self) -> AnnotationManager:
        ds_manager = self.manager
        train_manager = ds_manager.manager
        return train_manager.get_member(AnnotationManager)

    def set_path(self, npz_path: str):
        self.npz_path = npz_path
        self.base_path = npz_path.split('.np')[0]
        self.maps_dir = self.base_path + '_maps'

    def update_agents(self, agents: list[Agent]):
        for a in agents:
            a.agent_manager = self
        return agents

    def append(self, target):
        self._agents += self.update_agents(target.agents)

    def set_types(self, inputs_type: list[str], labels_type: list[str]):
        """
        Set the type of model inputs and outputs.
        Accept all types in `INPUT_TYPES`.
        """
        self.model_inputs = inputs_type
        self.model_labels = labels_type

    def get_inputs(self) -> list[tf.Tensor]:
        """
        Get all model inputs from agents.
        """
        return [self._get(T) for T in self.model_inputs]

    def get_labels(self) -> list[tf.Tensor]:
        """
        Get all model labels from agents.
        """
        return [self._get(T) for T in self.model_labels]

    def get_inputs_and_labels(self) -> list[tf.Tensor]:
        """
        Get model inputs and labels (only trajectories) from all agents.
        """
        inputs = self.get_inputs()
        labels = self.get_labels()

        return tuple(inputs + labels)

    def make_dataset(self, shuffle=False) -> tf.data.Dataset:
        """
        Get inputs from all agents and make the `tf.data.Dataset`
        object. Note that the dataset contains both model inputs
        and labels.
        """
        data = self.get_inputs_and_labels()
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(
                len(dataset),
                reshuffle_each_iteration=True
            )

        return dataset

    def save(self, save_dir: str):
        """
        Save data of all agents.

        :param save_dir: The directory to save agent data.
        """
        save_dict = {}
        for index, agent in enumerate(self.agents):
            save_dict[str(index)] = agent.zip_data()

        np.savez(save_dir, **save_dict)

    def load(self, path: Union[str, list[Agent]]):
        """
        Load agents' data from the saved file.

        :param path: The file path of the saved data.
        """
        if not type(path) in [str, list]:
            raise ValueError(path)

        if type(path) == list:
            self.agents = path
            return

        saved: dict = np.load(path, allow_pickle=True)
        if not len(saved):
            self.log(f'Please delete file `{path}` and re-run the program.',
                     level='error', raiseError=FileNotFoundError)

        if (v := saved['0'].tolist()['__version__']) < (v1 := Agent.__version__):
            self.log((f'Saved agent managers\' version is {v}, ' +
                      f'which is lower than current {v1}. Please delete' +
                      ' them and re-run this program, or there could' +
                      ' happen something wrong.'),
                     level='error')

        self.agents = [Agent().load_data(v.tolist()) for v in saved.values()]

    def init_map_managers(self, map_type: str, base_path: str):
        agents = self.agents
        self.t_manager = TrajMapManager(self, agents,
                                        self._get_obs_trajs(),
                                        map_type, base_path)
        self.s_manager = SocialMapManager(self, agents,
                                          self.t_manager,
                                          map_type, base_path)

    def load_maps(self):
        for agent, t_map, s_map in zip(
                self.agents,
                self.t_manager.load(POOLING_BEFORE_SAVING),
                self.s_manager.load(POOLING_BEFORE_SAVING)):

            agent.set_map(0.5*t_map + 0.5*s_map)

    def _get_obs_trajs(self) -> np.ndarray:
        return np.array([a.traj for a in self.agents])

    def _get(self, type_name: str) -> tf.Tensor:
        """
        Get model inputs or labels from a list of `Agent`-like objects.

        :param type_name: Types of all inputs, accept all type names \
            in `INPUT_TYPES`.
        :return inputs: A tensor of stacked inputs.
        """
        t = type_name
        if t == INPUT_TYPES.OBSERVED_TRAJ:
            return _get_obs_traj(self.agents)

        elif t == INPUT_TYPES.MAP:
            return _get_context_map(self.agents)

        elif t == INPUT_TYPES.DESTINATION_TRAJ:
            return _get_dest_traj(self.agents)

        elif t == INPUT_TYPES.GROUNDTRUTH_TRAJ:
            return _get_gt_traj(self.agents)

        elif t == INPUT_TYPES.GROUNDTRUTH_SPECTRUM:
            if t not in self.t_layers.keys():
                t_type, _ = get_transform_layers(self.args.T)
                self.t_layers[t] = t_type(
                    (self.args.pred_frames, self.args.dim))

            t_layer = self.t_layers[t]
            return t_layer(_get_gt_traj(self.agents, text='groundtruth spectrums'))

        elif t == INPUT_TYPES.ALL_SPECTRUM:
            if t not in self.t_layers.keys():
                t_type, _ = get_transform_layers(self.args.T)
                steps = self.args.obs_frames + self.args.pred_frames
                self.t_layers[t] = t_type((steps, self.args.dim))

            trajs = []
            for agent in tqdm(self.agents, 'Prepare trajectory spectrums (all)...'):
                trajs.append(np.concatenate(
                    [agent.traj, agent.groundtruth], axis=-2))

            t_layer = self.t_layers[t]
            return t_layer(tf.cast(trajs, tf.float32))

        else:
            raise ValueError(type_name)

    def print_info(self, **kwargs):
        pass


def _get_obs_traj(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get observed trajectories from agents.

    :param input_agents: A list of input agents, type = `list[Agent]`.
    :return inputs: A Tensor of observed trajectories.
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare trajectories...'):
        inputs.append(agent.traj)
    return tf.cast(inputs, tf.float32)


def _get_gt_traj(input_agents: list[Agent],
                 destination=False,
                 text='groundtruth') -> tf.Tensor:
    """
    Get groundtruth trajectories from agents.

    :param input_agents: A list of input agents, type = `list[Agent]`.
    :return inputs: A Tensor of gt trajectories.
    """
    inputs = []
    for agent in tqdm(input_agents, f'Prepare {text}...'):
        if destination:
            inputs.append(np.expand_dims(agent.groundtruth[-1], 0))
        else:
            inputs.append(agent.groundtruth)

    return tf.cast(inputs, tf.float32)


def _get_dest_traj(input_agents: list[Agent]) -> tf.Tensor:
    return _get_gt_traj(input_agents, destination=True, text='destinations')


def _get_context_map(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get the context map from agents.

    :param input_agents: A list of input agents, type = `list[Agent]`.
    :return inputs: A Tensor of maps.
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare maps...'):
        inputs.append(agent.Map)
    return tf.cast(inputs, tf.float32)
