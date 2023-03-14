"""
@Author: Conghao Wong
@Date: 2022-11-10 09:27:30
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:16:39
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ...base import BaseManager
from ...utils import AVOID_SIZE, INTEREST_SIZE
from ..trajectories import Agent
from .__base import BaseMapManager
from .__trajMap import TrajMapManager
from .__utils import add, cut


class SocialMapManager(BaseMapManager):
    """
    Social Map Manager
    ---
    The social map is a map that builds from all neighbor agents'
    observed trajectories. It indicates their potential social
    interactions in the prediction period. The value of the social map
    is in the range `[0, 1]`. A higher value indicates that
    the area may not suitable for walking under different kinds of
    social interactions.
    """

    def __init__(self, manager: BaseManager,
                 agents: list[Agent],
                 init_manager: TrajMapManager,
                 map_type: str,
                 base_path: str,
                 name: str = 'Social Map Manager'):

        super().__init__(manager, map_type, base_path, name)

        self.MAP_NAME = 'Social Map'
        self.FILE = 'socialMap.npy'
        self.FILE_WITH_POOLING = 'socialMap_pooling.npy'
        self.CONFIG_FILE = 'socialMap_configs.npy'

        if self.map_exsits():
            pass
        else:
            self.init(init_manager)
            self.build_all(agents)

    def init(self, init_manager: TrajMapManager):
        self.void_map, self.W, self.b = [init_manager.void_map,
                                         init_manager.W,
                                         init_manager.b]

    def build(self, agent: Agent,
              source: np.ndarray = None,
              regulation=True,
              max_neighbor=15,
              *args, **kwargs) -> np.ndarray:
        """
        Build a social map for a specific agent.
        TODO: Social maps for M-dimensional trajectories

        :param agent: The target `Agent` object to calculate the map.
        :param source: The source map, default are zeros.
        :param regulation: Controls if scale the map into [0, 1].
        :param max_neighbor: The maximum number of neighbors to calculate
            the social map. Set it to a smaller value to speed up the building
            on datasets that contain more agents.
        """

        # build the global social map
        if type(source) == type(None):
            source = self.void_map

        source = source.copy()

        # Destination
        source = add(target_map=source,
                     grid_trajs=self.real2grid(agent.pred_linear),
                     amplitude=[-2],
                     radius=INTEREST_SIZE)

        # Interplay
        traj_neighbors = agent.pred_linear_neighbor
        amp_neighbors = []

        vec_target = agent.pred_linear[-1] - agent.pred_linear[0]
        len_target = calculate_length(vec_target)

        vec_neighbor = traj_neighbors[:, -1] - traj_neighbors[:, 0]

        if len_target >= 0.05:
            cosine = activation(
                calculate_cosine(vec_target[np.newaxis, :], vec_neighbor),
                a=1.0,
                b=0.2)
            velocity = (calculate_length(vec_neighbor) /
                        calculate_length(vec_target[np.newaxis, :]))

        else:
            cosine = np.ones(len(traj_neighbors))
            velocity = 2

        amp_neighbors = - cosine * velocity

        amps = amp_neighbors.tolist()
        trajs = traj_neighbors.tolist()

        if len(trajs) > max_neighbor + 1:
            trajs = np.array(trajs)
            dis = calculate_length(trajs[:1, 0, :] - trajs[:, 0, :])
            index = np.argsort(dis)
            trajs = trajs[index[:max_neighbor+1]]

        source = add(target_map=source,
                     grid_trajs=self.real2grid(trajs),
                     amplitude=amps,
                     radius=AVOID_SIZE)

        if regulation:
            if (np.max(source) - np.min(source)) <= 0.01:
                source = 0.5 * np.ones_like(source)
            else:
                source = (source - np.min(source)) / \
                    (np.max(source) - np.min(source))

        # Get the local social map from the global map
        # center point: the last observed point
        center_real = agent.traj[-1:, :]
        center_pixel = self.real2grid(center_real)
        local_map = cut(source, center_pixel, self.HALF_SIZE)[0]

        return local_map


def calculate_cosine(vec1: np.ndarray,
                     vec2: np.ndarray):

    length1 = np.linalg.norm(vec1, axis=-1)
    length2 = np.linalg.norm(vec2, axis=-1)

    return (np.sum(vec1 * vec2, axis=-1) + 0.0001) / ((length1 * length2) + 0.0001)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def activation(x: np.ndarray, a=1, b=1):
    return np.less_equal(x, 0) * a * x + np.greater(x, 0) * b * x
