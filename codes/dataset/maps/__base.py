"""
@Author: Conghao Wong
@Date: 2022-11-10 09:27:06
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-22 09:24:10
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
import tensorflow as tf

from ...base import BaseManager, SecondaryBar
from ...utils import MAP_HALF_SIZE, dir_check
from ..trajectories import Agent

POOLING_LAYER = tf.keras.layers.MaxPool2D([5, 5], data_format='channels_last')


class BaseMapManager(BaseManager):
    """
    BaseMapManager
    ---
    A basic class for managing context maps that could
    describe interactions among agents and scenes.
    The `BaseMapManager` object is managed by `AgentManager`.

    Usage
    ---
    Subclass this class to manage the new map. The `build`
    method should be rewritten in the new class.
    """

    def __init__(self, manager: BaseManager,
                 map_type: str,
                 base_path: str,
                 name='Interaction Maps Manager'):

        super().__init__(manager=manager, name=name)

        self.map_type = map_type
        self.dir = dir_check(base_path)
        self.path = os.path.join(base_path, '{}')

        self.MAP_NAME: str = None
        self.FILE: str = None
        self.FILE_WITH_POOLING: str = None
        self.CONFIG_FILE: str = None

        self.HALF_SIZE = MAP_HALF_SIZE

        self.void_map: np.ndarray = None
        self.W: np.ndarray = None
        self.b: np.ndarray = None

    def real2grid(self, traj: np.ndarray) -> np.ndarray:
        if not type(traj) == np.ndarray:
            traj = np.array(traj)

        grid = ((traj - self.b) * self.W).astype(np.int32)
        return grid

    def pooling2D(self, maps: np.ndarray):
        """
        Apply MaxPooling on a batch of maps.

        :param maps: Maps, shape = (batch, a, b).
        """
        maps = maps[..., np.newaxis]
        return POOLING_LAYER(maps).numpy()[..., 0]

    def map_exsits(self, pooling=False):
        if not pooling:
            if os.path.exists(self.path.format(self.FILE)):
                return True
            else:
                return False
        else:
            if os.path.exists(self.path.format(self.FILE_WITH_POOLING)):
                return True
            else:
                return False

    def build(self, agent: Agent,
              source: np.ndarray = None,
              *args, **kwargs):
        """
        Build a map for a specific agent.
        """
        raise NotImplementedError

    def build_all(self, agents: list[Agent],
                  source: np.ndarray = None,
                  *args, **kwargs):
        """
        Build maps for all agents and save them.
        """
        maps = []
        for agent in SecondaryBar(agents,
                                  manager=self.manager.manager,
                                  desc=f'Building {self.MAP_NAME}...'):
            maps.append(self.build(agent, source, *args, **kwargs))

        # save maps
        np.save(self.path.format(self.FILE), maps)

    def load(self, pooling=False) -> np.ndarray:
        """
        Load maps from the saved file.
        """
        maps = np.load(self.path.format(self.FILE), allow_pickle=True)

        if not pooling:
            return maps
        else:
            if not self.map_exsits(pooling=True):
                maps_pooling = self.pooling2D(np.array(maps))
                np.save(self.path.format(self.FILE_WITH_POOLING), maps_pooling)

            else:
                maps_pooling = np.load(self.path.format(self.FILE_WITH_POOLING),
                                       allow_pickle=True)

            return maps_pooling
