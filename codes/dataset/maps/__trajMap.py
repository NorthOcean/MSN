"""
@Author: Conghao Wong
@Date: 2022-11-10 09:27:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 10:54:36
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import numpy as np

from ...base import BaseManager
from ...utils import (WINDOW_EXPAND_METER, WINDOW_EXPAND_PIXEL,
                      WINDOW_SIZE_METER, WINDOW_SIZE_PIXEL)
from ..trajectories import Agent
from .__base import BaseMapManager
from .__utils import add, cut


class TrajMapManager(BaseMapManager):
    """
    Trajectory Map Manager
    ---
    The trajectory map is a map that builds from all agents'
    observed trajectories. It indicates all possible walkable
    areas around the target agent. The value of the trajectory map
    is in the range `[0, 1]`. A higher value indicates that
    the area may not walkable.
    """

    def __init__(self, manager: BaseManager,
                 agents: list[Agent],
                 init_trajs: np.ndarray,
                 map_type: str,
                 base_path: str,
                 name: str = 'Trajectory Map Manager'):

        super().__init__(manager, map_type, base_path, name)

        self.MAP_NAME = 'Trajectory Map'
        self.FILE = 'trajMap.npy'
        self.FILE_WITH_POOLING = 'trajMap_pooling.npy'
        self.CONFIG_FILE = 'trajMap_configs.npy'

        self.GLOBAL_FILE = 'trajMap.png'
        self.GLOBAL_CONFIG_FILE = 'trajMap_configs.npy'

        # global trajectory map
        self.map: np.ndarray = None

        if self.map_exsits():
            self.load_configs()
        else:
            self.init(init_trajs)
            self.build_all(agents)

    def init(self, init_trajs: np.ndarray):
        path_global_map = self.path.format(self.GLOBAL_FILE)
        if os.path.exists(path_global_map):
            self.map = self.load_global()
        else:
            self.map = self.build_global(init_trajs)

    def init_global(self, init_trajs: np.ndarray):
        """
        Init the trajectory map via a list of agents.

        :param init_trajs: trajectories to init the guidance map.
            shape should be `((batch), obs, 2)`

        :return guidance_map: initialized trajectory map
        :return W: map parameter `W`
        :return b: map parameter `b`
        """

        traj = init_trajs

        # shape of `traj` should be [*, *, 2] or [*, 2]
        if len(traj.shape) == 3:
            traj = np.reshape(traj, [-1, 2])

        x_max = np.max(traj[:, 0])
        x_min = np.min(traj[:, 0])
        y_max = np.max(traj[:, 1])
        y_min = np.min(traj[:, 1])

        if self.map_type == 'pixel':
            a = WINDOW_SIZE_PIXEL
            e = WINDOW_EXPAND_PIXEL

        elif self.map_type == 'meter':
            a = WINDOW_SIZE_METER
            e = WINDOW_EXPAND_METER

        else:
            raise ValueError(self.map_type)

        guidance_map = np.zeros([int((x_max - x_min + 2 * e) * a) + 1,
                                 int((y_max - y_min + 2 * e) * a) + 1])
        W = np.array([a, a])
        b = np.array([x_min - e, y_min - e])

        return guidance_map.astype(np.float32), W, b

    def build_global(self, trajs: np.ndarray,
                     source: np.ndarray = None):

        if source is None:
            if self.void_map is None:
                self.void_map, self.W, self.b = self.init_global(trajs)

            source = self.void_map

        # build the global trajectory map
        source = source.copy()
        source = add(source,
                     self.real2grid(trajs),
                     amplitude=[1],
                     radius=7)

        source = np.minimum(source, 30)
        source = 1 - source / np.max(source)

        # save global trajectory map
        cv2.imwrite(self.path.format(self.GLOBAL_FILE), 255 * source)

        # save global map's configs
        np.save(self.path.format(self.GLOBAL_CONFIG_FILE),
                arr=dict(void_map=self.void_map,
                         W=self.W,
                         b=self.b),)

        return source

    def load_global(self):
        # load global trajectory map
        t_map = cv2.imread(self.path.format(self.GLOBAL_FILE))

        if t_map is None:
            raise FileNotFoundError

        t_map = (t_map[:, :, 0]).astype(np.float32)/255.0
        self.map = t_map
        self.load_configs()

    def load_configs(self):
        # load global map's configs
        config_path = self.path.format(self.GLOBAL_CONFIG_FILE)
        if not os.path.exists(config_path):
            self.log(f'Please delete the folder `{self.dir}` and' +
                     ' re-run this program.', 'error')
            exit()

        config_dict = np.load(config_path, allow_pickle=True).tolist()

        self.void_map = config_dict['void_map']
        self.W = config_dict['W']
        self.b = config_dict['b']

    def build(self, agent: Agent,
              source: np.ndarray = None,
              *args, **kwargs):

        # Cut the local trajectory map from the global map
        # Center point: the last observed point
        center_real = agent.traj[-1:, :]
        center_pixel = self.real2grid(center_real)
        local_map = cut(self.map, center_pixel, self.HALF_SIZE)[0]

        return local_map
