"""
@Author: Conghao Wong
@Date: 2022-06-21 10:44:39
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:18:51
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from .__agent import Agent


class Trajectory():
    """
    Entire Trajectory
    -----------------
    Manage one agent's entire trajectory in datasets.

    Properties
    ----------
    ```python
    >>> self.id
    >>> self.traj
    >>> self.neighbors
    >>> self.frames
    >>> self.start_frame
    >>> self.end_frame
    ```
    """

    def __init__(self, agent_id: str,
                 agent_type: str,
                 trajectory: np.ndarray,
                 neighbors: list[list[int]],
                 frames: list[int],
                 init_position: float,
                 dimension: int):
        """
        init

        :param agent_index: ID of the trajectory.
        :param agent_type: The type of the agent.
        :param neighbors: A list of lists that contain agents' ids \
            who appear in each frame. \
            index are frame indexes.
        :param trajectory: The target trajectory, \
            shape = `(all_frames, 2)`.
        :param frames: A list of frame ids, \
            shape = `(all_frames)`.
        :param init_position: The default position that indicates \
            the agent has gone out of the scene.
        """

        self._id = agent_id
        self._type = agent_type
        self._traj = trajectory
        self._neighbors = neighbors
        self._frames = frames

        self.dim = dimension

        base = self.traj.T[0]
        diff = base[:-1] - base[1:]

        appear = np.where(diff > init_position/2)[0]
        # disappear in the next step
        disappear = np.where(diff < -init_position/2)[0]

        self._start_frame = appear[0] + 1 if len(appear) else 0
        self._end_frame = disappear[0] + 1 if len(disappear) else len(base)

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @property
    def traj(self):
        """
        Trajectory, shape = `(frames, 2)`
        """
        return self._traj

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def frames(self):
        """
        frame id that the trajectory appears.
        """
        return self._frames

    @property
    def start_frame(self):
        """
        index of the first observed frame
        """
        return self._start_frame

    @property
    def end_frame(self):
        """
        index of the last observed frame
        """
        return self._end_frame

    def sample(self, start_frame, obs_frame, end_frame,
               matrix,
               frame_step=1,
               max_neighbor=15,
               add_noise=False) -> Agent:
        """
        Sample training data from the trajectory.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indexes of frames, not their ids.
        """
        neighbors = np.array(self.neighbors[obs_frame - frame_step])

        if len(neighbors) > max_neighbor + 1:
            nei_pos = matrix[obs_frame - frame_step, list(neighbors), :]
            tar_pos = self.traj[obs_frame - frame_step, np.newaxis, :]
            dis = calculate_length(nei_pos - tar_pos)
            neighbors = neighbors[np.argsort(dis)[1:max_neighbor+1]]

        nei_traj = matrix[start_frame:obs_frame:frame_step, list(neighbors), :]
        nei_traj = np.transpose(nei_traj, [1, 0, 2])
        tar_traj = self.traj[start_frame:end_frame:frame_step, :]

        return Agent().init_data(
            id=self.id,
            type=self.type,
            target_traj=tar_traj,
            neighbors_traj=nei_traj,
            frames=self.frames[start_frame:end_frame:frame_step],
            start_frame=start_frame,
            obs_frame=obs_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            add_noise=add_noise
        )


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)
