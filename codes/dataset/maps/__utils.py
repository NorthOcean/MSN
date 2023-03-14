"""
@Author: Conghao Wong
@Date: 2022-11-10 09:38:32
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:17:39
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np

MASK = cv2.imread('./figures/mask_circle.png')[:, :, 0]/50
MASKS = {}

DECAY_P = np.array([[0.0, 0.7, 1.0], [1.0, 1.0, 0.5]])
DECAYS = {}


def add(target_map: np.ndarray,
        grid_trajs: np.ndarray,
        amplitude: np.ndarray,
        radius: int,
        add_mask: np.ndarray = MASK,
        max_limit=False):

    if len(grid_trajs.shape) == 2:
        grid_trajs = grid_trajs[np.newaxis, :, :]

    n_traj, traj_len, dim = grid_trajs.shape[:3]

    if not traj_len in DECAYS.keys():
        DECAYS[traj_len] = np.interp(np.linspace(0, 1, traj_len),
                                     DECAY_P[0],
                                     DECAY_P[1])

    if not radius in MASKS.keys():
        MASKS[radius] = cv2.resize(add_mask, (radius*2+1, radius*2+1))

    a = np.array(amplitude)[:, np.newaxis] * \
        DECAYS[traj_len] * \
        np.ones([n_traj, traj_len], dtype=np.int32)

    points = np.reshape(grid_trajs, [-1, dim])
    amps = np.reshape(a, [-1])

    target_map = target_map.copy()
    target_map = add_traj(target_map,
                          points, amps, radius,
                          MASKS[radius],
                          max_limit=max_limit)

    return target_map


def add_traj(source_map: np.ndarray,
             traj: np.ndarray,
             amplitude: float,
             radius: int,
             add_mask: np.ndarray,
             max_limit=False):

    new_map = np.zeros_like(source_map)
    for pos, a in zip(traj, amplitude):
        if (pos[0]-radius >= 0
            and pos[1]-radius >= 0
            and pos[0]+radius+1 < new_map.shape[0]
                and pos[1]+radius+1 < new_map.shape[1]):

            new_map[pos[0]-radius:pos[0]+radius+1,
                    pos[1]-radius:pos[1]+radius+1] = \
                a * add_mask + new_map[pos[0]-radius:pos[0]+radius+1,
                                       pos[1]-radius:pos[1]+radius+1]

    if max_limit:
        new_map = np.sign(new_map)

    return new_map + source_map


def cut(target_map: np.ndarray,
        centers: np.ndarray,
        half_size: int) -> np.ndarray:
    """
    Cut several local maps from the target map.

    :param target_map: The target map, shape = (a, b).
    :param centers: Center positions (in grids), shape = (batch, 2).
    :param half_size: The half-size of the cut map.
    """
    a, b = target_map.shape[-2:]
    centers = centers.astype(np.int32)

    # reshape to (batch, 2)
    if centers.ndim == 1:
        centers = centers[np.newaxis, :]

    centers = np.maximum(centers, half_size)
    centers = np.array([np.minimum(centers[:, 0], a - half_size),
                        np.minimum(centers[:, 1], b - half_size)]).T

    cuts = []
    for c in centers:
        cuts.append(target_map[c[0] - half_size: c[0] + half_size,
                               c[1] - half_size: c[1] + half_size])

    return np.array(cuts)
