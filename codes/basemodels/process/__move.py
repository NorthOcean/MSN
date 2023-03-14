"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:40
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:14:08
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from .__base import BaseProcessLayer


class Move(BaseProcessLayer):
    """
    Move a specific point to (0, 0) according to the reference time step.
    The default reference time step is the last observation step.
    """

    def __init__(self, anntype: str = None, ref: int = -1,
                 *args, **kwargs):

        super().__init__(anntype, ref, *args, **kwargs)

    def update_paras(self, trajs: tf.Tensor) -> None:
        ref = tf.math.mod(self.ref, trajs.shape[-2])
        ref_point = tf.gather(trajs, [ref], axis=-2)
        self.paras = ref_point

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        """
        Move a specific point to (0, 0) according to the reference time step.

        :param trajs: Trajectories, shape = `[(batch,) (K,) obs, dim]`.
        :return trajs_moved: Moved trajectories.
        """
        if use_new_paras:
            self.update_paras(trajs)

        ref_points = self.paras
        trajs_moved = self.move(trajs, ref_points)
        return trajs_moved

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        """
        Move trajectories back to their original positions.

        :param trajs: Trajectories moved to (0, 0) with a reference point, \
            shape = `[(batch,) (K,) pred, dim]`.
        :return trajs_moved: Moved trajectories.
        """
        ref_point = self.paras
        trajs_moved = self.move(trajs, ref_point, inverse=True)
        return trajs_moved

    def move(self, trajs: tf.Tensor,
             ref_points: tf.Tensor,
             inverse=False):

        ndim = trajs.ndim
        while ref_points.ndim < ndim:
            ref_points = tf.expand_dims(ref_points, axis=-3)

        if inverse:
            ref_points = -1.0 * ref_points

        # start moving
        trajs_moved = trajs - ref_points
        return trajs_moved
