"""
@Author: Conghao Wong
@Date: 2022-09-01 10:40:50
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:15:00
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...utils import SCALE_THRESHOLD
from .__base import BaseProcessLayer


class Scale(BaseProcessLayer):
    """
    Scaling length of trajectories' direction vector into 1.
    The reference point when scaling is the `last` observation point.
    """

    def __init__(self, anntype: str, ref: int = -1,
                 *args, **kwargs):
        """
        `ref` is the index of reference point when scaling.
        For example, when `ref == -1`, it will take the last
        observation point as the reference point.
        The reference point will not be changed after scaling.
        When `ref == 'autoref'`, it will take the last observation
        point as the refpoint when running preprocess, and take the
        first predicted point as the refpoint when running postprocess.
        """

        if ref == 'autoref':
            self.auto_ref = True
        else:
            self.auto_ref = False
            ref = int(ref)

        super().__init__(anntype, ref, *args, **kwargs)

    def update_paras(self, trajs: tf.Tensor) -> None:
        steps = trajs.shape[-2]
        vectors = (tf.gather(trajs, steps-1, axis=-2) -
                   tf.gather(trajs, 0, axis=-2))

        scales = []
        ref_points = []
        for [x, y] in self.order:
            vector = tf.gather(vectors, [x, y], axis=-1)
            scale = tf.linalg.norm(vector, axis=-1)
            scale = tf.maximum(SCALE_THRESHOLD, scale)

            # reshape into (batch, 1, 1)
            while scale.ndim < 3:
                scale = tf.expand_dims(scale, -1)
            scales.append(scale)

            # ref points: the `ref`-th points of observations
            if not self.auto_ref:
                _trajs = tf.gather(trajs, [x, y], axis=-1)
                _ref = tf.math.mod(self.ref, steps)
                _ref_point = tf.gather(_trajs, [_ref], axis=-2)
                ref_points.append(_ref_point)

        self.paras = (scales, ref_points)

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        """
        Scaling length of trajectories' direction vector into 1.
        The reference point when scaling is the `last` observation point.

        :param trajs: Input trajectories, shape = `[(batch,) obs, 2]`.
        :return trajs_scaled: Scaled trajectories.
        """
        if use_new_paras:
            self.update_paras(trajs)

        (scales, ref_points) = self.paras
        trajs_scaled = self.scale(trajs, scales,
                                  inverse=False,
                                  ref_points=ref_points,
                                  autoref_index=trajs.shape[-2]-1)
        return trajs_scaled

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        """
        Scale trajectories back to their original.
        The reference point is the `first` prediction point.

        :param trajs: Trajectories, shape = `[(batch,) (K,) pred, 2]`.
        :param para_dict: A dict of used parameters, contains `scale: tf.Tensor`.
        :return trajs_scaled: Scaled trajectories.
        """
        (scales, ref_points) = self.paras
        trajs_scaled = self.scale(trajs, scales,
                                  inverse=True,
                                  ref_points=ref_points,
                                  autoref_index=0)
        return trajs_scaled

    def scale(self, trajs: tf.Tensor,
              scales: list[tf.Tensor],
              inverse=False,
              ref_points: list[tf.Tensor] = None,
              autoref_index: int = None):

        ndim = trajs.ndim

        trajs_scaled = []
        for index, [x, y] in enumerate(self.order):
            traj = tf.gather(trajs, [x, y], axis=-1)

            # get scale
            scale = scales[index]

            if inverse:
                scale = 1.0 / scale

            while scale.ndim < ndim:
                scale = tf.expand_dims(scale, -1)

            # get reference point
            if self.auto_ref:
                ref_point = tf.gather(traj, [autoref_index], axis=-2)
            else:
                ref_point = ref_points[index]

            while ref_point.ndim < ndim:
                ref_point = tf.expand_dims(ref_point, axis=-3)

            # start scaling
            _trajs_scaled = (traj - ref_point) / scale + ref_point
            trajs_scaled.append(_trajs_scaled)

        trajs_scaled = tf.concat(trajs_scaled, axis=-1)
        return trajs_scaled
