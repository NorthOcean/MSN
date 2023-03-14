"""
@Author: Conghao Wong
@Date: 2022-11-29 09:39:09
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-29 09:48:21
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import interpolation

from ...__args import HandlerArgs
from .__baseInterpHandler import _BaseInterpHandlerModel


class LinearHandlerModel(_BaseInterpHandlerModel):

    def __init__(self, Args: HandlerArgs, structure=None, *args, **kwargs):
        super().__init__(Args, structure, *args, **kwargs)

        self.interp_layer = interpolation.LinearPositionInterpolation()

    def interp(self, index: tf.Tensor, value: tf.Tensor, obs_traj: tf.Tensor) -> tf.Tensor:
        # Concat keypoints with the last observed point
        index = tf.concat([[-1], index], axis=0)
        obs_position = obs_traj[..., -1:, :]
        value = tf.concat([obs_position, value], axis=-2)

        # Calculate linear interpolation -> (batch, pred, 2)
        return self.interp_layer.call(index, value)


class LinearSpeedHandlerModel(_BaseInterpHandlerModel):

    def __init__(self, Args: HandlerArgs, structure=None, *args, **kwargs):
        super().__init__(Args, structure, *args, **kwargs)

        self.interp_layer = interpolation.LinearSpeedInterpolation()

    def interp(self, index: tf.Tensor, value: tf.Tensor, obs_traj: tf.Tensor) -> tf.Tensor:
        # Concat keypoints with the last observed point
        index = tf.concat([[-1], index], axis=0)
        obs_position = obs_traj[..., -1:, :]
        value = tf.concat([obs_position, value], axis=-2)

        # Calculate linear interpolation -> (batch, pred, 2)
        v0 = obs_traj[..., -1:, :] - obs_traj[..., -2:-1, :]
        return self.interp_layer.call(index, value, init_speed=v0)


class LinearAccHandlerModel(_BaseInterpHandlerModel):

    def __init__(self, Args: HandlerArgs, structure=None, *args, **kwargs):
        super().__init__(Args, structure, *args, **kwargs)

        self.interp_layer = interpolation.LinearAccInterpolation()

    def interp(self, index: tf.Tensor, value: tf.Tensor, obs_traj: tf.Tensor) -> tf.Tensor:
        # Concat keypoints with the last observed point
        index = tf.concat([[-1], index], axis=0)
        obs_position = obs_traj[..., -1:, :]
        value = tf.concat([obs_position, value], axis=-2)

        # Calculate linear interpolation -> (batch, pred, 2)
        v_last = obs_traj[..., -1:, :] - obs_traj[..., -2:-1, :]
        v_second_to_last = obs_traj[..., -2:-1, :] - obs_traj[..., -3:-2, :]
        return self.interp_layer.call(index, value,
                                      init_speed=v_last,
                                      init_acc=v_last - v_second_to_last)
