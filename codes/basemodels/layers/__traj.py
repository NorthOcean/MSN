"""
@Author: Conghao Wong
@Date: 2021-12-21 15:25:47
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-22 09:16:39
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...utils import POOLING_BEFORE_SAVING
from .__transformLayers import _BaseTransformLayer


class TrajEncoding(tf.keras.layers.Layer):
    """
    Encode trajectories into the traj feature
    """

    def __init__(self, units: int = 64,
                 activation=None,
                 transform_layer: _BaseTransformLayer = None,
                 channels_first=True,
                 *args, **kwargs):
        """
        Init a trajectory encoding module.

        :param units: Feature dimension.
        :param activation: Activations used in the output layer.
        :param transform_layer: Controls if encode trajectories \
            with some transform methods (like FFTs).
        :param channels_first: Controls if running computations on \
            the last dimension of the inputs.
        """

        super().__init__(*args, **kwargs)

        self.Tlayer = None
        self.channels_first = channels_first

        if transform_layer:
            self.Tlayer = transform_layer
            self.fc2 = tf.keras.layers.Dense(units, tf.nn.relu)

        self.fc1 = tf.keras.layers.Dense(units, activation)

    def call(self, trajs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Encode trajectories into the high-dimension features.

        :param trajs: Trajs, shape = `(batch, N, 2)`.
        :return features: Features, shape = `(batch, N, units)`.      
            NOTE: If the transform layer was set, it will return a feature 
            with the `shape = (batch, Tsteps, units)`.
        """
        if self.Tlayer:
            t = self.Tlayer(trajs)  # (batch, Tsteps, Tchannels)

            if not self.channels_first:
                t = tf.transpose(t, [0, 2, 1])  # (batch, Tchannels, Tsteps)

            fc2 = self.fc2(t)
            return self.fc1(fc2)

        else:
            return self.fc1(trajs)


class ContextEncoding(tf.keras.layers.Layer):
    """
    Encode context maps into the context feature
    """

    def __init__(self, output_channels: int,
                 units: int = 64,
                 activation=None,
                 *args, **kwargs):
        """
        Init a context encoding module.
        The context encoding layer finally outputs a `tf.Tensor`
        with shape `(batch_size, output_channels, units)`.

        :param output_channels: Output channels.
        :param units: Output feature dimension.
        :param activation: Activations used in the output layer.
        """

        super().__init__(*args, **kwargs)

        if not POOLING_BEFORE_SAVING:
            self.pool = tf.keras.layers.MaxPooling2D(pool_size=[5, 5],
                                                     data_format='channels_last')

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(output_channels * units, activation)
        self.reshape = tf.keras.layers.Reshape((output_channels, units))

    def call(self, context_map: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Encode context maps into context features.

        :param context_map: Maps, shape = `(batch, a, a)`.
        :return feature: Features, shape = `(batch, output_channels, units)`.
        """
        if not POOLING_BEFORE_SAVING:
            context_map = self.pool(context_map[:, :, :, tf.newaxis])

        flat = self.flatten(context_map)
        fc = self.fc(flat)
        return self.reshape(fc)
