"""
@Author: Conghao Wong
@Date: 2022-11-28 21:08:27
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-29 09:20:32
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class LinearAccInterpolation(tf.keras.layers.Layer):
    """
    Piecewise linear interpolation on the acceleration.
    For a trajectory `y(t)`, this interpolation method considers
    the acceleration as `a(t) = a0 + delta_a * t`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, index: tf.Tensor, value: tf.Tensor,
             init_speed: tf.Tensor, init_acc: tf.Tensor):
        """
        Piecewise linear interpolation on the acceleration.
        The results do not contain the start point.

        :param index: Indexes of keypoints, shape = `(n)`.
        :param value: Keypoints values, shape = `(..., n, dim)`.
        :param init_speed: The initial speed on the last observed
        time step. It should has the shape `(..., 1, dim)`.
        :param init_acc: The initial acceleration on the last observed
        time step. It should has the shape `(..., 1, dim)`.

        :return yp: Interpolations, shape = `(..., m, dim)`, where
        `m = index[-1] - index[0]`.
        """

        x = index
        y = value

        accs = [init_acc]
        speeds = [init_speed]
        results = [y[..., 0:1, :]]

        for output_index in range(len(x) - 1):
            x_start = x[output_index]
            x_end = x[output_index+1]
            n = x_end - x_start

            # shape = (..., 1, dim)
            y_start = y[..., output_index:output_index+1, :]
            y_end = y[..., output_index+1:output_index+2, :]
            delta_y = y_end - y_start

            if not x_end - x_start > 1:
                results += [y_end]
                speeds += [y_end - y_start]
                accs += [speeds[-1] - speeds[-2]]
                continue

            v0 = speeds[-1]
            a0 = accs[-1]
            delta_a = ((delta_y - n*v0) - (n*(n+1) * a0)/2) / \
                (n*(n+1)/4 + n*(n+1)*(2*n+1)/12)

            for _ in tf.range(x_start + 1, x_end + 1):
                accs.append(accs[-1] + delta_a)
                speeds.append(speeds[-1] + accs[-1])
                results.append(results[-1] + speeds[-1])

        return tf.concat(results[1:], axis=-2)
