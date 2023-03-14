"""
@Author: Conghao Wong
@Date: 2022-11-28 20:58:04
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-29 09:15:09
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class LinearPositionInterpolation(tf.keras.layers.Layer):
    """
    Piecewise linear interpolation.
    For a trajectory `y(t)`, this interpolation method considers
    `y(t) = y0 + delta_y * t`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, index: tf.Tensor, value: tf.Tensor):
        """
        Piecewise linear interpolation.
        The results do not contain the start point.

        :param index: Indexes of keypoints, shape = `(n)`.
        :param value: Keypoints values, shape = `(..., n, dim)`.

        :return yp: Interpolations, shape = `(..., m, dim)`, where
        `m = index[-1] - index[0]`.
        """

        x = index
        y = value

        results = []
        for output_index in range(len(x) - 1):
            x_start = x[output_index]
            x_end = x[output_index+1]
            n = x_end - x_start

            # shape = (..., 1, dim)
            y_start = y[..., output_index:output_index+1, :]
            y_end = y[..., output_index+1:output_index+2, :]
            delta_y = y_end - y_start

            for x_p in tf.range(x_start+1, x_end+1):
                results.append(delta_y * (x_p - x_start) / n + y_start)

        # shape = (..., m, dim)
        return tf.concat(results, axis=-2)
