"""
@Author: Conghao Wong
@Date: 2022-11-28 21:16:28
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-29 09:23:34
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class NewtonInterpolation(tf.keras.layers.Layer):
    """
    Newton interpolation layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, index: tf.Tensor, value: tf.Tensor,
             ord: int, interval: float = 1):
        """
        Newton interpolation.
        The results do not contain the start point.

        :param index: Indexes of keypoints, shape = `(n)`.
        :param value: Keypoints values, shape = `(..., n, dim)`.
        :param ord: The order to calculate interpolations.
        :param interval: The interpolation interval.

        :return yp: Interpolations, shape = `(..., m, dim)`, where
        `m = index[-1] - index[0]`.
        """

        x = index
        y = value

        diff_quotient = [y]
        for i in range(ord):
            last_res = diff_quotient[i]
            diff_y = last_res[..., :-1, :] - last_res[..., 1:, :]
            diff_x = (x[:-1-i] - x[1+i:])[:, tf.newaxis]
            diff_quotient.append(diff_y/diff_x)

        # shape = (m)
        x_p = tf.range(x[0]+1, x[-1]+1, delta=interval)

        # shape = (ord+1, ..., dim)
        coe = tf.stack([d[..., 0, :] for d in diff_quotient])

        # shape = (m, n)
        xs = x_p[:, tf.newaxis] - x

        xs_prod = [tf.ones_like(x_p)[:, tf.newaxis]]
        for i in range(ord):
            xs_prod.append(tf.reduce_prod(xs[:, :i+1], axis=-1, keepdims=True))

        # shape = (m, ord+1)
        xs_prod = tf.concat(xs_prod, axis=-1)

        res = tf.tensordot(xs_prod, coe, axes=1)
        return tf.stack(tf.unstack(res, axis=0), axis=-2)
