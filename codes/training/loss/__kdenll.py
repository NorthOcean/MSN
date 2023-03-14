"""
@Author: Conghao Wong
@Date: 2022-10-19 09:07:47
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:01:55
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf
from scipy.stats import gaussian_kde


def KDENLL_2D(pred: tf.Tensor,
           GT: tf.Tensor,
           coe: float = 1.0) -> tf.Tensor:
    """
    Calculate KDE NLL on 2D predictions.
    The calculation pipeline is the same as the KDENLL mentioned in
    `trajectron` and `trajectron++`.
    However, this metric is a bit sensitive to the value `a_min`.

    :param pred: The predicted trajectories with shape = \
        `(batch, K, steps, 2)` or `(batch, steps, 2)`.
    :param GT: Ground truth future trajectory, shape = `(batch, steps, 2)`.
    """
    # reshape to (batch, K, steps, dim)
    if pred.ndim == 3:
        pred = pred[: tf.newaxis, :, :]

    # transpose to (batch, steps, dim, K)
    pred = tf.transpose(pred, [0, 2, 3, 1])

    # reshape to (batch, steps, dim, 1)
    # GT = GT[..., tf.newaxis]

    batch_size, steps = pred.shape[:2]

    # calculate KDE nll
    # now calculating on CPU with numpy (scipy)
    pred = pred.numpy()
    GT = GT.numpy()

    results = []
    for batch in range(batch_size):
        for step in range(steps):
            kde = gaussian_kde(dataset=pred[batch, step])
            pdf = kde.logpdf(GT[batch, step])   
            pdf = np.clip(pdf, a_min=-20, a_max=None)[0]
            results.append(pdf)

    results = tf.cast(results, tf.float32)
    return -1.0 * tf.reduce_mean(results)

