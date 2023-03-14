"""
@Author: Conghao Wong
@Date: 2022-10-12 09:06:50
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:22:26
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


def ADE_2D(pred: tf.Tensor,
           GT: tf.Tensor,
           coe: float = 1.0) -> tf.Tensor:
    """
    Calculate `ADE` or `minADE`.

    :param pred: The predicted trajectories with shape = \
        `(batch, K, steps, 2)` or `(batch, steps, 2)`.
    :param GT: Ground truth future trajectory, shape = `(batch, steps, 2)`.

    :return loss_ade:
        Return `ADE` when input_shape = [batch, pred_frames, 2];
        Return `minADE` when input_shape = [batch, K, pred_frames, 2].
    """
    if pred.ndim == 3:
        pred = pred[:, tf.newaxis, :, :]

    all_ade = tf.reduce_mean(
        tf.linalg.norm(
            pred - GT[:, tf.newaxis, :, :],
            ord=2, axis=-1
        ), axis=-1)
    best_ade = tf.reduce_min(all_ade, axis=1)
    return coe * tf.reduce_mean(best_ade)


def FDE_2D(pred: tf.Tensor,
           GT: tf.Tensor,
           coe: float = 1.0) -> tf.Tensor:
    """
    Calculate `FDE` or `minFDE`.

    :param pred: The predicted trajectories with shape = \
        `(batch, K, steps, 2)` or `(batch, steps, 2)`.
    :param GT: Ground truth future trajectory, shape = `(batch, steps, 2)`.

    :return fde:
        Return `FDE` when input_shape = [batch, pred_frames, 2];
        Return `minFDE` when input_shape = [batch, K, pred_frames, 2].
    """
    return ADE_2D([pred[..., -1:, :]], GT[..., -1:, :], coe=coe)


def diff(pred: tf.Tensor,
         GT: tf.Tensor,
         ordd: int = 2,
         coe: float = 1.0) -> list[tf.Tensor]:
    """
    loss_functions with difference limit.

    :param pred: The predicted trajectories with shape = \
        `(batch, K, steps, 2)` or `(batch, steps, 2)`.
    :param GT: Ground truth future trajectory, shape = `(batch, steps, 2)`.

    :return loss: a list of Tensor, `len(loss) = ord + 1`.
    """
    pred_diff = __difference(pred, ordd=ordd)
    GT_diff = __difference(GT, ordd=ordd)

    loss = []
    for pred_, gt_ in zip(pred_diff, GT_diff):
        loss.append(ADE_2D([pred_], gt_, coe=coe))

    return loss


def __difference(trajs: tf.Tensor, direction='back', ordd=1) -> list[tf.Tensor]:
    """
    :param trajs: Trajectories, shape = `[(K,) batch, pred, 2]`.
    :param direction: Direction of the difference, can be `'back'` or `'forward'`.
    :param ord: Repeat times.

    :return result: results list, `len(results) = ord + 1`
    """
    outputs = [trajs]
    for repeat in range(ordd):
        outputs_current = \
            outputs[-1][:, :, 1:, :] - outputs[-1][:, :, :-1, :] if len(trajs.shape) == 4 else \
            outputs[-1][:, 1:, :] - outputs[-1][:, :-1, :]
        outputs.append(outputs_current)
    return outputs
