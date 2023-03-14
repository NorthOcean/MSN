"""
@Author: Conghao Wong
@Date: 2022-10-20 20:09:14
@LastEditors: Beihao Xia
@LastEditTime: 2023-03-03 16:15:49
@Description: file content
@Github: https://northocean.github.io
@Copyright 2023 Beihao Xia, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import transformer
from codes.utils import POOLING_BEFORE_SAVING

from ..__args import HandlerArgs
from .__baseHandler import BaseHandlerModel, BaseHandlerStructure


class MSNBetaModel(BaseHandlerModel):

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, points, asHandler,
                         key_points, structure, *args, **kwargs)

        self.set_preprocess(move=0)

        # Force args
        self.args._set('key_points', '11')
        self.args._set('T', 'none')

        # Layers
        # context feature
        if not POOLING_BEFORE_SAVING:
            self.average_pooling = tf.keras.layers.AveragePooling2D([5, 5],
                                                                input_shape=[100, 100, 1])

        self.flatten = tf.keras.layers.Flatten()
        self.context_dense1 = tf.keras.layers.Dense((self.args.obs_frames+1) * 64,
                                                    activation=tf.nn.tanh)

        # traj embedding
        self.pos_embedding = tf.keras.layers.Dense(64, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate()

        # trajectory transformer
        self.transformer = transformer.Transformer(num_layers=4,
                                                   d_model=128,
                                                   num_heads=8,
                                                   dff=512,
                                                   input_vocab_size=None,
                                                   target_vocab_size=2,
                                                   pe_input=Args.obs_frames + 1,
                                                   pe_target=Args.pred_frames)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None):

        positions_ = inputs[0]
        maps = inputs[1]
        destinations = keypoints

        # concat positions and destinations
        positions = tf.concat([positions_, destinations], axis=1)

        # traj embedding, shape == (batch, obs+1, 64)
        positions_embedding = self.pos_embedding(positions)

        # context feature, shape == (batch, obs+1, 64)
        if not POOLING_BEFORE_SAVING:
            average_pooling = self.average_pooling(maps[:, :, :, tf.newaxis])
        else:
            average_pooling = maps

        flatten = self.flatten(average_pooling)
        context_feature = self.context_dense1(flatten)
        context_feature = tf.reshape(context_feature,
                                     [-1, self.args.obs_frames+1, 64])

        # concat, shape == (batch, obs+1, 128)
        concat_feature = self.concat([positions_embedding, context_feature])

        t_inputs = concat_feature
        t_outputs = linear_prediction(positions[:, -2:, :],
                                      self.args.pred_frames,
                                      return_zeros=False)
        # return t_outputs
        predictions, _ = self.transformer.call(t_inputs, t_outputs)

        return predictions


class MSNBeta(BaseHandlerStructure):

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)
        self.set_model_type(MSNBetaModel)


def linear_prediction(end_points: tf.Tensor, number, return_zeros=None):
    """
    Linear prediction from start points (not contain) to end points.

    :param end_points: start points and end points, shape == (batch, 2, 2)
    :param number: number of prediction points, DO NOT contain start point
    """
    if return_zeros:
        return tf.pad(end_points[:, -1:, :], [0, 0], [number-1, 0], [0, 0])

    start = end_points[:, :1, :]
    end = end_points[:, -1:, :]

    r = []
    for n in range(1, number):
        p = n / number
        r_c = (end - start) * p + start  # shape = (batch, 1, 2)
        r.append(r_c)

    r.append(end)
    return tf.concat(r, axis=1)
