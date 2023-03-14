"""
@Author: Conghao Wong
@Date: 2022-06-22 09:35:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-12-06 10:01:45
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf

from codes import INPUT_TYPES
from codes.managers import Model, SecondaryBar, Structure

from ..__args import HandlerArgs


class BaseHandlerModel(Model):

    is_interp_handler = False

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args: HandlerArgs = Args
        self.structure: BaseHandlerStructure = structure

        # GT in the inputs is only used when training
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.MAP,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Parameters
        self.asHandler = asHandler
        self.d = feature_dim
        self.points = points
        self.key_points = key_points
        self.accept_batchK_inputs = False

        if self.asHandler or key_points != 'null':
            pi = [int(i) for i in key_points.split('_')]
            self.points_index = tf.cast(pi, tf.float32)

        # Preprocess
        preprocess = {}
        for index, operation in enumerate(['move', 'scale', 'rotate']):
            if self.args.preprocess[index] == '1':
                preprocess[operation] = 'auto'

        self.set_preprocess(**preprocess)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None):

        raise NotImplementedError

    def call_as_handler(self, inputs: list[tf.Tensor],
                        keypoints: tf.Tensor,
                        keypoints_index: tf.Tensor,
                        training=None, mask=None):
        """
        Call as the second stage handler model.
        Do NOT call this method when training.

        :param inputs: a list of trajs and context maps
        :param keypoints: predicted keypoints, shape is `(batch, K, n_k, 2)`
        :param keypoints_index: index of predicted keypoints, shape is `(n_k)`
        """

        if not self.accept_batchK_inputs:
            p_all = []
            for k in SecondaryBar(range(keypoints.shape[1]),
                                  manager=self.structure.manager,
                                  desc='Running Stage-2 Sub-Network...'):

                # Run stage-2 network on a batch of inputs
                pred = self.call(inputs=inputs,
                                 keypoints=keypoints[:, k, :, :],
                                 keypoints_index=keypoints_index)

                if type(pred) not in [list, tuple]:
                    pred = [pred]

                # A single output shape is (batch, pred, dim).
                p_all.append(pred[0])

            return tf.transpose(tf.stack(p_all), [1, 0, 2, 3])

        else:
            return self.call(inputs=inputs,
                             keypoints=keypoints,
                             keypoints_index=keypoints_index)

    def forward(self, inputs: list[tf.Tensor],
                training=None,
                *args, **kwargs):

        keypoints = [inputs[-1]]

        inputs_p = self.process(inputs, preprocess=True, training=training)
        keypoints_p = self.process(keypoints, preprocess=True,
                                   update_paras=False,
                                   training=training)

        # only when training the single model
        if not self.asHandler:
            gt_processed = keypoints_p[0]

            if self.key_points == 'null':
                index = np.arange(self.args.pred_frames-1)
                np.random.shuffle(index)
                index = tf.concat([np.sort(index[:self.points-1]),
                                   [self.args.pred_frames-1]], axis=0)
            else:
                index = tf.cast(self.points_index, tf.int32)

            points = tf.gather(gt_processed, index, axis=1)
            index = tf.cast(index, tf.float32)

            outputs = self.call(inputs_p,
                                keypoints=points,
                                keypoints_index=index,
                                training=True)

        # use as the second stage model
        else:
            outputs = self.call_as_handler(inputs_p,
                                           keypoints=keypoints_p[0],
                                           keypoints_index=self.points_index,
                                           training=None)

        outputs_p = self.process(outputs, preprocess=False, training=training)
        return outputs_p

    def print_info(self, **kwargs):
        info = {'Transform type': self.args.T,
                'Number of keypoints': self.args.points}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseHandlerStructure(Structure):

    model_type = None

    def __init__(self, terminal_args: list[str],
                 manager: Structure = None,
                 is_temporary=False):

        name = 'Train Manager'
        if is_temporary:
            name += ' (Second-Stage Sub-network)'

        super().__init__(args=HandlerArgs(terminal_args, is_temporary),
                         manager=manager,
                         name=name)

        self.args: HandlerArgs
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.loss.set({self.loss.l2: 1.0})

        if self.args.key_points == 'null':
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0})
        else:
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0,
                              self.metrics.avgKey: 0.0})

    def set_model_type(self, new_type: type[BaseHandlerModel]):
        self.model_type = new_type

    def create_model(self, asHandler=False):
        return self.model_type(self.args,
                               feature_dim=self.args.feature_dim,
                               points=self.args.points,
                               asHandler=asHandler,
                               key_points=self.args.key_points,
                               structure=self)
