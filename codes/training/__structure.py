"""
@Author: Conghao Wong
@Date: 2022-06-20 16:27:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-12-06 18:45:43
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Union, overload

import numpy as np
import tensorflow as tf

from ..args import Args
from ..base import BaseManager
from ..basemodels import Model
from ..constant import INPUT_TYPES
from ..dataset import AgentManager, AnnotationManager, DatasetManager
from ..utils import WEIGHTS_FORMAT, dir_check
from ..vis import Visualization
from .loss import LossManager


class Structure(BaseManager):
    """
    Structure
    ---------
    Training manager that manages all training/test-related methods.

    Member Managers
    ---------------
    - Model, `type = Model`;
    - Dataset Manager, `type = DatasetManager`;
    - Annotation Manager, `type = AnnotationManager`;
    - Loss Manager, `type = LossManager`;
    - Metrics Manager, `type = LossManager`.

    Public Methods
    --------------
    ```python
    # Start training or testing
    (method) train_or_test: (self: Self@Structure) -> None
    ```

    Other methods should be rewritten when subclassing.
    """

    def __init__(self, args: Union[list[str], Args] = None,
                 manager: BaseManager = None,
                 name='Train Manager'):

        if issubclass(type(args), Args):
            init_args = args
        else:
            init_args = Args(args)

        super().__init__(init_args, manager, name)

        # init managers
        self.dsmanager = DatasetManager(self)
        self.annmanager = AnnotationManager(self, self.dsmanager.info.anntype)
        self.loss = LossManager(self, name='Loss')
        self.metrics = LossManager(self, name='Metrics')

        # init model options
        self.model: Model = None
        self.set_gpu()
        self.noTraining = False
        self.optimizer = self.set_optimizer()

        # Set labels, loss functions, and metrics
        self.label_types: list[str] = []
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.loss.set({self.loss.ADE: 1.0})

        if self.args.anntype == 'boundingbox':
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0,
                              self.metrics.avgCenter: 0.0,
                              self.metrics.finalCenter: 0.0,
                              self.metrics.AIoU: 0.0,
                              self.metrics.FIoU: 0.0})
        else:
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0})

    def set_labels(self, *args):
        """
        Set label types when calculating loss and metrics.
        Accept keywords:
        ```python
        codes.constant.INPUT_TYPES.OBSERVED_TRAJ
        codes.constant.INPUT_TYPES.MAP
        codes.constant.INPUT_TYPES.DESTINATION_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_SPECTRUM
        codes.constant.INPUT_TYPES.ALL_SPECTRUM
        ```

        :param input_names: Name of the inputs.\
            Type = `str`, accept several keywords.
        """
        self.label_types = [item for item in args]

    def set_optimizer(self, epoch: int = None) -> tf.keras.optimizers.Optimizer:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
        return self.optimizer

    def set_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu.replace('_', ',')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def create_model(self) -> Model:
        """
        Create models.
        Please *rewrite* this when training new models.

        :return model: created model
        """
        raise NotImplementedError('MODEL is not defined!')

    def gradient_operations(self, inputs: list[tf.Tensor],
                            labels: tf.Tensor,
                            loss_move_average: tf.Variable,
                            *args, **kwargs) -> tuple[tf.Tensor, dict[str, tf.Tensor], tf.Tensor]:
        """
        Run gradient descent once during training.

        :param inputs: Model inputs.
        :param labels: Ground truth.
        :param loss_move_average: Moving average loss.

        :return loss: The sum of all single loss functions.
        :return loss_dict: A dict of all loss functions.
        :return loss_move_average: Moving average loss.
        """

        with tf.GradientTape() as tape:
            outputs = self.model.forward(inputs, training=True)
            loss, loss_dict = self.loss.call(outputs, labels,
                                             training=True,
                                             coefficient=1.0)

            loss_move_average = 0.7 * loss + 0.3 * loss_move_average

        grads = tape.gradient(loss_move_average,
                              self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_variables))

        return loss, loss_dict, loss_move_average

    def model_validate(self, inputs: list[tf.Tensor],
                       labels: tf.Tensor,
                       training=None) -> tuple[list[tf.Tensor], tf.Tensor, dict[str, tf.Tensor]]:
        """
        Run one step forward and calculate metrics.

        :param inputs: Model inputs.
        :param labels: Ground truth.

        :return model_output: Model output.
        :return metrics: The weighted sum of all loss.
        :return loss_dict: A dict contains all loss.
        """
        outputs = self.model.forward(inputs, training)
        metrics, metrics_dict = \
            self.metrics.call(outputs, labels,
                              training=None,
                              coefficient=self.dsmanager.info.scale)

        return outputs, metrics, metrics_dict

    def train_or_test(self):
        """
        Load models and datasets, then start training or testing.
        """
        # init model and dataset manager
        self.model = self.create_model()
        self.dsmanager.set_types(inputs_type=self.model.input_types,
                                 labels_type=self.label_types)

        # start training or testing
        if self.noTraining:
            self.test()

        elif self.args.load == 'null':
            # restore weights before training (optional)
            if self.args.restore != 'null':
                self.model.load_weights_from_logDir(self.args.restore)
            self.train()

        else:
            self.model.load_weights_from_logDir(self.args.load)
            self.test()

    def train(self):
        """
        Start training according to the args.
        """
        self.log(f'Start training with args = {self.args._args_runnning}')

        clips_train = self.dsmanager.info.train_sets
        clips_val = self.dsmanager.info.test_sets
        ds_train = self.dsmanager.load_dataset(clips_train, 'train')
        ds_val = self.dsmanager.load_dataset(clips_val, 'test')

        # train on all test/train clips
        _, _, best_metric, best_epoch = self.__train(ds_train, ds_val)
        self.print_train_results(best_epoch=best_epoch,
                                 best_metric=best_metric)

    def test(self):
        """
        Run a test on the given dataset.
        """
        self.log(f'Start test with args = {self.args._args_runnning}')
        test_sets = self.dsmanager.info.test_sets
        r = None

        # test on a single sub-dataset
        if self.args.test_mode == 'one':
            clip = self.args.force_clip
            ds_test = self.dsmanager.load_dataset(clip, 'test')
            r = self.__test(ds_test)

        # test on all test datasets separately
        elif self.args.test_mode == 'all':
            metrics_dict = {}
            for clip in test_sets:
                ds_test = self.dsmanager.load_dataset(clip, 'test')
                _, m_dict, _ = self.__test(ds_test)
                metrics_dict[clip] = m_dict

            self.print_test_results(metrics_dict)

        # test on all test datasets together
        elif self.args.test_mode == 'mix':
            ds_test = self.dsmanager.load_dataset(test_sets, 'test')
            r = self.__test(ds_test)

        else:
            raise NotImplementedError(self.args.test_mode)

        # Write test results
        if r:
            metric, metrics_dict, outputs = r
            self.print_test_results(metrics_dict)
            self.write_test_results(outputs=outputs,
                                    clips=self.dsmanager.processed_clips['test'])

    def __train(self, ds_train: tf.data.Dataset, ds_val: tf.data.Dataset):
        """
        Train the model on the given dataset.

        :param ds_train: The train dataset.
        :param ds_val: The val dataset.

        :return loss_dict:
        :return metrics_dict:
        :return best_metric:
        :return best_epoch:
        """
        # print training infomation
        self.dsmanager.print_info()
        self.model.print_info()
        self.print_info()

        # make a log directory and save current args
        self.args._save_as_json(self.args.log_dir)

        # open tensorboard
        tb = tf.summary.create_file_writer(self.args.log_dir)

        # init variables for training
        loss_move = tf.Variable(0, dtype=tf.float32)
        loss_dict = {}
        metrics_dict = {}

        best_epoch = 0
        best_metric = 10000.0
        best_metrics_dict = {'-': best_metric}
        test_epochs = []
        train_number = len(ds_train)

        # divide with batch size
        ds_train = ds_train.repeat(
            self.args.epochs).batch(self.args.batch_size)

        # start training
        batch_number = len(ds_train)

        epochs = []
        for batch_id, dat in enumerate(self.timebar(ds_train, text='Training...')):

            epoch = (batch_id * self.args.batch_size) // train_number

            # Update learning rate and optimizer
            if not epoch in epochs:
                self.set_optimizer(epoch)
                epochs.append(epoch)

            # Run training once
            len_labels = len(self.label_types)
            loss, loss_dict, loss_move = self.gradient_operations(
                inputs=dat[:-len_labels],
                labels=dat[-len_labels:],
                loss_move_average=loss_move,
                epoch=epoch,
            )

            # Check if `nan` in the loss dictionary
            if tf.math.is_nan(loss):
                self.log(f'Find `nan` values in the loss dictionary, ' +
                         f'stop training... ' +
                         f'Best metrics obtained from the last epoch: ' +
                         f'{best_metrics_dict}.',
                         level='error', raiseError=ValueError)

            # Run validation
            if ((epoch >= self.args.start_test_percent * self.args.epochs)
                    and ((epoch - 1) % self.args.test_step == 0)
                    and (not epoch in test_epochs)
                    and (epoch > 0)) or (batch_id == batch_number - 1):

                metric, metrics_dict = self.__test_on_dataset(
                    ds=ds_val,
                    show_timebar=False,
                    test_during_training=True
                )
                test_epochs.append(epoch)

                # Save model
                if metric <= best_metric:
                    best_metric = metric
                    best_metrics_dict = metrics_dict
                    best_epoch = epoch

                    self.model.save_weights(os.path.join(
                        self.args.log_dir,
                        f'{self.args.model_name}_epoch{epoch}' + WEIGHTS_FORMAT
                    ))

                    np.savetxt(os.path.join(self.args.log_dir, 'best_ade_epoch.txt'),
                               np.array([best_metric, best_epoch]))

            # Save results into log files
            log_dict = dict(epoch=epoch,
                            best=list(best_metrics_dict.values()),
                            **loss_dict,
                            **metrics_dict)

            # Show to users
            self.update_timebar(log_dict, pos='end')

            # Write tensorboard
            with tb.as_default():
                for name in log_dict.keys():
                    if name == 'best':
                        continue

                    value = log_dict[name]
                    tf.summary.scalar(name, value, step=epoch)

        return log_dict, metrics_dict, best_metric, best_epoch

    def __test(self, ds_test: tf.data.Dataset) -> \
            tuple[float, dict[str, float], list[tf.Tensor]]:
        """
        Test model on the given dataset.

        :param ds_test: The test dataset.

        :return metric:
        :return metrics_dict
        :return outputs: model outputs
        """
        # Print test information
        self.dsmanager.print_info()
        self.model.print_info()
        self.print_info()

        # make a log directory and save current args
        if self.args.update_saved_args:
            if self.args.load != 'null':
                self.args._save_as_json(self.args.load)
            else:
                self.args._save_as_json(self.args.log_dir)

        # Run test
        outputs, metric, metrics_dict = self.__test_on_dataset(
            ds=ds_test,
            return_results=True,
            show_timebar=True,
        )

        return metric, metrics_dict, outputs

    @overload
    def __test_on_dataset(self, ds: tf.data.Dataset,
                          show_timebar=False,
                          test_during_training=False) \
        -> tuple[float, dict[str, float]]: ...

    @overload
    def __test_on_dataset(self, ds: tf.data.Dataset,
                          return_results=False,
                          show_timebar=False,
                          test_during_training=False) \
        -> tuple[list[tf.Tensor], float, dict[str, float]]: ...

    def __test_on_dataset(self, ds: tf.data.Dataset,
                          return_results=False,
                          show_timebar=False,
                          test_during_training=False):
        """
        Run a test on the given dataset.

        :param ds: The test `tf.data.Dataset` object.
        :param return_results: Controls items to return (the defaule value is `False`).
        :param show_timebar: Controls whether to show the process.
        :param test_during_training: Indicates whether to test during training.

        Returns if `return_results == False`:
        :return metric: The weighted sum of all metrics.
        :return metric_dict: A dict of all metrics.

        Returns if `return_results == True`:
        :return outputs: A list of model outputs.
        :return metric: The weighted sum of all metrics.
        :return metric_dict: A dict of all metrics.
        """
        # init variables for test
        outputs_all = []
        metrics_all = []
        metrics_dict_all = {}

        # divide with batch size
        ds = ds.batch(self.args.batch_size)

        # hide time bar when training
        timebar = self.timebar(ds, 'Test...') if show_timebar else ds

        count = []
        for dat in timebar:
            len_labels = len(self.label_types)
            outputs, metrics, metrics_dict = self.model_validate(
                inputs=dat[:-len_labels],
                labels=dat[-len_labels:],
                training=False,
            )

            # Add metrics and outputs to their dicts
            count.append(outputs[0].shape[0])
            metrics_all.append(metrics)
            metrics_dict_all = append_batch_results(
                metrics_dict_all, metrics_dict)

            if return_results:
                outputs_all = append_batch_results(outputs_all, outputs)

        # Stack all model results
        if return_results:
            outputs_all = stack_results(outputs_all)

        # calculate average metric
        m_avg = weighted_avg_results([metrics_all], count, numpy=True)[0]
        mdict_avg = weighted_avg_results(metrics_dict_all, count, numpy=True)

        # Inference time
        if not test_during_training:
            if len(self.model.inference_times) < 3:
                self.log('The "AverageInferenceTime" is for reference only and you can set a lower "batch_size" ' +
                         'or change a bigger dataset to obtain a more accurate result.')

            mdict_avg['Average Inference Time'] = f'{self.model.average_inference_time} ms'
            mdict_avg['Fastest Inference Time'] = f'{self.model.fastest_inference_time} ms'

        if return_results:
            return outputs_all, m_avg, mdict_avg
        else:
            return m_avg, mdict_avg

    def print_info(self, **kwargs):
        info = {'Batch size': self.args.batch_size,
                'GPU index': self.args.gpu,
                'Train epochs': self.args.epochs,
                'Learning rate': self.args.lr}

        kwargs.update(**info)
        return super().print_info(**kwargs)

    def print_train_results(self, best_epoch: int, best_metric: float):
        """
        Print train results on the screen.
        """
        self.log('Training done.')
        self.log('During training, the model reaches the best metric ' +
                 f'`{best_metric}` at epoch {best_epoch}.')

        self.log(f'Tensorboard file is saved at `{self.args.log_dir}`. ' +
                 'To open this log file, please use `tensorboard ' +
                 f'--logdir {self.args.log_dir}`')
        self.log(f'Trained model is saved at `{self.args.log_dir}`. ' +
                 'To re-test this model, please use ' +
                 f'`python main.py --load {self.args.log_dir}`.')

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        """
        Print test results on the screen.
        """
        self.print_parameters(title='Test Results',
                              **kwargs,
                              **loss_dict)
        self.log(f'split: {self.args.split}, ' +
                 f'load: {self.args.load}, ' +
                 f'metrics: {loss_dict}')

    def write_test_results(self, outputs: list[tf.Tensor], clips: list[str]):
        """
        Save visualized prediction results.
        """

        # get agents' information
        agents = self.dsmanager.get_member(AgentManager, mindex=0)

        if (((self.args.draw_results != 'null') or
             (self.args.draw_videos != 'null'))
                and len(clips) == 1):

            # draw results on video frames
            clip = clips[0]
            tv = Visualization(self, self.args.dataset, clip)

            save_base_path = dir_check(self.args.log_dir) \
                if self.args.load == 'null' \
                else self.args.load

            img_dir = dir_check(os.path.join(save_base_path, 'VisualTrajs'))
            save_format = os.path.join(img_dir, clip + '_{}')
            self.log(f'Start saving images into `{img_dir}`...')

            pred_all = outputs[0].numpy()

            if self.args.draw_index == 'all':
                agent_indexes = list(range(len(pred_all)))
            else:
                _indexes = self.args.draw_index.split('_')
                agent_indexes = [int(i) for i in _indexes]

            for index in self.timebar(agent_indexes, 'Saving...'):
                # write traj
                agent = agents.agents[index]
                agent._traj_pred = pred_all[index]

                # choose to draw as a video or a single image
                if self.args.draw_videos != 'null':
                    save_image = False
                    frames = agent.frames
                else:
                    save_image = True
                    frames = [agent.frames[self.args.obs_frames-1]]

                tv.draw(agent=agent,
                        frames=frames,
                        save_name=save_format.format(index),
                        draw_dis=self.args.draw_distribution,
                        save_as_images=save_image)

            self.log(f'Prediction result images are saved at {img_dir}')


def stack_results(results: list[tf.Tensor]):
    for index, tensor in enumerate(results):
        results[index] = tf.concat(tensor, axis=0)
    return results


@overload
def append_batch_results(results_container: list[list[tf.Tensor]],
                         new_results: list[tf.Tensor]) -> list[list[tf.Tensor]]: ...


@overload
def append_batch_results(results_container: dict[str, list[tf.Tensor]],
                         new_results: dict[str, tf.Tensor]) -> dict[str, list[tf.Tensor]]: ...


def append_batch_results(source, new):
    if type(new) in [list, tuple]:
        if not len(source):
            for _ in range(len(new)):
                source.append([])

        for index, value in enumerate(new):
            source[index].append(value)

    elif type(new) in [dict]:
        if not len(source):
            for key in new.keys():
                source[key] = []

        for [key, value] in new.items():
            source[key].append(value)

    else:
        raise TypeError(new)

    return source


@overload
def weighted_avg_results(target: list[list[tf.Tensor]],
                         weights: list,
                         numpy=False) -> list[tf.Tensor]: ...


@overload
def weighted_avg_results(target: dict[str, list[tf.Tensor]],
                         weights: list,
                         numpy=False) -> dict[str, tf.Tensor]: ...


def weighted_avg_results(target, weights: list, numpy=False):

    weights = tf.cast(weights, tf.float32)
    count = tf.reduce_sum(weights)

    if type(target) in [list, tuple]:
        new_res = []
        for item in target:
            sum = tf.reduce_sum(item * weights)
            res = sum/count
            if numpy:
                res = res.numpy()
            new_res.append(res)

    elif type(target) in [dict]:
        new_res = {}
        for key, value in target.items():
            sum = tf.reduce_sum(value * weights)
            res = sum/count
            if numpy:
                res = res.numpy()
            new_res[key] = res

    else:
        raise TypeError(target)

    return new_res
