"""
@Author: Conghao Wong
@Date: 2022-06-20 10:53:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-14 16:35:45
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import re

from ..utils import DATASET_CONFIG_DIR, TIME, dir_check
from .__argsManager import DYNAMIC, STATIC, TEMPORARY, ArgsManager

NA = 'Unavailable'


class Args(ArgsManager):
    """
    A set of args used for training or evaluating prediction models.
    """

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def batch_size(self) -> int:
        """
        Batch size when implementation.
        """
        return self._arg('batch_size', 5000, argtype=DYNAMIC, short_name='bs')

    @property
    def dataset(self) -> str:
        """
        Name of the video dataset to train or evaluate.
        For example, `'ETH-UCY'` or `'SDD'`.
        NOTE: DO NOT set this argument manually.
        """

        def preprocess(self: Args):
            if self.force_dataset != 'null':
                if self.update_saved_args:
                    self.log('Parameters cannot be saved when forced ' +
                             'parameters (`force_dataset`, `force_split`) ' +
                             'are set. Please remove them and run again.',
                             level='error', raiseError=ValueError)

                self._set('dataset', self.force_dataset)

            # This argument can only be set manually by codes
            # or read from the saved JSON file
            elif 'dataset' not in self._args_load.keys():

                dirs = os.listdir(DATASET_CONFIG_DIR)

                plist_files = []
                for d in dirs:
                    try:
                        _path = os.path.join(DATASET_CONFIG_DIR, d)
                        for p in os.listdir(_path):
                            if p.endswith('.plist'):
                                plist_files.append(os.path.join(_path, p))
                    except:
                        pass

                dataset = None
                for f in plist_files:
                    res = re.findall(
                        f'{DATASET_CONFIG_DIR}/(.*)/({self.split}.plist)', f)

                    if len(res):
                        dataset = res[0][0]
                        break

                if not dataset:
                    raise ValueError(self.split)

                self._set('dataset', dataset)

        return self._arg('dataset', NA, argtype=STATIC,
                         preprocess=preprocess)

    @property
    def force_dataset(self) -> str:
        """
        Force test dataset (ignore the train/test split).
        It only works when `test_mode` has been set to `one`.
        """
        return self._arg('force_dataset', 'null', argtype=TEMPORARY)

    @property
    def split(self) -> str:
        """
        The dataset split that used to train and evaluate.
        """

        def preprocess(self: Args):
            if self.force_split != 'null':
                if self.update_saved_args:
                    self.log('Parameters cannot be saved when forced ' +
                             'parameters (`force_dataset`, `force_split`) ' +
                             'are set. Please remove them and run again.',
                             level='error', raiseError=ValueError)

                self._set('split', self.force_split)

        return self._arg('split', 'zara1', argtype=STATIC,
                         short_name='s',
                         preprocess=preprocess)

    @property
    def force_split(self) -> str:
        """
        Force test dataset (ignore the train/test split). 
        It only works when `test_mode` has been set to `one`.
        """
        return self._arg('force_split', 'null', argtype=TEMPORARY)

    @property
    def epochs(self) -> int:
        """
        Maximum training epochs.
        """
        return self._arg('epochs', 500, argtype=STATIC)

    @property
    def force_clip(self) -> str:
        """
        Force test video clip (ignore the train/test split).
        It only works when `test_mode` has been set to `one`. 
        """

        def preprocess(self: Args):
            if self.draw_results != 'null':
                self._set('force_clip', self.draw_results)

            if self.draw_videos != 'null':
                self._set('force_clip', self.draw_videos)

        return self._arg('force_clip', 'null', argtype=TEMPORARY,
                         preprocess=preprocess)

    @property
    def gpu(self) -> str:
        """
        Speed up training or test if you have at least one NVidia GPU. 
        If you have no GPUs or want to run the code on your CPU, 
        please set it to `-1`.
        NOTE: It only supports training or testing on one GPU.
        """
        return self._arg('gpu', '0', argtype=TEMPORARY)

    @property
    def save_base_dir(self) -> str:
        """
        Base folder to save all running logs.
        """
        return self._arg('save_base_dir', './logs', argtype=STATIC)

    @property
    def start_test_percent(self) -> float:
        """
        Set when (at which epoch) to start validation during training.
        The range of this arg should be `0 <= x <= 1`. 
        Validation may start at epoch
        `args.epochs * args.start_test_percent`.
        """
        return self._arg('start_test_percent', 0.0, argtype=STATIC)

    @property
    def log_dir(self) -> str:
        """
        Folder to save training logs and model weights.
        Logs will save at `args.save_base_dir/current_model`.
        DO NOT change this arg manually. (You can still change
        the path by passing the `save_base_dir` arg.)
        """

        def preprocess(self: Args):
            if self._is_temporary:
                return

            # This argument can only be set manually by codes
            # or read from the saved JSON file
            if 'log_dir' not in self._args_load.keys():

                log_dir_current = (TIME +
                                   self.model_name +
                                   self.model +
                                   self.split)

                default_log_dir = os.path.join(dir_check(self.save_base_dir),
                                               log_dir_current)

                self._set('log_dir', dir_check(default_log_dir))

        return self._arg('log_dir', NA, argtype=STATIC,
                         preprocess=preprocess)

    @property
    def load(self) -> str:
        """
        Folder to load model (to test). If set to `null`, the
        training manager will start training new models according
        to other given args.
        """
        return self._arg('load', 'null', argtype=TEMPORARY, short_name='l')

    @property
    def model(self) -> str:
        """
        The model type used to train or test.
        """
        return self._arg('model', 'none', argtype=STATIC)

    @property
    def model_name(self) -> str:
        """
        Customized model name.
        """
        return self._arg('model_name', 'model', argtype=STATIC)

    @property
    def restore(self) -> str:
        """
        Path to restore the pre-trained weights before training.
        It will not restore any weights if `args.restore == 'null'`.
        """
        return self._arg('restore', 'null', argtype=TEMPORARY)

    @property
    def restore_args(self) -> str:
        """
        Path to restore the reference args before training.
        It will not restore any args if `args.restore_args == 'null'`.
        """
        return self._arg('restore_args', 'null', argtype=TEMPORARY)

    @property
    def test_step(self) -> int:
        """
        Epoch interval to run validation during training.
        """
        return self._arg('test_step', 3, argtype=STATIC)

    """
    Trajectory Prediction Args
    """
    @property
    def obs_frames(self) -> int:
        """
        Observation frames for prediction.
        """
        return self._arg('obs_frames', 8, argtype=STATIC, short_name='obs')

    @property
    def pred_frames(self) -> int:
        """
        Prediction frames.
        """
        return self._arg('pred_frames', 12, argtype=STATIC, short_name='pred')

    @property
    def draw_results(self) -> str:
        """
        Controls whether to draw visualized results on video frames.
        Accept the name of one video clip. The codes will first try to
        load the video file according to the path saved in the `plist`
        file (saved in `dataset_configs` folder), and if it loads successfully
        it will draw the results on that video, otherwise it will draw results
        on a blank canvas. Note that `test_mode` will be set to `'one'` and
        `force_split` will be set to `draw_results` if `draw_results != 'null'`.
        """
        return self._arg('draw_results', 'null', argtype=TEMPORARY, short_name='dr')

    @property
    def draw_videos(self) -> str:
        """
        Controls whether draw visualized results on video frames and save as images.
        Accept the name of one video clip.
        The codes will first try to load the video according to the path
        saved in the `plist` file, and if successful it will draw the
        visualization on the video, otherwise it will draw on a blank canvas.
        Note that `test_mode` will be set to `'one'` and `force_split`
        will be set to `draw_videos` if `draw_videos != 'null'`.
        """
        return self._arg('draw_videos', 'null', argtype=TEMPORARY)

    @property
    def draw_index(self) -> str:
        """
        Indexes of test agents to visualize.
        Numbers are split with `_`.
        For example, `'123_456_789'`.
        """
        return self._arg('draw_index', 'all', argtype=TEMPORARY)

    @property
    def draw_distribution(self) -> int:
        """
        Controls if draw distributions of predictions instead of points.
        If `draw_distribution == 0`, it will draw results as normal coordinates;
        If `draw_distribution == 1`, it will draw all results in the distribution
        way, and points from different time steps will be drawn with different colors.
        """
        return self._arg('draw_distribution', 0, argtype=TEMPORARY, short_name='dd')

    @property
    def step(self) -> int:
        """
        Frame interval for sampling training data.
        """
        return self._arg('step', 1, argtype=DYNAMIC)

    @property
    def test_mode(self) -> str:
        """
        Test settings. It can be `'one'`, `'all'`, or `'mix'`.
        When setting it to `one`, it will test the model on the `args.force_split` only;
        When setting it to `all`, it will test on each of the test datasets in `args.split`;
        When setting it to `mix`, it will test on all test datasets in `args.split` together.
        """

        def preprocess(self: Args):
            if self.draw_results != 'null' or self.draw_videos != 'null':
                self._set('test_mode', 'one')

        return self._arg('test_mode', 'mix', argtype=TEMPORARY,
                         preprocess=preprocess)

    @property
    def lr(self) -> float:
        """
        Learning rate.
        """
        return self._arg('lr', 0.001, argtype=STATIC)

    @property
    def K(self) -> int:
        """
        Number of multiple generations when testing.
        This arg only works for multiple-generation models.
        """
        return self._arg('K', 20, argtype=DYNAMIC)

    @property
    def K_train(self) -> int:
        """
        The number of multiple generations when training.
        This arg only works for multiple-generation models.
        """
        return self._arg('K_train', 10, argtype=STATIC)

    @property
    def use_extra_maps(self) -> int:
        """
        Controls if uses the calculated trajectory maps or the given
        trajectory maps. The training manager will load maps from 
        `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`,
        and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` 
        if set this argument to `1`.
        """
        return self._arg('use_extra_maps', 0, argtype=DYNAMIC)

    @property
    def dim(self) -> int:
        """
        Dimension of the `trajectory`.
        For example,
        - coordinate (x, y) -> `dim = 2`;
        - boundingbox (xl, yl, xr, yr) -> `dim = 4`.
        """

        def preprocess(self: Args):
            if self.anntype == 'coordinate':
                self._set('dim', 2)
            elif self.anntype == 'boundingbox':
                self._set('dim', 4)
            else:
                raise ValueError(self.anntype)

        return self._arg('dim', -1, argtype=STATIC,
                         preprocess=preprocess)

    @property
    def anntype(self) -> str:
        """
        Model's predicted annotation type.
        Can be `'coordinate'` or `'boundingbox'`.
        """
        return self._arg('anntype', 'coordinate', argtype=STATIC)

    @property
    def interval(self) -> float:
        """
        Time interval of each sampled trajectory point.
        """
        return self._arg('interval', 0.4, argtype=STATIC)

    @property
    def pmove(self) -> int:
        """
        Index of the reference point when moving trajectories.
        """
        return self._arg('pmove', -1, argtype=STATIC)

    @property
    def pscale(self) -> str:
        """
        Index of the reference point when scaling trajectories.
        """
        return self._arg('pscale', 'autoref', argtype=STATIC)

    @property
    def protate(self) -> float:
        """
        Reference degree when rotating trajectories.
        """
        return self._arg('protate', 0.0, argtype=STATIC)

    @property
    def update_saved_args(self) -> int:
        """
        Choose whether to update (overwrite) the saved arg files or not.
        """
        return self._arg('update_saved_args', 0, argtype=TEMPORARY)

    @property
    def auto_dimension(self) -> int:
        """
        Choose whether to handle the dimension adaptively.
        It is now only used for silverballers models that are trained
        with annotation type `coordinate` but want to test on datasets
        with annotation type `boundingbox`.
        """
        return self._arg('auto_dimension', 0, argtype=TEMPORARY)
