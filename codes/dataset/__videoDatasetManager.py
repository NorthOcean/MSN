"""
@Author: Conghao Wong
@Date: 2022-08-03 09:34:55
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-23 18:43:41
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import random

import tensorflow as tf

from ..base import BaseManager
from ..constant import INPUT_TYPES
from .__agentManager import AgentManager
from .trajectories import Dataset, VideoClipManager


class DatasetManager(BaseManager):
    """
    DatasetsManager
    ---------------
    Manage all trajectory prediction data from one dataset split,
    and then make them into the `tf.data.Dataset` object to train or test.
    The `DatasetManager` object is managed by the `Structure` object.

    Member Managers
    ---------------
    - VideoClip Manager for each video clip, `type = VideoClipManager`;
    - Main Agent Manager, `type = AgentManager`;
    - Agent Manager for each video clip, `type = AgentManager`.

    Public Methods
    ---
    ```python
    # Set types of model inputs and labels in this dataset
    (method) set_types: (self: Self@DatasetManager, 
                         inputs_type: list[str], 
                         labels_type: list[str] = None) -> None

    # Load data and make them into the `Dataset` object from video clips
    (method) load_dataset: (self: Self@DatasetManager,
                            clips: list[str],
                            mode: str) -> DatasetV2
    ```
    """

    def __init__(self, manager: BaseManager, name='Dataset Manager'):
        super().__init__(manager=manager, name=name)

        self.info = Dataset(self.args.dataset, self.args.split)
        self.model_input_type: list[str] = None
        self.model_label_type: list[str] = None
        self.processed_clips: dict[str, list[str]] = {'train': [], 'test': []}

    def set_types(self, inputs_type: list[str], labels_type: list[str] = None):
        """
        Set types of model inputs and labels in this dataset.
        Accept all types in `INPUT_TYPES`.
        """
        self.model_input_type = inputs_type
        if labels_type is not None:
            self.model_label_type = labels_type

    def load_dataset(self, clips: list[str], mode: str) -> tf.data.Dataset:
        """
        Load train samples in sub-datasets (i.e., video clips).

        :param clips: Clips to load. Set it to `'auto'` to load train agents.
        :param mode: The load mode, can be `'test'` or `'train'`.

        :return dataset: The loaded `tf.data.Dataset` object.
        """
        if type(clips) == str:
            clips = [clips]

        # init managers
        clip_managers = [VideoClipManager(self, d) for d in clips]
        agent_manager = AgentManager(self, 'Agent Manager (Chief)')

        # shuffle agents and video clips when training
        if mode == 'train':
            shuffle = True
            random.shuffle(clip_managers)
        else:
            shuffle = False

        # load agent data in each video clip
        for clip in self.timebar(clip_managers):
            # update time bar
            s = f'Prepare data of {mode} agents in `{clip.clip_name}`...'
            self.update_timebar(s, pos='start')

            # file name
            base_dir = os.path.join(clip.path, clip.clip_name)
            if (self.args.obs_frames, self.args.pred_frames) == (8, 12):
                f_name = 'agent'
            else:
                f_name = f'agent_{self.args.obs_frames}to{self.args.pred_frames}'

            endstring = '' if self.args.step == 4 else str(self.args.step)
            f_name = f_name + endstring + '.npz'
            data_path = os.path.join(base_dir, f_name)

            # load agents in this video clip
            agents = AgentManager(self, f'Agent Manager ({clip.clip_name})')
            agents.set_path(npz_path=data_path)

            if not os.path.exists(data_path):
                new_agents = clip.sample_train_data()
                agents.load(new_agents)
                agents.save(data_path)
            else:
                agents.load(data_path)

            # load or make context maps
            if INPUT_TYPES.MAP in self.model_input_type:
                agents.init_map_managers(map_type=self.info.type,
                                         base_path=agents.maps_dir)

                agents.load_maps()

            agent_manager.append(agents)
            agents.destory()

        agent_manager.set_types(inputs_type=self.model_input_type,
                                labels_type=self.model_label_type)

        self.processed_clips[mode] += clips
        return agent_manager.make_dataset(shuffle=shuffle)

    def print_info(self, **kwargs):
        t_info = {'Dataset name': self.info.name,
                  'Dataset annotation type': self.info.anntype,
                  'Split name': self.info.split}

        for mode in ['train', 'test']:
            if len(t := self.processed_clips[mode]):
                t_info.update({f'Clips to {mode}': t})

        return super().print_info(**t_info, **kwargs)
