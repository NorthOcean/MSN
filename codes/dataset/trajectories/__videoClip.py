"""
@Author: Conghao Wong
@Date: 2022-07-19 10:32:41
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:19:31
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Union

from ...utils import DATASET_CONFIG_DIR, dir_check, load_from_plist
from .__videoDataset import Dataset


class VideoClip():
    """
    VideoClip
    ---------
    Base structure for controlling each video dataset.

    Properties
    -----------------
    ```python
    >>> self.dataset        # name of the dataset
    >>> self.name           # video clip name
    >>> self.annpath        # dataset annotation file
    >>> self.order          # X-Y order in the annotation file
    >>> self.paras          # [sample_step, frame_rate]
    >>> self.video_path     # video path   
    >>> self.matrix         # transfer matrix from real scales to pixels
    ```

    Public Methods
    ---
    ```python
    # Load video clip information from the saved `plist` file
    (method) get: (self: Self@VideoClip) -> VideoClip

    # Update dataset information (including dataset splits)
    (method) update_datasetInfo: (self: Self@VideoClip, dataset: str | Dataset,
                                  split: str = None) -> None
    ```
    """

    # Saving paths
    BASE_DIR = './dataset_configs'
    CONFIG_FILE = os.path.join(BASE_DIR, '{}', 'subsets', '{}.plist')

    def __init__(self, name: str,
                 dataset: str,
                 annpath: str = None,
                 order: tuple[int, int] = None,
                 paras: tuple[int, int] = None,
                 video_path: str = None,
                 matrix: list[float] = None,
                 datasetInfo: Dataset = None,
                 *args, **kwargs):

        self.__name = name
        self.__annpath = annpath
        self.__order = order
        self.__paras = paras
        self.__video_path = video_path
        self.__matrix = matrix
        self.__dataset = dataset

        self.CONFIG_FILE = self.CONFIG_FILE.format(self.dataset, self.name)

        # init dataset information
        self.datasetInfo = datasetInfo
        if datasetInfo is None:
            self.update_datasetInfo(dataset=dataset)

        # make dirs
        dirs = [self.CONFIG_FILE]
        for d in dirs:
            _dir = os.path.dirname(d)
            dir_check(_dir)

    def get(self):
        """
        Load video clip information from the saved `plist` file.
        """
        plist_path = self.CONFIG_FILE

        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(f'Dataset file `{plist_path}` NOT FOUND.')

        return VideoClip(**dic)

    def update_datasetInfo(self, dataset: Union[str, Dataset],
                           split: str = None):
        """
        Update dataset information (including dataset splits).

        :param dataset: The name of the dataset, or the `Dataset` object.
        :param split: The name of the dataset split.
        """
        if type(dataset) is str:
            if not split:
                ds_split_dir = os.path.join(DATASET_CONFIG_DIR, dataset)
                split_names = [p.split('.plist')[0] for p in
                               os.listdir(ds_split_dir) if p.endswith('plist')]
                split = split_names[0]

            self.datasetInfo = Dataset(dataset, split)

        elif issubclass(type(dataset), Dataset):
            self.datasetInfo = dataset

        else:
            raise ValueError(dataset)

    @property
    def dataset(self) -> str:
        """
        Name of the dataset.
        """
        return self.__dataset

    @property
    def name(self):
        """
        Name of the video clip.
        """
        return self.__name

    @property
    def annpath(self) -> str:
        """
        Path of the annotation file. 
        """
        return self.__annpath

    @property
    def order(self) -> list[int]:
        """
        X-Y order in the annotation file.
        """
        return self.__order

    @property
    def paras(self) -> tuple[int, int]:
        """
        [sample_step, frame_rate]
        """
        return self.__paras

    @property
    def video_path(self) -> str:
        """
        Path of the video file.
        """
        return self.__video_path

    @property
    def matrix(self) -> list[float]:
        """
        transfer weights from real scales to pixels.
        """
        return self.__matrix
