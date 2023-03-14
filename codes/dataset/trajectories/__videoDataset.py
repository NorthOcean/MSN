"""
@Author: Conghao Wong
@Date: 2022-07-19 11:19:58
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-10 11:20:08
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

from ...utils import DATASET_CONFIG_DIR, load_from_plist


class Dataset():
    """
    Dataset
    -------
    Manage a full trajectory prediction dataset.
    A dataset may contain several video clips.
    One `Dataset` object only controls one dataset split.

    Properties
    ---
    ```python
    # Name of the video dataset
    (property) name: (self: Self@Dataset) -> str

    # Annotation type of the dataset
    (property) type: (self: Self@Dataset) -> str

    # Global data scaling scale
    (property) scale: (self: Self@Dataset) -> float

    # Video scaling when saving visualized results
    (property) scale_vis: (self: Self@Dataset) -> float

    # Maximum dimension of trajectories recorded in this dataset
    (property) dimension: (self: Self@Dataset) -> int

    # Type of annotations
    (property) anntype: (self: Self@Dataset) -> str
    ```
    """

    # Saving paths
    BASE_DIR = DATASET_CONFIG_DIR
    CONFIG_FILE = os.path.join(BASE_DIR, '{}', '{}.plist')

    def __init__(self, name: str, split: str):
        """
        :param name: The name of the image dataset.
        :param split: The split name of the dataset.
        """
        split_path = self.CONFIG_FILE.format(name, split)

        try:
            dic = load_from_plist(split_path)
        except:
            raise FileNotFoundError(f'Dataset file `{split_path}` NOT FOUND.')

        self.__name = dic['dataset']
        self.__type = dic['type']
        self.__scale = dic['scale']
        self.__scale_vis = dic['scale_vis']
        self.__dimension = dic['dimension']
        self.__anntype = dic['anntype']

        self.split: str = split
        self.train_sets: list[str] = dic['train']
        self.test_sets: list[str] = dic['test']
        self.val_sets: list[str] = dic['val']

    @property
    def name(self) -> str:
        """
        Name of the video dataset.
        For example, `ETH-UCY` or `SDD`.
        """
        return self.__name

    @property
    def type(self) -> str:
        """
        Annotation type of the dataset.
        For example, `'pixel'` or `'meter'`.
        """
        return self.__type

    @property
    def scale(self) -> float:
        """
        Global data scaling scale.
        """
        return self.__scale

    @property
    def scale_vis(self) -> float:
        """
        Video scaling when saving visualized results.
        """
        return self.__scale_vis

    @property
    def dimension(self) -> int:
        """
        Maximum dimension of trajectories recorded in this dataset.
        For example, `(x, y)` -> `dimension = 2`.
        """
        return self.__dimension

    @property
    def anntype(self) -> str:
        """
        Type of annotations.
        For example, `'coordinate'` or `'boundingbox'`.
        """
        return self.__anntype
