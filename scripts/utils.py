"""
@Author: Conghao Wong
@Date: 2022-11-11 09:28:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-14 11:19:57
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np


def get_value(key: str, args: list[str], default=None):
    """
    `key` is started with `--`.
    For example, `--logs`.
    """
    args = np.array(args)
    index = np.where(args == key)[0][0]

    try:
        return str(args[index+1])
    except IndexError:
        return default
