"""
@Author: Conghao Wong
@Date: 2022-06-20 21:41:10
@LastEditors: Beihao Xia
@LastEditTime: 2023-03-06 11:15:40
@Description: file content
@Github: https://northocean.github.io
@Copyright 2023 Beihao Xia, All Rights Reserved.
"""

from codes.args import DYNAMIC, STATIC, TEMPORARY, Args


class _BaseSilverballersArgs(Args):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

        self._set_default('K', 1)
        self._set_default('K_train', 1)

    @property
    def Kc(self) -> int:
        """
        The number of style channels in `Agent` model.
        """
        return self._arg('Kc', 20, argtype=STATIC)

    @property
    def key_points(self) -> str:
        """
        A list of key time steps to be predicted in the agent model.
        For example, `'0_6_11'`.
        """
        return self._arg('key_points', '0_6_11', argtype=STATIC)

    @property
    def preprocess(self) -> str:
        """
        Controls whether to run any pre-process before the model inference.
        It accepts a 3-bit-like string value (like `'111'`):
        - The first bit: `MOVE` trajectories to (0, 0);
        - The second bit: re-`SCALE` trajectories;
        - The third bit: `ROTATE` trajectories.
        """
        return self._arg('preprocess', '111', argtype=STATIC)

    @property
    def feature_dim(self) -> int:
        """
        Feature dimensions that are used in most layers.
        """
        return self._arg('feature_dim', 128, argtype=STATIC)


class AgentArgs(_BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def depth(self) -> int:
        """
        Depth of the random noise vector.
        """
        return self._arg('depth', 16, argtype=STATIC)


class HandlerArgs(_BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

        self._set_default('key_points', 'null', overwrite=False)

    @property
    def points(self) -> int:
        """
        The number of keypoints accepted in the handler model.
        """
        return self._arg('points', 1, argtype=STATIC)


class SilverballersArgs(_BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def loada(self) -> str:
        """
        Path to load the first-stage agent model.
        """
        return self._arg('loada', 'null', argtype=TEMPORARY, short_name='la')

    @property
    def loadb(self) -> str:
        """
        Path to load the second-stage handler model.
        """
        return self._arg('loadb', 'null', argtype=TEMPORARY, short_name='lb')
