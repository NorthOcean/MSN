"""
@Author: Conghao Wong
@Date: 2022-07-15 20:13:07
@LastEditors: Conghao Wong
@LastEditTime: 2022-12-06 19:47:07
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from ..args import DYNAMIC, Args
from ..basemodels import Model, layers
from ..training import Structure


class LinearArgs(Args):
    def __init__(self, terminal_args: list[str] = None, is_temporary=False) -> None:
        super().__init__(terminal_args, is_temporary)
    
    @property
    def weights(self) -> float:
        """
        The weights in the calculation of the mean squared error at 
        different moments of observation.
        Set to `0.0` to disable this function.
        """
        return self._arg('weights', default=0.0, argtype=DYNAMIC)


class LinearModel(Model):
    def __init__(self, Args: LinearArgs, structure=None, *args, **kwargs):
        super().__init__(Args, structure, *args, **kwargs)

        self.linear = layers.LinearLayerND(obs_frames=self.args.obs_frames,
                                           pred_frames=self.args.pred_frames,
                                           diff=Args.weights)

    def call(self, inputs, training=None, *args, **kwargs):
        trajs = inputs[0]
        return self.linear.call(trajs)


class Linear(Structure):
    def __init__(self, terminal_args: list[str]):

        self.args = LinearArgs(terminal_args)
        super().__init__(self.args)
        self.noTraining = True

    def create_model(self, *args, **kwargs) -> Model:
        return LinearModel(self.args, structure=self)




    