"""
@Author: Conghao Wong
@Date: 2022-06-22 10:36:15
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-21 16:02:40
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from . import agents, handlers
from .__baseSilverballers import BaseSilverballers


class V(BaseSilverballers):
    """
    V^2-Net
    ---

    Training structure for the V^2-Net model.
    It has keypoints-interpolation two sub-networks.
    Both these sub-networks implement on agents' trajectory spectrums.
    """

    def __init__(self, terminal_args: list[str]):

        self.set_models(agentModel=agents.VAModel,
                        handlerModel=handlers.VBModel)

        super().__init__(terminal_args)


class Silverballers47C(BaseSilverballers):

    def __init__(self, terminal_args: list[str]):

        self.set_models(agentModel=agents.Agent47CModel,
                        handlerModel=handlers.BurnwoodCModel)

        super().__init__(terminal_args)
