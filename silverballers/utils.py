"""
@Author: Conghao Wong
@Date: 2022-07-27 20:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-29 10:20:12
@Description: file content
@Github: https://northocean.github.io
@Copyright 2023 Beihao Xia, All Rights Reserved.
"""

from codes import INTERPOLATION_TYPES
from codes.args import Args

from . import agents, handlers
from .__args import SilverballersArgs
from .__baseSilverballers import BaseSilverballers


class SilverballersMKII(BaseSilverballers):

    def __init__(self, terminal_args: list[str]):

        min_args = SilverballersArgs(terminal_args, is_temporary=True)
        a_model_path = min_args.loada
        b_model_path = min_args.loadb

        # Assign the model type of the first-stage subnetwork
        min_args_a = Args(is_temporary=True)._load_from_json(a_model_path)
        agent_model = get_model(min_args_a.model)

        # Assign the model type of the second-stage subnetwork
        interp_model = INTERPOLATION_TYPES.get_type(b_model_path)
        if interp_model is None:
            min_args_b = Args(is_temporary=True)._load_from_json(b_model_path)
            handler_model = get_model(min_args_b.model)
        else:
            handler_model = get_model(interp_model)

        self.set_models(agentModel=agent_model, handlerModel=handler_model)

        super().__init__(terminal_args)


__SILVERBALLERS_DICT = dict(

    # Agent Structures and Models

    # MSN
    msna=[agents.MSNAlpha, agents.MSNAlphaModel],
    msnb=[handlers.MSNBeta, handlers.MSNBetaModel],

    # Silverballers Structures
    MKII=[SilverballersMKII, None],
)

# Interpolation Handlers
__SILVERBALLERS_DICT.update({
    INTERPOLATION_TYPES.LINEAR: [None, handlers.interp.LinearHandlerModel],
    INTERPOLATION_TYPES.LINEAR_SPEED: [None, handlers.interp.LinearSpeedHandlerModel],
    INTERPOLATION_TYPES.LINEAR_ACC: [None, handlers.interp.LinearAccHandlerModel],
    INTERPOLATION_TYPES.NEWTON: [None, handlers.interp.NewtonHandlerModel],
})


def get_structure(model_name: str):
    return __get(model_name)[0]


def get_model(model_name: str):
    return __get(model_name)[1]


def __get(model_name: str):
    if not model_name in __SILVERBALLERS_DICT.keys():
        raise NotImplementedError(
            f'model type `{model_name}` is not supported.')

    return __SILVERBALLERS_DICT[model_name]
