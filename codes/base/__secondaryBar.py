"""
@Author: Conghao Wong
@Date: 2022-10-17 15:02:03
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-17 15:05:15
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Iterable, TypeVar

from .__baseManager import BaseManager

T = TypeVar('T')


class __SecondaryBar(BaseManager):

    def __init__(self, item: Iterable,
                 manager: BaseManager,
                 desc: str = 'Calculating:',
                 pos: str = 'end',
                 name='Secondary InformationBar Manager'):

        if name == 'Secondary InformationBar Manager':
            name += f' ({desc[:20]})'

        super().__init__(manager=manager, name=name)

        if not '__getitem__' in item.__dir__():
            item = list(item)

        self.item = item
        self.desc = desc + ' {}%'
        self.pos = pos

        self.max = len(item)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.max:
            raise StopIteration

        # get value
        value = self.item[self.count]
        self.count += 1

        # update timebar
        percent = (self.count * 100) // self.max
        self.manager.update_timebar(item=self.desc.format(percent),
                                    pos=self.pos)

        return value

    def print_info(self, **kwargs):
        return super().print_info(description=self.desc, **kwargs)


# It is only used for type-hinting
def SecondaryBar(item: T,
                 manager: BaseManager,
                 desc: str = 'Calculating:',
                 pos: str = 'end',
                 name='Secondary InformationBar Manager') -> T:
    """
    Init

    :param item: an iterable object
    :param manager: target manager object to be updated
    :param desc: text to show on the main timebar
    :param pos: text position, can be `'start'` or `'end'`
    """
    return __SecondaryBar(item, manager, desc, pos, name)
