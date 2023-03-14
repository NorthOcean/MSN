"""
@Author: Conghao Wong
@Date: 2022-10-17 14:57:03
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-11 10:09:19
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import TypeVar

from .__args import Args
from .__baseObject import BaseObject

T = TypeVar('T')


class BaseManager(BaseObject):
    """
    BaseManager
    ----------
    Basic object with logging-related and members/manager functions.
    It also adds support for the `Args` object.

    Public Methods
    --------------
    ### Manager and members' methods
    ```python
    # get a member by type
    (method) get_member: (self: Self@BaseManager,
                          mtype: Type[T@get_member],
                          mindex: int = 0) -> T@get_member

    # get all members with the same type
    (method) find_members_by_type: (self: Self@BaseManager,
                                    mtype: Type[T@find_members_by_type]) \
                                         -> list[T@find_members_by_type]
    ```

    ### Log and print methods
    ```
    # log information
    (method) log: (self: Self@BaseManager, s: str, level: str = 'info') -> str

    # print parameters with the format
    (method) print_parameters: (self: Self@BaseManager, 
                                title: str = 'null',
                                **kwargs: Any) -> None

    # print information about the manager object itself
    (method) print_info: (self: Self@BaseManager, **kwargs: Any) -> None

    # print information of the manager and its members
    (method) print_info_all: (self: Self@BaseManager,
                              include_self: bool = True) -> None

    # timebar
    (method) log_timebar: (inputs, text='', return_enumerate=True) -> (enumerate | tqdm)
    ```
    """

    def __init__(self, args: Args = None,
                 manager: BaseObject = None,
                 name: str = None):

        super().__init__(name)
        self._args: Args = args
        self.manager: BaseManager = manager
        self.members: list[BaseManager] = []
        self.members_dict: dict[type[BaseManager], list[BaseManager]] = {}

        if manager:
            mtype = type(self)
            if not mtype in self.manager.members_dict.keys():
                self.manager.members_dict[mtype] = []

            self.manager.members_dict[mtype].append(self)
            self.manager.members.append(self)

    @property
    def args(self) -> Args:
        if self._args:
            return self._args
        elif self.manager:
            return self.manager.args
        else:
            return None

    @args.setter
    def args(self, value: T) -> T:
        self._args = value

    def destory(self):
        if self.manager:
            self.manager.members.remove(self)
            self.manager.members_dict[type(self)].remove(self)

        del self

    def get_member(self, mtype: type[T], mindex: int = 0) -> T:
        """
        Get a member manager by class name.

        :param mtype: Type of the member manager.
        :param mindex: Index of the member.

        :return member: Member manager with the specific type.
        """
        return self.members_dict[mtype][mindex]

    def find_members_by_type(self, mtype: type[T]) -> list[T]:
        """
        Find member managers by class name.

        :param mtype: Type of the member manager.

        :return members: A list of member objects.
        """
        return self.members_dict[mtype]

    def print_info_all(self, include_self=True):
        """
        Print information about the object itself and all its members.
        It is used to debug only.
        """
        if include_self:
            self.print_info(title='DEBUG', object=self, members=self.members)

        for s in self.members:
            s.print_info(title='DEBUG', object=s,
                         manager=self, members=s.members)
            s.print_info_all(include_self=False)
