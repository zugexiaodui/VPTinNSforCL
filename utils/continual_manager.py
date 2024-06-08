import random
import numpy as np
from copy import deepcopy
from typing import Literal


class ClassIncrementalManager():
    def __init__(self, class_list: list[int], num_tasks: int, rand_seed: int = 0, shuffle=True, shuffle_level: Literal['class', 'task'] = 'class'):
        assert len(class_list) % num_tasks == 0, f"{len(class_list)}, {num_tasks}"
        self.__rng = np.random.Generator(np.random.PCG64(rand_seed))

        self.__task_class_list = np.array(class_list, dtype=np.int64)
        if shuffle:
            if shuffle_level == 'class':
                self.__rng.shuffle(self.__task_class_list)
                self.__task_class_list = np.reshape(self.__task_class_list, [num_tasks, -1])
            elif shuffle_level == 'task':
                self.__task_class_list = np.reshape(self.__task_class_list, [num_tasks, -1])
                self.__rng.shuffle(self.__task_class_list)
        else:
            self.__task_class_list = np.reshape(self.__task_class_list, [num_tasks, -1])

        self.__all_classes = self.__task_class_list.flatten().tolist()
        self.__task_class_list = self.__task_class_list.tolist()

        self.__current_taskid = -1
        self.__num_tasks = num_tasks

        self.storage = {}

    @property
    def current_taskid(self) -> int:
        assert self.__current_taskid >= 0, "Not initialized"
        return self.__current_taskid

    @property
    def all_classes(self) -> list[int]:
        return self.__all_classes

    @property
    def task_class_list(self) -> list[list[int]]:
        return self.__task_class_list

    @property
    def num_tasks(self) -> int:
        return self.__num_tasks

    @property
    def num_classes_per_task(self) -> int:
        return len(self.__task_class_list[0])

    @property
    def current_task_classes(self) -> list[int]:
        return deepcopy(self.__task_class_list[self.current_taskid])

    @property
    def sofar_task_classes(self) -> list[list[int]]:
        extend = True
        classes = []
        for i in range(self.current_taskid + 1):
            if extend:
                classes.extend(self.__task_class_list[i])
            else:
                classes.append(self.__task_class_list[i])
        return deepcopy(classes)

    def get_classes(self, taskid: int) -> list[int]:
        return self.__task_class_list[taskid]

    def __iter__(self):
        return self

    def __next__(self):
        self.__current_taskid += 1
        if self.current_taskid >= len(self):
            self.__current_taskid = -1
            raise StopIteration()
        return self.current_taskid, self.current_task_classes

    def __len__(self) -> int:
        return self.num_tasks
