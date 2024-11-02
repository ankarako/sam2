from tiny_ecs.ecs import Registry
from tiny_ecs.event import Event

import numpy as np

class EvntImportedImage(Event):
    def __init__(self, reg: Registry, img: np.ndarray):
        super().__init__(reg)
        self.img = img

class EvntImportedDirectory(Event):
    def __init__(self, reg: Registry, dir: str):
        super().__init__(reg)
        self.dir = dir

class EvntExportDirectory(Event):
    def __init__(self, reg: Registry, dir: str):
        super().__init__(reg)
        self.dir = dir
