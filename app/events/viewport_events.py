from tiny_ecs.ecs import Registry
from tiny_ecs.event import Event

import numpy as np

class EvntViewportClicked(Event):
    def __init__(self, reg: Registry, pos_x: int, pos_y: int, img: np.ndarray):
        super().__init__(reg)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.img_np = img