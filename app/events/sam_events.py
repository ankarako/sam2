from tiny_ecs.ecs import Registry
from tiny_ecs.event import Event

import numpy as np

class EvntSAMPredictedImg(Event):
    def __init__(self, reg: Registry, masks: np.ndarray, is_logits: bool=False):
        super().__init__(reg)
        self.masks = masks
        self.mask_is_logits = is_logits

class EvntSAMPredictedFrame(Event):
    def __init__(self, reg: Registry, logits: np.ndarray, frame_id: int):
        super().__init__(reg)
        self.logits = logits
        self.frame_id = frame_id

class EvntSAMPredictedVideo(Event):
    def __init__(self, reg: Registry):
        super().__init__(reg)