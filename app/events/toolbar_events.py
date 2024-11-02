from tiny_ecs.ecs import Registry
from tiny_ecs.event import Event


class EvntImportImage(Event):
    def __init__(self, reg: Registry, filepath: str):
        super().__init__(reg)
        self.filepath = filepath


class EvntImportDirectory(Event):
    def __init__(self, reg: Registry, dir: str):
        super().__init__(reg)
        self.dir = dir