from typing import Sequence
from tiny_ecs import Application, ISystem, AppTag, Registry
from app.systems import (
    SAMSystem, ToolbarSystem, ViewportSystem, IOSystem
)

k_systems = [
    IOSystem(),
    ToolbarSystem(),
    SAMSystem(),
    ViewportSystem()
]

class SAM2SegmentationApp(Application):
    def __init__(self, systems: Sequence[ISystem]):
        """
        Initialize the application

        :param systems A list of system instances to use in the application
        """
        super().__init__("SAM2 Segmentation", 1500, 1000, systems)
        self.init()
    
    def run(self):
        """
        Run the application
        """
        while not self.request_shutdown:
            self.update()



if __name__ == "__main__":
    segm_app = SAM2SegmentationApp(k_systems)
    segm_app.run()
    segm_app.shutdown()