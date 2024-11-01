from typing import Sequence
from tiny_ecs import Application, ISystem, AppTag, Registry

k_systems = [

]

class SAM2SegmentationApp(Application):
    def __init__(self, systems: Sequence[ISystem]):
        super().__init__("SAM2 Segmentation", 1000, 1000, systems)
        self.init()
    
    def run(self):
        while not self.request_shutdown:
            self.update()

    
if __name__ == "__main__":
    segm_app = SAM2SegmentationApp(k_systems)
    segm_app.run()