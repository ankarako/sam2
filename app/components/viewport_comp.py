from tiny_ecs.comps import Component
import numpy as np

class ViewportComp(Component):
    def __init__(self):
        self.width = 500
        self.height = 500
        self.curr_tex_id = 0
        self.curr_img_np = None
        self.aspect = 1.0
        self.scale = 1.0

        # predicted masks
        self.mask_overlays = None
        self.mask_selected_idx = 0
        self.overlay_color = np.array([1.0, 0, 0])
        self.alpha = 0.5

        # for video frames
        self.frame_filepaths = []
        self.curr_tracked_frame_id = 0