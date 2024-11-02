from tiny_ecs.ecs import Component
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAMComp(Component):
    def __init__(self):
        # sam type
        self.sam_type_img = True

        # checkpoint info
        self.chkpt_root = ""
        self.chkpt_filename = ""
        
        # available configurations
        self.conf_root = ""
        self.conf_filenames = []
        self.conf_selected_idx = 0

        # predictor
        self.sam_predictor: SAM2ImagePredictor = None
        self.sam_inference_state = None
        #
        self.device_gpu = True

        # video
        self.tracking = False
        self.tracked = False
        self.frame_predictions = { }