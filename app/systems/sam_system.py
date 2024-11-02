from tiny_ecs.ecs import Registry, ISystem
from tiny_ecs.comps import AppTag
from tiny_ecs.event import EventEmitter
from app.components.sam_comp import SAMComp
from app.events import EvntViewportClicked, EvntSAMPredictedImg, EvntImportedDirectory, EvntSAMPredictedFrame, EvntSAMPredictedVideo

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

import os
import glob
import imgui
import numpy as np
import torch

k_config_chkpt_table = {
    'sam2.1_hiera_b+.yaml': 'sam2.1_hiera_base_plus.pt',
    'sam2.1_hiera_l.yaml': 'sam2.1_hiera_large.pt',
    'sam2.1_hiera_s.yaml': 'sam2.1_hiera_small.pt',
    'sam2.1_hiera_t.yaml': 'sam2.1_hiera_tiny.pt',
}

class SAMSystem(ISystem):
    def __init__(self):
        super().__init__()
    
    def init(self, reg: Registry) -> None:
        """
        Initialize the SAM system
        """
        # create our SAMComp
        app_entt = reg.view(AppTag)[-1]
        event_emitter: EventEmitter = reg.get(EventEmitter, app_entt)
        sam_comp = SAMComp()

        # get available checkpoints and configurations and
        # set default checkpoint and configuration paths
        cwd = os.getcwd()
        chkp_root = os.path.join(cwd, 'checkpoints', )
        sam_comp.chkpt_root = chkp_root

        conf_root = os.path.join(cwd, 'sam2', 'configs', 'sam2.1')
        conf_filepaths = glob.glob(os.path.join(conf_root, '*.yaml'))
        sam_comp.conf_root = conf_root
        for conf_filepath in conf_filepaths:
            sam_comp.conf_filenames += [os.path.basename(conf_filepath)]

        # select the basic configuration as default
        for idx, filename in enumerate(sam_comp.conf_filenames):
            if 'base' in filename:
                sam_comp.conf_selected_idx = idx
        sam_comp.chkpt_filename = k_config_chkpt_table[sam_comp.conf_filenames[sam_comp.conf_selected_idx]]
        reg.register(sam_comp, app_entt)

        # register callbacks
        event_emitter.on(EvntViewportClicked, self.on_viewport_clicked)
        event_emitter.on(EvntImportedDirectory, self.on_imported_dir)
        

    def load_sam(self, sam_comp: SAMComp) -> None:
        """
        Load the SAM model specified by the configuration in the specified 
        SAMComp object.
        """
        if sam_comp.sam_predictor is not None:
            sam_comp.sam_predictor = None

        print("Loading SAM")
        print(f"\tchkpt: {sam_comp.chkpt_filename}")
        print(f"\tconf: {sam_comp.conf_filenames[sam_comp.conf_selected_idx]}")
        print("\ttype: image" if sam_comp.sam_type_img else "\ttype: video")
        sam_chpt_filepath = os.path.join(sam_comp.chkpt_root, sam_comp.chkpt_filename)
        sam_conf_filepath = os.path.join('configs', 'sam2.1', sam_comp.conf_filenames[sam_comp.conf_selected_idx])
        if sam_comp.sam_type_img:
            sam_comp.sam_predictor = SAM2ImagePredictor(build_sam2(sam_conf_filepath, sam_chpt_filepath))
        else:
            device = torch.device('cuda') if sam_comp.device_gpu else torch.device('cpu')
            sam_comp.sam_predictor = build_sam2_video_predictor(sam_conf_filepath, sam_chpt_filepath, device)
        print("Loaded.")

    
    def on_viewport_clicked(self, evt: EvntViewportClicked):
        """
        Callback for predicting mask when viewport is clicked
        """
        app_entt = evt.reg.view(AppTag)[-1]
        event_emitter: EventEmitter = evt.reg.get(EventEmitter, app_entt)

        sam_comp: SAMComp = evt.reg.get(SAMComp, app_entt)

        if sam_comp.sam_type_img:
            # sam prediction type is image
            if sam_comp.sam_predictor is not None:
                with torch.inference_mode(), torch.autocast("cuda" if sam_comp.device_gpu else "cpu", dtype=torch.bfloat16):
                    sam_comp.sam_predictor.set_image(evt.img_np)
                    input_point = np.array([[evt.pos_x, evt.pos_y]])
                    input_label = np.array([1])
                    masks, scores, logits = sam_comp.sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                    )
                    event_emitter.publish(EvntSAMPredictedImg(evt.reg, masks))
        else:
            # sam prediction type is video tracking
            input_point = np.array([[evt.pos_x, evt.pos_y]])
            input_label = np.array([1])
            _, out_obj_ids, out_mask_logits = sam_comp.sam_predictor.add_new_points_or_box(
                inference_state=sam_comp.sam_inference_state,
                frame_idx=0,
                obj_id=1,
                points=input_point,
                labels=input_label
            )
            event_emitter.publish(EvntSAMPredictedImg(evt.reg, out_mask_logits[0][0].clone().cpu().numpy(), is_logits=True))
    
    def on_imported_dir(self, evt: EvntImportedDirectory) -> None:
        app_entt = evt.reg.view(AppTag)[-1]
        sam_comp: SAMComp = evt.reg.get(SAMComp, app_entt)
        if sam_comp.sam_predictor is not None:
            sam_comp.sam_inference_state = sam_comp.sam_predictor.init_state(video_path=evt.dir)
        else:
            print("Please load SAM model and re-import directory.")
    
    def update(self, reg: Registry) -> None:
        """
        Perform one system update step
        """
        # get our SAMComp from the app_entt
        app_entt = reg.view(AppTag)[-1]
        sam_comp: SAMComp = reg.get(SAMComp, app_entt)
        event_emitter: EventEmitter = reg.get(EventEmitter, app_entt)

        imgui.begin("SAM")

        # select SAM type, image or video
        imgui.text("SAM Type")
        if imgui.radio_button("Image", sam_comp.sam_type_img):
            sam_comp.sam_type_img = True

        imgui.same_line()
        if imgui.radio_button("Video", not sam_comp.sam_type_img):
            sam_comp.sam_type_img = False

        # select SAM device
        imgui.text("Device")
        if imgui.radio_button("CPU", not sam_comp.device_gpu):
            sam_comp.device_gpu = False
        
        imgui.same_line()
        if imgui.radio_button("GPU", sam_comp.device_gpu):
            sam_comp.device_gpu = True
        
        # select configurations
        with imgui.begin_combo('configs', sam_comp.conf_filenames[sam_comp.conf_selected_idx]) as conf_combo:
            if conf_combo.opened:
                for idx, conf_filename in enumerate(sam_comp.conf_filenames):
                    is_selected = (idx == sam_comp.conf_selected_idx)
                    if imgui.selectable(conf_filename, is_selected)[0]:
                        # modify selected configuration and checkpoint
                        sam_comp.conf_selected_idx = idx
                        sam_comp.chkpt_filename = k_config_chkpt_table[sam_comp.conf_filenames[sam_comp.conf_selected_idx]]
                        # TODO: throw event for re-initializing sam predictor

        if imgui.button("Load"):
            self.load_sam(sam_comp)

        if imgui.button("Track"):
            sam_comp.tracking = True
            for out_frame_id, out_obj_id, out_mask_logits in sam_comp.sam_predictor.propagate_in_video(sam_comp.sam_inference_state):
                mask_logits = out_mask_logits[0][0].clone().cpu().numpy()
                sam_comp.frame_predictions[out_frame_id] = mask_logits
            event_emitter.publish(EvntSAMPredictedVideo(reg))
            sam_comp.tracking = False
            sam_comp.tracked = True
        imgui.end()

            
    
    def shutdown(self, reg: Registry) -> None:
        pass