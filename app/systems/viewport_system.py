from tiny_ecs.ecs import Registry, ISystem
from tiny_ecs.comps import AppTag
from tiny_ecs.event import EventEmitter

from app.events import (
    EvntImportedImage, EvntViewportClicked, EvntSAMPredictedImg, EvntImportedDirectory, EvntSAMPredictedFrame, EvntSAMPredictedVideo
)
from app.components import ViewportComp, SAMComp
import imgui

import numpy as np
import OpenGL.GL as gl
import cv2
import glob
import os
from PIL import Image

def create_gl_texture_np_array(img: np.ndarray) -> int:
    """
    Create an OpenGL texture from the specified numpy array.

    :param img An np.ndarray specifying the image to upload.
    """
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    # set texture params
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    # upload image data to the texture
    img_height, img_width, img_channels = img.shape
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img_width, img_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img)

    # unbind texture
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id


def delete_gl_texture(texture_id: int) -> None:
    """
    Delete the specified GL texture
    """
    if texture_id != 0:
        gl.glDeleteTextures([texture_id])


def update_gl_texture(texture_id: int, img: np.ndarray) -> None:
    """
    Update the specified GL texture with the specified image.

    :param texture_id The GL texture id to update.
    :param img The image to upload.
    """
    if texture_id != 0:
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        img_height, img_width, img_channels = img.shape
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img_width, img_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


class ViewportSystem(ISystem):
    def __init__(self):
        super().__init__()

    def init(self, reg: Registry) -> None:
        """
        Initialize the system
        """
        # register events
        app_entt = reg.view(AppTag)[-1]
        event_emitter: EventEmitter = reg.get(EventEmitter, app_entt)

        view_comp = ViewportComp()
        reg.register(view_comp, app_entt)

        # imported image event
        event_emitter.on(EvntImportedImage, self.on_imported_img)
        event_emitter.on(EvntSAMPredictedImg, self.on_predicted_img)
        event_emitter.on(EvntImportedDirectory, self.on_imported_dir)
        event_emitter.on(EvntSAMPredictedVideo, self.on_predicted_video)
        event_emitter.on(EvntSAMPredictedFrame, self.on_predicted_frame)
        
        
    def on_imported_img(self, evt: EvntImportedImage) -> None:
        """
        Callback for responding to EvntImportedImage events
        """
        app_entt = evt.reg.view(AppTag)[-1]
        view_comp: ViewportComp = evt.reg.get(ViewportComp, app_entt)
        if view_comp.curr_tex_id != 0:
            delete_gl_texture(view_comp.curr_tex_id)

        view_comp.curr_img_np = evt.img
        view_comp.curr_tex_id = create_gl_texture_np_array(evt.img)

        # set correct aspect ratio
        height, width, c = evt.img.shape
        aspect = float(height) / float(width)
        view_comp.aspect = aspect
    
    def on_imported_dir(self, evt: EvntImportedDirectory) -> None:
        """
        Callback for imported directory
        """
        app_entt = evt.reg.view(AppTag)[-1]
        view_comp: ViewportComp = evt.reg.get(ViewportComp, app_entt)
        if view_comp.curr_tex_id != 0:
            delete_gl_texture(view_comp.curr_tex_id)
        
        # load first frame
        frame_filepaths = sorted(glob.glob(os.path.join(evt.dir, '*.jpg')))
        view_comp.frame_filepaths = frame_filepaths
        
        img = Image.open(frame_filepaths[0])
        img_np = np.array(img.convert('RGB'))
        view_comp.curr_img_np = img_np
        # load frame
        view_comp.curr_tex_id = create_gl_texture_np_array(img_np)

        # set correct aspect ratio
        height, width, c = img_np.shape
        aspect = float(height) / float(width)
        view_comp.aspect = aspect

    def update_point_prompt(self, view_comp: ViewportComp, pos_x: int, pos_y: int) -> None:
        """
        Update the viewport image with the selected point prompt

        :param view_comp The ViewportComp
        :param pos_x the clicked pos x
        :param pos_y the clicked pos y
        """
        img = view_comp.curr_img_np.copy()
        cv2.circle(img, (pos_x, pos_y), 20, (255, 0, 0), thickness=-1)
        update_gl_texture(view_comp.curr_tex_id, img)

    
    def on_predicted_img(self, evt: EvntSAMPredictedImg) -> None:
        """
        Callback for presenting SAM predictions
        """
        app_entt = evt.reg.view(AppTag)[-1]
        view_comp: ViewportComp = evt.reg.get(ViewportComp, app_entt)
        if not evt.mask_is_logits:
            view_comp.mask_overlays = evt.masks

            # by default show the first prediction
            mask = view_comp.mask_overlays[view_comp.mask_selected_idx]
            img = view_comp.curr_img_np.copy()
            mask_colored = np.zeros_like(img)
            mask_colored[mask > 0, :] = view_comp.overlay_color * 255
            overlayed = cv2.addWeighted(img, 1 - view_comp.alpha, mask_colored, view_comp.alpha, 0)
            update_gl_texture(view_comp.curr_tex_id, overlayed)
        else:
            view_comp.mask_overlays = evt.masks[None, ...]
            mask = evt.masks
            mask[mask <= 0] = 0
            img = view_comp.curr_img_np.copy()
            mask_colored = np.zeros_like(img)
            mask_colored[mask > 0, :] = view_comp.overlay_color * 255
            overlayed = cv2.addWeighted(img, 1 - view_comp.alpha, mask_colored, view_comp.alpha, 0)
            update_gl_texture(view_comp.curr_tex_id, overlayed)
    
    def on_predicted_frame(self, evt: EvntSAMPredictedFrame) -> None:
        app_entt = evt.reg.view(AppTag)[-1]
        view_comp: ViewportComp = evt.reg.get(ViewportComp, app_entt)
        sam_comp: SAMComp = evt.reg.get(SAMComp, app_entt)

        logits = sam_comp.frame_predictions[view_comp.curr_tracked_frame_id]
        logits[logits <= 0] = 0
        img = Image.open(view_comp.frame_filepaths[view_comp.curr_tracked_frame_id])
        img_np = np.array(img.convert('RGB'))
        mask_colored = np.zeros_like(img_np)
        mask_colored[logits > 0, :] = view_comp.overlay_color * 255
        overlayed = cv2.addWeighted(img_np, 1 - view_comp.alpha, mask_colored, view_comp.alpha, 0)
        update_gl_texture(view_comp.curr_tex_id, overlayed)



    def on_predicted_video(self, evt: EvntSAMPredictedVideo) -> None:
        app_entt = evt.reg.view(AppTag)[-1]
        view_comp: ViewportComp = evt.reg.get(ViewportComp, app_entt)
        sam_comp: SAMComp = evt.reg.get(SAMComp, app_entt)
        
    
    def update_prediction_overlay(self, view_comp: ViewportComp) -> None:
        mask = view_comp.mask_overlays[view_comp.mask_selected_idx]
        img = view_comp.curr_img_np.copy()
        mask_colored = np.zeros_like(img)
        mask_colored[mask > 0, :] = view_comp.overlay_color * 255
        overlayed = cv2.addWeighted(img, 1 - view_comp.alpha, mask_colored, view_comp.alpha, 0)
        update_gl_texture(view_comp.curr_tex_id, overlayed)
        
    
    def update(self, reg: Registry) -> None:
        """
        Perform one system update step
        """
        app_entt = reg.view(AppTag)[-1]
        event_emitter: EventEmitter = reg.get(EventEmitter, app_entt)
        view_comp: ViewportComp = reg.get(ViewportComp, app_entt)

        imgui.begin("Viewport", False, imgui.WINDOW_NO_MOVE)
        cursor_pos = imgui.get_cursor_screen_pos()
        # could calculate height here according to toolbar
        # but OK, it's a very simple app
        imgui.set_window_position(0, 18)
        view_comp.width = imgui.get_window_width()
        view_comp.height = view_comp.width * view_comp.aspect
        imgui.set_window_size(view_comp.width, view_comp.height)

        if view_comp.curr_tex_id != 0:
            imgui.image(view_comp.curr_tex_id, view_comp.width, view_comp.height)

            # update image scale
            view_comp.scale = 1 / (view_comp.height / view_comp.curr_img_np.shape[0])

            if imgui.is_item_clicked(0):
                
                mouse_pos = imgui.get_mouse_pos()
                rel_x = mouse_pos.x - cursor_pos.x
                rel_y = mouse_pos.y - cursor_pos.y

                if 0 <= rel_x < view_comp.curr_img_np.shape[1] / view_comp.scale and 0 <= rel_y < view_comp.curr_img_np.shape[0] / view_comp.scale:
                    # set clicked image position at image coordinates
                    # correct image scale
                    clicked_pos_x = int(rel_x * view_comp.scale)
                    clicked_pos_y = int(rel_y * view_comp.scale)
                    # print(f"img shape: ({view_comp.curr_img_np.shape[1]}, {view_comp.curr_img_np.shape[0]})")
                    # print(f"mouse pos: ({clicked_pos_x}, {clicked_pos_y})")
                    
                    # Show selected point on viewport
                    self.update_point_prompt(view_comp, clicked_pos_x, clicked_pos_y)
                    evt = EvntViewportClicked(reg, clicked_pos_x, clicked_pos_y, view_comp.curr_img_np)
                    event_emitter.publish(evt)
        imgui.end()

        # mask selection tools on another window
        imgui.begin("Overlay settings")
        if view_comp.mask_overlays is not None:
            npreds = view_comp.mask_overlays.shape[0]
            prediction_list = list(range(npreds))
            imgui.set_next_item_width(30)
            with imgui.begin_combo('predictions', str(prediction_list[view_comp.mask_selected_idx])) as pred_combo:
                if pred_combo.opened:
                    for idx, pred_id in enumerate(prediction_list):
                        is_selected = (idx == view_comp.mask_selected_idx)
                        if imgui.selectable(str(pred_id), is_selected)[0]:
                            view_comp.mask_selected_idx = idx
                            self.update_prediction_overlay(view_comp)

            imgui.same_line()
            imgui.set_next_item_width(50)
            color_updated, new_color = imgui.color_edit3("overlay color", *view_comp.overlay_color)
            if color_updated:
                view_comp.overlay_color = np.array(new_color)
                self.update_prediction_overlay(view_comp)

            imgui.same_line()
            imgui.set_next_item_width(100)
            alpha_updated, alpha = imgui.slider_float("alpha", view_comp.alpha, 0.0, 1.0)
            if alpha_updated:
                view_comp.alpha = alpha
                self.update_prediction_overlay(view_comp)
        imgui.end()

        sam_comp: SAMComp = reg.get(SAMComp, app_entt)
        imgui.begin("Video sequencer")
        if sam_comp.tracked:
            nframes = len(sam_comp.frame_predictions)
            tracked_frame_id_updated, tracked_frame_id = imgui.slider_int("frames", view_comp.curr_tracked_frame_id, 0, nframes - 1)
            if tracked_frame_id_updated:
                view_comp.curr_tracked_frame_id = tracked_frame_id
                event_emitter.publish(EvntSAMPredictedFrame(reg, sam_comp.frame_predictions[tracked_frame_id], tracked_frame_id))
        imgui.end()
    
    def shutdown(self, reg: Registry) -> None:
        return super().shutdown(reg)