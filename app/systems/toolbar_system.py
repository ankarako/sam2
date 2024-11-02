from tiny_ecs.ecs import Registry, ISystem
from tiny_ecs.comps import AppTag
from tiny_ecs.event import EventEmitter

from app.events import EvntImportImage, EvntImportDirectory, EvntExportDirectory
from app.components import SAMComp, ViewportComp

import imgui
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

k_img_exts = [
    ("JPEG files", "*.jpg"),
    ("JPEG files", "*.JPG"),
    ("JPEG files", "*.jpeg"),
    ("JPEG files", "*.JPEG"),
    ("PNG files", "*.png"),
    ("PNG files", "*.PNG"),
    ("All files", "*.*")
]

class ToolbarSystem(ISystem):
    def __init__(self):
        super().__init__()
    
    def init(self, reg: Registry) -> None:
        pass
    
    def update(self, reg: Registry) -> None:
        # get event emitter
        app_entt = reg.view(AppTag)[-1]
        event_emitter: EventEmitter = reg.get(EventEmitter, app_entt)

        sam_comp: SAMComp = reg.get(SAMComp, app_entt)
        view_comp: ViewportComp = reg.get(ViewportComp, app_entt)
        
        with imgui.begin_main_menu_bar() as menu_bar:
            if menu_bar.opened:
                with imgui.begin_menu("File") as file_menu:
                    if file_menu.opened:
                        if imgui.menu_item('Import image...')[0]:
                            # open file dialog and fire event
                            img_filepath = filedialog.askopenfilename(filetypes=k_img_exts)
                            if len(img_filepath) != 0:
                                evt = EvntImportImage(reg, img_filepath)
                                event_emitter.publish(evt)
                            
                        if imgui.menu_item('Import video...')[0]:
                            # TODO: throw event for loading video
                            frames_dir = filedialog.askdirectory()
                            if len(frames_dir) != 0:
                                evt = EvntImportDirectory(reg, frames_dir)
                                event_emitter.publish(evt)

                        if imgui.menu_item('Export')[0]:
                            # TODO: throw event for exporting segmentations
                            if sam_comp.tracked and len(sam_comp.frame_predictions) > 0:
                                output_dir = filedialog.askdirectory()
                                if len(output_dir) != 0:
                                    evt = EvntExportDirectory(reg, output_dir)
                                    event_emitter.publish(evt)
                            pass
    
    def shutdown(self, reg: Registry) -> None:
        pass