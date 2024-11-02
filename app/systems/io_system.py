from tiny_ecs.ecs import Registry, ISystem
from tiny_ecs.comps import AppTag
from tiny_ecs.event import EventEmitter

from app.events import (
    EvntImportImage, EvntImportedImage, EvntImportDirectory, EvntImportedDirectory, EvntExportDirectory
)
from app.components import SAMComp, ViewportComp
from PIL import Image
import numpy as np
import glob
import os
from tqdm import tqdm

class IOSystem(ISystem):
    def __init__(self) -> ISystem:
        super().__init__()
    
    def init(self, reg: Registry) -> None:
        app_entt = reg.view(AppTag)[-1]
        event_emitter: EventEmitter = reg.get(EventEmitter, app_entt)
        
        # register io fns
        event_emitter.on(EvntImportImage, self.on_import_image)
        event_emitter.on(EvntImportDirectory, self.on_import_dir)
        event_emitter.on(EvntExportDirectory, self.on_export_dir)

    def on_import_image(self, evt: EvntImportImage):
        """
        Callback for responding to EvntImportImage events.

        :param evt The event to respond to.
        """
        app_entt = evt.reg.view(AppTag)[-1]
        event_emitter: EventEmitter = evt.reg.get(EventEmitter, app_entt)

        img = Image.open(evt.filepath)
        img_np = np.array(img.convert('RGB'))
        evt_out = EvntImportedImage(evt.reg, img_np)
        event_emitter.publish(evt_out)
    
    def on_import_dir(self, evt: EvntImportDirectory):
        """
        Callback for responding to EvntImportDirectoryu events.

        :param evt The event to respond to
        """
        app_entt = evt.reg.view(AppTag)[-1]
        event_emitter: EventEmitter = evt.reg.get(EventEmitter, app_entt)

        # just propagate the event because video 
        # tracking is a thing on its own
        evt_out = EvntImportedDirectory(evt.reg, evt.dir)
        event_emitter.publish(evt_out)

    def on_export_dir(self, evt: EvntExportDirectory):
        """
        """
        app_entt = evt.reg.view(AppTag)[-1]
        sam_comp: SAMComp = evt.reg.get(SAMComp, app_entt)
        export_loop = tqdm(sam_comp.frame_predictions, total=len(sam_comp.frame_predictions), desc='exporting predictions')
        if not os.path.exists(evt.dir):
            os.mkdir(evt.dir)

        for frame_id in export_loop:
            filename = f"{frame_id:05d}.png"
            filepath = os.path.join(evt.dir, filename)
            logits = sam_comp.frame_predictions[frame_id]
            logits[logits <= 0] = 0
            logits[logits > 0] = 255
            # output_mask = np.zeros_like(logits)
            # output_mask[logits > 0] = 255
            pil_image = Image.fromarray(logits.astype(np.uint8), mode='L')
            pil_image.save(filepath)

    def shutdown(self, reg: Registry) -> None:
        return super().shutdown(reg)