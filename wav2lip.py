import sys
import numpy as np
import torch

wav2lip_path = "D:\lip_sync\Wav2Lip"
sys.path.append(wav2lip_path)
from wav2lip_node import wav2lip_   

class Wav2Lip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "images": ("IMAGE", ),                
                "audio_path": ("STRING", {"default": "D:\\lip_sync\\audio.wav"}),
                "mode": (["sequential", "repetitive"], {"default": "sequential"}),
                "face_detect_batch": ("INT", {"default": 8, "min": 1, "max": 100}),
            }
        }
    RETURN_TYPES = ("IMAGE", "VHS_AUDIO",)
    RETURN_NAMES = ("images", "audio_path",)   
    FUNCTION = "todo"
    CATEGORY = "eNode/etc"

    def todo(self, images, audio_path, mode, face_detect_batch): 
        in_img_list = []        
        for i in images:
            in_img = i.numpy().squeeze()
            in_img = (in_img * 255).astype(np.uint8)
            in_img_list.append(in_img)

        images = wav2lip_(in_img_list, audio_path, face_detect_batch, mode)
        del in_img, in_img_list

        out_img_list = []
        for i in images:
            out_img = i.astype(np.float32) / 255.0
            out_img = torch.from_numpy(out_img)
            out_img_list.append(out_img)

        images = torch.stack(out_img_list, dim=0)
        del out_img, out_img_list

        return (images, audio_path,)    
    
NODE_CLASS_MAPPINGS = {
    "Wav2Lip":                  Wav2Lip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip":                  "Wav2Lip",
}    