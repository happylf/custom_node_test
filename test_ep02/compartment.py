from .test_ep02 import *
from .supplement import *

class CN_IPAd1_Backg:
    @classmethod
    def INPUT_TYPES(s):        
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "in_posi": ("STRING", {"default": "", "multiline": True}),
                "in_nega": ("STRING", {"default": "", "multiline": True}),
                "CN_img": ("IMAGE",),
                "CN_name": (["None"]+folder_paths.get_filename_list("controlnet"),),
                "IPAd_img": ("IMAGE",),
                "IPAd_name": (["None"]+folder_paths.get_filename_list("ipadapter"), ),                
                "backg_img": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),     
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),                      
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }   
        }
    RETURN_TYPES=("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES=("image", "person_img", "mask_img")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02/Compartment"

    def todo(self, ckpt_name, in_posi, in_nega, CN_img, CN_name, IPAd_img, IPAd_name, backg_img, 
             seed, steps, cfg, sampler_name, scheduler, threshold, width, height, batch_size):  
        # sampler     
        set_IPAdapter = IPAdapter_set(IPAd_img, IPAd_name)
        motion_model = "None"
        pre_sampler = PreSampler_CN().todo(ckpt_name, in_posi, in_nega, CN_name, motion_model, 
            width, height, batch_size, CN_img, set_IPAdapter)[0]
        temp_img, _ = SamplerInsp(pre_sampler, seed, steps, cfg, sampler_name, scheduler, denoise=1.0)

        # ImageCompositeMasked
        person_img = img_scale_adjust(temp_img, width, height) 
        mask_img, _, _ = yolo_detect(person_img, threshold)
        backg_img = img_scale_adjust(backg_img, width, height)
        x = 30
        y = 30
        resize_source = False
        image = nodes_mask.ImageCompositeMasked().composite(backg_img, person_img, x, y, resize_source, mask_img)[0]

        return image, person_img, mask_img