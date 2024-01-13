import torch
import os       
import sys
import json
from GPUtil import showUtilization as gpu_usage

import nodes 
import comfy.utils
import folder_paths

import impact.logics as logics
import impact_pack as impact

from .common import *

from animatediff.model_utils import get_available_motion_models

# IPAdapter_plus
import ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapter

MAX_RESOLUTION=8192

out_image = []

class PreSampler_CN:
    @classmethod
    def INPUT_TYPES(s):        
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "in_posi": ("STRING", {"default": "", "multiline": True}),
                "in_nega": ("STRING", {"default": "", "multiline": True}),
                "CN_name": (["None"]+folder_paths.get_filename_list("controlnet"),),
                "motion_model": (["None"]+get_available_motion_models(), {"default": "None"}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "CN_img": ("IMAGE",),
                "set_IPAdapter": ("SET_IPA",),
            }
        }
    RETURN_TYPES=("PRESET01", )
    RETURN_NAMES=("pre_sampler", )    
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02"

    def todo(self, ckpt_name, in_posi, in_nega, CN_name, motion_model, width, height, batch_size, 
             CN_img=None , set_IPAdapter=None): 
        # load checkpoint model
        model, clip, vae = load_ckpt_model(ckpt_name)

        # AnimatteDiff Loader/CLIPSetLastLayer
        if motion_model != "None":
            model, clip = ani_diff(model, clip, motion_model)

        # prompt encoding           
        posi_cond, nega_cond = PromptEncoding(clip, in_posi, in_nega) 

        # apply IPAdapter
        if set_IPAdapter != None:
            model = IPAd_apply(set_IPAdapter, model)

        # apply controlnet to condition
        if CN_name != "None":
            posi_cond, nega_cond = cn_apply(posi_cond, nega_cond, CN_name, CN_img, width, height)
        # latent
        latent_image = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]

        pre_sampler=(model, vae, posi_cond, nega_cond, latent_image)

        return(pre_sampler, )

class set_IPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ipadapter_file": (folder_paths.get_filename_list("ipadapter"), ),
                "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            }
        }       
    RETURN_TYPES = ("SET_IPA",)
    RETURN_NAMES = ("set_IPAdapter",)
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02"

    def todo(self, image, ipadapter_file, clip_name, weight, noise, weight_type, start_at, end_at):       
        ipadapter = IPAdapter.IPAdapterModelLoader().load_ipadapter_model(ipadapter_file)[0]
        clip_vision = nodes.CLIPVisionLoader().load_clip(clip_name)[0]

        out_set_IPAdapter = (ipadapter, weight, clip_vision, image, weight_type, noise, start_at, end_at)

        return (out_set_IPAdapter,)

class Yolo_Detector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image_list": ("IMAGE", ),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "labels": ("STRING", {"multiline": True, "default": "all", 
                "placeholder": "List the types of segments to be allowed, separated by commas"}),                            
            }
        }
    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("person_masks", "others_masks", "mask_color_img")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02/Detector"

    def todo(self, image_list, threshold, labels):      
        person_masks, others_masks, mask_color_img = yolo_detect(image_list, threshold)

        return (person_masks, others_masks, mask_color_img)

class MovingControl:
    @classmethod
    def INPUT_TYPES(s):        
        return {
            "required": {
                "moving_set": ("STRING", {"default": "", "multiline": True}),
                "curr_idx": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "image_limit": ("INT", {"default": 1, "min": 1, "max": 10000}),                                     
            }   
        }
    RETURN_TYPES=("INT", "INT", "INT")
    RETURN_NAMES=("x", "y", "r")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02/etc"

    def todo(self, moving_set, curr_idx, image_limit):
        moving_list = moving_calc(moving_set, image_limit) 
        x = moving_list[curr_idx][1]
        y = moving_list[curr_idx][2]
        r = moving_list[curr_idx][3]

        return x, y, r
    
class Edit_pre_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
            },    
            "optional": {
                "pre_sampler": ("PRESET01",),                   
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "samples":  ("LATENT",),
            }
        }
    RETURN_TYPES = ("PRESET01", "MODEL", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("pre_sampler", "model", "vae", "positive", "negative", "samples")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02/etc"

    def todo(self, pre_sampler=None, model=None, vae=None, positive=None, negative=None, samples=None):
        if pre_sampler != None:
            out_model, out_vae, out_positive, out_negative, out_samples = pre_sampler
        if model != None:
            out_model = model
        if vae != None:
            out_vae = vae
        if positive != None:
            out_positive = positive
        if negative != None:
            out_negative = negative
        if samples != None:
            out_samples = samples
        out_pre_sampler = out_model, out_vae, out_positive, out_negative, out_samples 

        return out_pre_sampler, out_model, out_vae, out_positive, out_negative, out_samples
        
class Sampler01:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "pre_sampler": ("PRESET01",),   
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),                                                      
            }
        }
    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02/Sampler"

    def todo(self, pre_sampler, seed, steps, cfg, sampler_name, scheduler, denoise): 
        images, samples = SamplerInsp(pre_sampler, seed, steps, cfg, sampler_name, scheduler, denoise)

        return (images, samples)    
    
class LoopDecision01:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "sum_val": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "send_val": ("INT", {"default": 16, "min": 1, "max": 10000}),
                "image_limit": ("INT", {"default": 16, "min": 1, "max": 10000}),
            }
        }
    RETURN_TYPES = ("INT", "INT", "IMAGE")
    RETURN_NAMES = ("sum_val", "send_val", "out_img")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02/etc"

    def todo(self, image, sum_val, send_val, image_limit): 
        torch.cuda.empty_cache()
        gpu_usage()

        global out_image
        if sum_val==0:
            out_image.clear()

        out_image.append(image)

        img_cnt = image.size(0)
        sum_val = img_cnt + sum_val
        if sum_val >= image_limit:
            out_img = torch.cat(out_image, 0)
            logics.ImpactConditionalStopIteration().doit(True)  # finish Auto-Queue
        else:
            out_img = image
            remain = image_limit - sum_val
            if remain < send_val:
               send_val = remain  
    
        return (sum_val, send_val, out_img)   
    
class debug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "moving_set": ("STRING", {"default": "", "multiline": True}),
                 "x": ("INT", {"default": 1, "min": 0, "max": 10000}),                 
            }
        }
    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("moving_set", "x")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02/etc"

    def todo(self, moving_set, x): 

        return moving_set, x
