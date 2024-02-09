import torch
import os       
import sys
import json
import math
from GPUtil import showUtilization as gpu_usage

import nodes 
import comfy.utils
import folder_paths

import impact.logics as logics
import impact_pack as impact

from .common import *

from animatediff.utils_model import get_available_motion_models

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

class Pre_transformation:
    @classmethod
    def INPUT_TYPES(s):        
        return {
            "required": {
                "cur_batch": ("INT", {"default": 0, "min": 0, "max": 10000}),              
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "posi_from": ("STRING", {"default": "", "multiline": True}),
                "posi_to": ("STRING", {"default": "", "multiline": True}),
                "in_nega": ("STRING", {"default": "", "multiline": True}),
                "CN_name": (["None"]+folder_paths.get_filename_list("controlnet"),),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "max_frame": ("INT", {"default": 1, "min": 1, "max": 10000}),
            },
            "optional": {
                "CN_img": ("IMAGE",),
                "IPAd_from_img": ("IMAGE",),
                "IPAd_to_img": ("IMAGE",),
            }
        }
    RETURN_TYPES=("PRESET01", "MODEL")
    FUNCTION = "todo"
    CATEGORY = "TestNode/TestEp02"

    def todo(self,cur_batch,ckpt_name,posi_from,posi_to,in_nega,CN_name,width,height,
             batch_size,max_frame,CN_img=None,IPAd_from_img=None, IPAd_to_img=None): 
        # load checkpoint model
        model, clip, vae = load_ckpt_model(ckpt_name)

        # prompt encoding           
        posi_to, nega_cond = PromptEncoding(clip, posi_to, in_nega) 
        posi_from, nega_cond = PromptEncoding(clip, posi_from, in_nega) 
        # ConditioningAverage
        to_weight = cur_batch / (max_frame - 1) 
        t = nodes.ConditioningAverage().addWeighted(posi_to, posi_from, to_weight)[0]

        t_c = t[0][0]
        t_p = t[0][1]['pooled_output']

        if IPAd_from_img != None:
            t_mask = torch.full((1, height, width), to_weight, dtype=torch.float32, device="cpu")
            f_mask = torch.full((1, height, width), 1-to_weight, dtype=torch.float32, device="cpu")

        for i in range(1, batch_size):
            to_weight = (cur_batch + i)/ (max_frame - 1)
            t = nodes.ConditioningAverage().addWeighted(posi_to, posi_from, to_weight)[0]

            t_c = torch.cat([t_c, t[0][0]], dim=0)
            t_p = torch.cat([t_p, t[0][1]['pooled_output']], dim=0)   

            if IPAd_from_img != None:
                t_m = torch.full((1, height, width), to_weight, dtype=torch.float32, device="cpu")
                f_m = torch.full((1, height, width), 1-to_weight, dtype=torch.float32, device="cpu")

                t_mask = torch.cat((t_mask, t_m), dim=0)
                f_mask = torch.cat((f_mask, f_m), dim=0)
        posi_cond = [[t_c, {"pooled_output": t_p}]]

        # apply controlnet to condition           
        if CN_name != "None":  
            posi_cond, nega_cond = cn_apply(posi_cond, nega_cond, CN_name, CN_img)
            '''
            h = CN_img.shape[1]
            w = CN_img.shape[2]
            area_cond = nodes.ConditioningSetArea().append(posi_cond, w, h, x, y, 1.0)
            posi_cond = nodes.ConditioningCombine().combine(posi_common, area_cond)[0]
            '''
        # apply IPAdapter
        if IPAd_from_img != None:
            IPAd_name = "ip-adapter-plus_sd15.bin"    
            set_IPAdapter = IPAdapter_set(IPAd_from_img, IPAd_name, 2)
            model_1 = IPAd_apply(set_IPAdapter, model, f_mask)
            set_IPAdapter = IPAdapter_set(IPAd_to_img, IPAd_name, 2)
            model=IPAd_apply(set_IPAdapter, model_1, t_mask)

        # latent
        latent_image = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]

        pre_sampler=(model, vae, posi_cond, nega_cond, latent_image)
         
        return pre_sampler, model

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

class Cond_batch:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "cond_from": ("CONDITIONING",), 
            "cond_to": ("CONDITIONING",), 
            "cur_batch": ("INT", {"default": 0, "min": 0, "max": 10000}),              
            "max_frame": ("INT", {"default": 1, "min": 1, "max": 10000}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "todo"

    CATEGORY = "TestNode/TestEp02/etc"

    def todo(self, cond_from, cond_to, cur_batch, max_frame, batch_size):
        c = cond_from[0][0]
        p = cond_from[0][1]['pooled_output']

        for i in range(1, batch_size):
            to_weight = (cur_batch + i)/ (max_frame - 1)
            t = nodes.ConditioningAverage().addWeighted(cond_to, cond_from, to_weight)[0]

            c = torch.cat([c, t[0][0]], dim=0)
            p = torch.cat([p, t[0][1]['pooled_output']], dim=0)   
        out_cond = [[c, {"pooled_output": p}]]

        return (out_cond,)    
    
class Puzzle15:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),                
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "todo"

    CATEGORY = "TestNode/TestEp02/etc"

    def todo(self, image):
        puzzle, aiMoves = puz15_init()

        h = image.shape[1]
        w = image.shape[2]
        crop_w = w/4
        crop_h = h/4
        crop_img = puz15_crop(image, crop_w, crop_h)
        mask = torch.full((1, h, w), 1, dtype=torch.float32, device="cpu")

        out_img = []
        count_down = len(aiMoves) - 1
        for m in aiMoves:
            puzzle.move(m)
            t_img = puz15_draw(image, crop_img, crop_w, crop_h, mask, puzzle, count_down)
            out_img.append(t_img)
            count_down -= 1 
        out_img = torch.cat(out_img, 0)    

        return (out_img,)       
