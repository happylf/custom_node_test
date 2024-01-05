import torch
from GPUtil import showUtilization as gpu_usage

import nodes 
import folder_paths

import os       
import sys

from .supplement import *

import comfy_extras.nodes_mask as nodes_mask

import impact.core as core
import impact.subcore as subcore
from segment_anything import sam_model_registry
from ultralytics import YOLO

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

# Inspire-Pack 
Inspire_path = os.path.join(custom_nodes_path, "ComfyUI-Inspire-Pack")
sys.path.append(Inspire_path)
import inspire.a1111_compat as Inspire_a1111 

# AnimateDiff, ControlNet import
AniDiff_path = os.path.join(custom_nodes_path, "ComfyUI-AnimateDiff-Evolved")
Controlnet_path  = os.path.join(custom_nodes_path, "ComfyUI-Advanced-ControlNet")
sys.path.append(AniDiff_path)
sys.path.append(Controlnet_path)
import animatediff.nodes as AD_nodes
import control.nodes as control_nodes

# IPAdapter_plus
import ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapter

def load_ckpt_model(ckpt_name):
    out=nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name, output_vae=True, output_clip=True)
    out_model = out[0]
    out_clip = out[1] 
    out_vae = out[2]

    return out_model, out_clip, out_vae

def ani_diff(ckpt_model, ckpt_clip, motion_model):
    # CLIPSetLastLayer
    out_clip = ckpt_clip.clone()
    out_clip.clip_layer = (-2)

    # AnimateDiffUniformContextOptions
    context_length = 16
    context_stride = 1
    context_overlap = 4
    context_schedule = 'uniform'
    closed_loop = False
    context_options = AD_nodes.AnimateDiffUniformContextOptions().create_options(context_length,
                            context_stride, context_overlap, context_schedule, closed_loop)[0]

    # AnimatteDiff Loader
    beta_schedule = 'sqrt_linear (AnimateDiff)'
    motion_lora = None
    motion_model_settings = None
    motion_scale = 1
    apply_v2_models_properly = False

    out_model = AD_nodes.AnimateDiffLoaderWithContext().load_mm_and_inject_params( 
                    ckpt_model, motion_model, beta_schedule, context_options, motion_lora, 
                    motion_model_settings, motion_scale, apply_v2_models_properly)[0]

    return out_model, out_clip

def PromptEncoding(clip, in_posi, in_nega):
    in_posi = "masterpiece, best quality, high resolution," + in_posi 
    in_nega = "deformed, bad quality, bad anatomy, bad hand, extra fingers," + in_nega

    out_posi_cond = nodes.CLIPTextEncode().encode(clip, in_posi)[0]
    out_nega_cond = nodes.CLIPTextEncode().encode(clip, in_nega)[0]    

    return out_posi_cond, out_nega_cond

def IPAd_apply(set_IPAdapter, in_model):
    ipadapter,weight,clip_vision,image,weight_type,noise,start_at,end_at=set_IPAdapter       
   
    # Apply IPAdapter to model
    embeds = None
    attn_mask = None
    unfold_batch=False
    out_model=IPAdapter.IPAdapterApply().apply_ipadapter(ipadapter,in_model,weight,
        clip_vision,image,weight_type,noise, embeds,attn_mask,start_at,end_at,unfold_batch)[0]

    return out_model

def cn_apply(in_posi_cond, in_nega_cond, CN_name, CN_img, width, height):
    CN_img = img_scale_adjust(CN_img, width, height)

    strength = 1.0
    start_percent = 0.0
    end_percent = 1.0
    '''
    CN = nodes.ControlNetLoader().load_controlnet(CN_name)[0]   
    out_posi_cond, out_nega_cond = nodes.ControlNetApplyAdvanced().apply_controlnet(
        in_posi_cond,in_nega_cond,CN,CN_img,strength,start_percent,end_percent)
    '''
    # Load ControlNet Model (Advanced)
    timestep_keyframe = None
    CN = control_nodes.ControlNetLoaderAdvanced().load_controlnet(CN_name,timestep_keyframe)[0]  
    mask_optional = None
    out_posi_cond, out_nega_cond=control_nodes.AdvancedControlNetApply().apply_controlnet(
        in_posi_cond,in_nega_cond,CN,CN_img,strength,start_percent,end_percent,mask_optional)

    return out_posi_cond, out_nega_cond

def yolo_detect(image_list, threshold):
        # from UltralyticsDetectorProvider
        model_name = "segm/yolov8m-seg.pt"
        model_path = folder_paths.get_full_path("ultralytics", model_name)
        model = subcore.load_yolo(model_path)
        segm_detector = subcore.UltraSegmDetector(model)

        dilation = 10
        crop_factor = 3.0
        drop_size = 10
        detailer_hook = None       
 
        person_masks_list = []
        others_masks_list = []
        mask_color_img_list = []

        for image in image_list:
            image = image.unsqueeze(0)
            segs = segm_detector.detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook)

            # SEGSLabelFilter
            person_segs = []
            others_segs = []

            for x in segs[1]:
                if x.label == 'person':
                    person_segs.append(x)
                else:    
                    others_segs.append(x)

            person_masks = core.segs_to_combined_mask((segs[0], person_segs))
            others_masks = core.segs_to_combined_mask((segs[0], others_segs))

            person_masks_list.append(person_masks)            
            others_masks_list.append(others_masks)            
 
            height = segs[0][0]
            width = segs[0][1]
            x = y = 0
            resize_source = False
            d=nodes.EmptyImage().generate(width,height,batch_size=1,color=255)[0]      #blue
            s=nodes.EmptyImage().generate(width,height,batch_size=1,color=16711680)[0] #red - person
            temp_image = nodes_mask.ImageCompositeMasked().composite(d, s, x, y, 
                                      resize_source, person_masks)[0]
            s=nodes.EmptyImage().generate(width,height,batch_size=1,color=65280)[0]    #green - others
            mask_color_img=nodes_mask.ImageCompositeMasked().composite(
                temp_image, s, x, y, resize_source, others_masks)[0]
                 
            mask_color_img_list.append(mask_color_img)

        # list to batch masks
        out_person_masks = torch.stack(person_masks_list, dim=0)               
        out_others_masks = torch.stack(others_masks_list, dim=0)            
        out_mask_color_img = torch.cat(mask_color_img_list, 0)

        return out_person_masks, out_others_masks, out_mask_color_img

def Sampler(pre_sampler, seed, steps, cfg, sampler_name, scheduler, denoise):
    torch.cuda.empty_cache()
    gpu_usage()

    model, vae, positive, negative, latent_image = pre_sampler
    samples = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, 
                            positive, negative, latent_image, denoise)[0]  
    images = nodes.VAEDecode().decode(vae, samples)[0]
    images = Reactor_apply(images)

    del pre_sampler, model, vae, positive, negative, latent_image
    return images, samples

def SamplerInsp(pre_sampler, seed, steps, cfg, sampler_name, scheduler, denoise):
    torch.cuda.empty_cache()
    gpu_usage()

    model, vae, positive, negative, latent_image = pre_sampler

    noise_mode = "GPU(=A1111)"
    batch_seed_mode="comfy"    
    samples = Inspire_a1111.KSampler_inspire().sample(model, seed, steps, cfg, sampler_name, 
        scheduler, positive, negative, latent_image, denoise, noise_mode, batch_seed_mode,
        variation_seed=None, variation_strength=None)[0]
    
    images = nodes.VAEDecode().decode(vae, samples)[0]
    images = Reactor_apply(images)
    
    del pre_sampler, model, vae, positive, negative, latent_image
    return images, samples

def img_scale_adjust(image, width, height):
    upscale_method = "nearest-exact"
    crop = "disabled"
    image = nodes.ImageScale().upscale(image, upscale_method, width, height, crop)[0]

    return image

def IPAdapter_set(IPAd_img, IPAd_name):
    IPAd_model = IPAdapter.IPAdapterModelLoader().load_ipadapter_model(IPAd_name)[0]
    clip_name = "sd1.5 model.safetensors"
    clip_vision = nodes.CLIPVisionLoader().load_clip(clip_name)[0]   
    weight = 1.0
    weight_type = "original"
    noise = 0.0
    start_at = 0.0
    end_at = 1.0
    set_IPAdapter = (IPAd_model, weight, clip_vision, IPAd_img, weight_type, noise, start_at, end_at)

    return set_IPAdapter 