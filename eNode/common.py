import torch
import json

from .supplement import *

import comfy_extras.nodes_mask as nodes_mask

import impact.core as core
import impact.detectors as detectors 
import impact.impact_pack as impact_pack
import impact.segs_nodes as segs_nodes
import impact.subcore as subcore
import impact.subpack_nodes as subpack
import impact.animatediff_nodes as impact_ani
from segment_anything import sam_model_registry
from ultralytics import YOLO

# AnimateDiff, ControlNet import
AniDiff_path = os.path.join(custom_nodes_path, "ComfyUI-AnimateDiff-Evolved")
Controlnet_path  = os.path.join(custom_nodes_path, "ComfyUI-Advanced-ControlNet")
sys.path.append(AniDiff_path)
sys.path.append(Controlnet_path)
import animatediff.nodes_gen2 as AD_gen2
import animatediff.nodes_context as AD_context
from animatediff.utils_model import get_available_motion_models

import adv_control.nodes as control_nodes

# DWPose
import comfyui_controlnet_aux.node_wrappers.dwpose as DWPose

# IPAdapterplus
import ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapter

def loadCkptModel(ckpt_name):
    out=nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name, output_vae=True, output_clip=True)
    out_model = out[0]
    out_clip = out[1] 
    out_vae = out[2]

    # CLIPSetLastLayer
    stop_at_clip_layer = -1
    out_clip = nodes.CLIPSetLastLayer().set_last_layer(out_clip, stop_at_clip_layer)[0]

    return out_model, out_clip, out_vae

def condBatch(cond_from, cond_to, batch_size):
    c = cond_from[0][0]
    p = cond_from[0][1]['pooled_output']

    for i in range(1, batch_size):
        to_weight = i/(batch_size-1)
        t = nodes.ConditioningAverage().addWeighted(cond_to, cond_from, to_weight)[0]

        c = torch.cat([c, t[0][0]], dim=0)
        p = torch.cat([p, t[0][1]['pooled_output']], dim=0)   
    out_cond = [[c, {"pooled_output": p}]]

    return out_cond

def aniDiff(ckpt_model, motion_model, context_options, context_length, context_overlap):
    # LoadAnimateDiffModel
    ad_settings = None
    motion_model = AD_gen2.LoadAnimateDiffModelNode().load_motion_model(motion_model,ad_settings)[0]

    # ApplyAnimateDiffModel
    start_percent = 0.0
    end_percent = 1.0
    motion_lora=ad_keyframes=scale_multival=effect_multival=prev_m_models=None
    m_models = AD_gen2.ApplyAnimateDiffModelNode().apply_motion_model(motion_model, 
                    start_percent, end_percent, motion_lora, ad_keyframes,
                    scale_multival, effect_multival, prev_m_models,)[0]
    
    # ContextOptions
    context_stride = 1
    closed_loop = False
    fuse_method = "pyramid"
    use_on_equal_length=False
    start_percent = 0.0
    guarantee_steps =1
    prev_context=view_opts=None
    match context_options:
        case 'StandardUniform':
            context_options = AD_context.StandardUniformContextOptionsNode().create_options(
                context_length,context_stride,context_overlap,fuse_method,use_on_equal_length,
                start_percent,guarantee_steps,view_opts, prev_context)[0]
        case 'StandardStatic':
            context_options = AD_context.StandardStaticContextOptionsNode().create_options(
                context_length,context_overlap,fuse_method,use_on_equal_length,
                start_percent,guarantee_steps,view_opts, prev_context)[0]
        case 'LoopedUniform':
            context_options = AD_context.LoopedUniformContextOptionsNode().create_options(
                context_length,context_stride,context_overlap,closed_loop,fuse_method,use_on_equal_length,
                start_percent,guarantee_steps,view_opts, prev_context)[0]
        case 'Batched [Non-AD]':
            context_options = AD_context.BatchedContextOptionsNode().create_options(
                context_length, start_percent, guarantee_steps, prev_context)[0]

    # UseEvolvedSampling
    beta_schedule = 'sqrt_linear (AnimateDiff)'
    sample_settings=beta_schedule_override=None
    out_model = AD_gen2.UseEvolvedSamplingNode().use_evolved_sampling(
        ckpt_model, beta_schedule, m_models, context_options,
        sample_settings, beta_schedule_override)[0]

    return out_model

def contNetApply(contnet_img,control_net_name,width,height,posi_cond,nega_cond):
    contnet_img = imgResize_(contnet_img, width, height)
    timestep_keyframe = None
    control_net=control_nodes.ControlNetLoaderAdvanced().load_controlnet(control_net_name,timestep_keyframe)[0]

    # ApplyControlNet
    strength = 1.0  
    start_percent = 0.0
    end_percent = 1.0
    mask_optional=model_optional=timestep_kf=latent_kf_override=weights_override=None
    posi_cond,nega_cond,_=control_nodes.AdvancedControlNetApply().apply_controlnet(
        posi_cond,nega_cond,control_net,contnet_img,strength,start_percent,end_percent,
        mask_optional,model_optional,timestep_kf,latent_kf_override,weights_override)

    return posi_cond, nega_cond

'''
def makeDWPose(image, resolution):
    detect_hand=detect_body=detect_face="enable"
    bbox_detector="yolox_l.onnx"
    pose_estimator="dw-ll_ucoco_384.onnx"
    kwargs={}
    dwpose = DWPose.DWPose_Preprocessor().estimate_pose(image,detect_hand,detect_body,
                detect_face,resolution,bbox_detector,pose_estimator,**kwargs)

    return dwpose['result'][0]
'''

def applyIPAdapAdv(image,weight,weight_type,start_at,end_at,model,ipadapter,image_negative,attn_mask):
    preset = 'PLUS (high strength)'
    lora_strength=0.0
    provider="CPU"
    model,ipadapter=IPAdapter.IPAdapterUnifiedLoader().load_models(model,preset,lora_strength,provider,ipadapter)
    
    combine_embeds="concat"
    weight_faceidv2=None
    clip_vision=None
    insightface=None
    embeds_scaling='V only'
    weight_style=1.0
    weight_composition=1.0
    expand_style=False,
    image_style=None
    image_composition=None 
    model=IPAdapter.IPAdapterAdvanced().apply_ipadapter(model,ipadapter,start_at,end_at,weight,weight_style,weight_composition,expand_style,weight_type,combine_embeds,weight_faceidv2,image,image_style,image_composition, image_negative,clip_vision,attn_mask,insightface,embeds_scaling)[0]

    return model,ipadapter

def detectPerson(images, scale_by, option):
    # from UltralyticsDetectorProvider
    model_name = "segm/person_yolov8m-seg.pt"
    bbox_detector, _ = subpack.UltralyticsDetectorProvider().doit(model_name)

    # SAMLoader
    model_name = "sam_vit_b_01ec64.pth"
    device_mode = "AUTO"
    sam_model_opt = impact_pack.SAMLoader().load_model(model_name, device_mode)[0]

    bbox_threshold = 0.5
    bbox_dilation = 0
    crop_factor = 3.0
    drop_size = 10
    sub_threshold = 0.50
    sub_dilation = 0
    sub_bbox_expansion = 0
    sam_mask_hint_threshold = 0.70
    post_dilation = 0
    segm_detector_opt = None
    detailer_hook=None
    if option == "aniDiff":
        segs = detectors.SimpleDetectorForAnimateDiff().doit(bbox_detector,images, bbox_threshold, bbox_dilation, crop_factor,drop_size, sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold, masking_mode="Pivot SEGS",segs_pivot="Combined mask", sam_model_opt=None, segm_detector_opt=None)[0]
    else:    
        segs =  detectors.SimpleDetectorForEach().detect(bbox_detector, images, bbox_threshold, bbox_dilation, crop_factor, drop_size, sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold, post_dilation, sam_model_opt, segm_detector_opt, detailer_hook)[0]

    # Upscale Image By
    upscale_method = "lanczos"
    images = nodes.ImageScaleBy().upscale(images, upscale_method, scale_by)[0]

    bbox_size = int(images.shape[2]/2)
    min_overlap = int(images.shape[2]/10)

    crop_factor = 1.5
    filter_segs_dilation = 30
    mask_irregularity = 0.70
    irregular_mask_mode = "Reuse fast"
    filter_in_segs_opt = segs
    filter_out_segs_opt = None
    segs_in = segs_nodes.MakeTileSEGS().doit(images, bbox_size, crop_factor, min_overlap, filter_segs_dilation, 
                mask_irregularity, irregular_mask_mode, filter_in_segs_opt, filter_out_segs_opt)[0]
    
    filter_in_segs_opt = None
    filter_out_segs_opt = segs
    segs_out = segs_nodes.MakeTileSEGS().doit(images, bbox_size, crop_factor, min_overlap, filter_segs_dilation, 
                mask_irregularity, irregular_mask_mode, filter_in_segs_opt, filter_out_segs_opt)[0]

    return images, segs_in, segs_out

def segsDetailer(images,segs_in,segs_out,option,seed,steps,cfg,sampler_name,scheduler,denoise_in,denoise_out,basic_pipe):
    guide_size = int(images.shape[2]/2)
    guide_size_for = "bbox"
    max_size = 1024
    feather = 5

    if option == "aniDiff":        
        images,segs_out,_,_=impact_ani.DetailerForEachPipeForAnimateDiff().doit(images,segs_out,guide_size,guide_size_for,max_size,seed,steps,cfg,sampler_name,scheduler,denoise_out,feather,basic_pipe,refiner_ratio=None,detailer_hook=None,refiner_basic_pipe_opt=None,inpaint_model=False, noise_mask_feather=0)
        
        images, segs_in,_,_=impact_ani.DetailerForEachPipeForAnimateDiff().doit(images,segs_in,guide_size,guide_size_for,max_size,seed,steps,cfg,sampler_name,scheduler,denoise_in,feather,basic_pipe,refiner_ratio=None,detailer_hook=None,refiner_basic_pipe_opt=None,inpaint_model=False, noise_mask_feather=0)
    else:
        noise_mask=force_inpaint="enabled"
        wildcard = ""

        images, segs_out,_,_=impact_pack.DetailerForEachPipe().doit(images,segs_out,guide_size,guide_size_for,max_size,seed,steps,cfg,sampler_name,scheduler,denoise_out,feather,noise_mask,force_inpaint,basic_pipe,wildcard,refiner_ratio=None,detailer_hook=None,refiner_basic_pipe_opt=None,cycle=1,inpaint_model=False,noise_mask_feather=0)
        
        images, segs_in,_,_=impact_pack.DetailerForEachPipe().doit(images,segs_in,guide_size,guide_size_for,max_size,seed,steps,cfg,sampler_name,scheduler,denoise_in,feather,noise_mask,force_inpaint,basic_pipe,wildcard,refiner_ratio=None,detailer_hook=None,refiner_basic_pipe_opt=None,cycle=1,inpaint_model=False,noise_mask_feather=0)

    return images, segs_in, segs_out        

def Sampler(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise, face_model):
    model, _, vae, positive, negative = basic_pipe

    samples=nodes.KSampler().sample(model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,denoise)[0]  
    images=nodes.VAEDecode().decode(vae, samples)[0]
    if face_model != "none":
        images = reactorNode_(images, face_model)
    
    return images, samples

