import torch
from GPUtil import showUtilization as gpu_usage

import nodes 
import folder_paths

import os       
import sys

from .supplement import *

import comfy_extras.nodes_mask as nodes_mask

import impact.core as core
import impact.detectors as detectors 
import impact.impact_pack as impact_pack
import impact.segs_nodes as segs_nodes
import impact.subcore as subcore
import impact.subpack_nodes as subpack
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
import animatediff.nodes_gen2 as AD_gen2
import animatediff.nodes_context as AD_context

import adv_control.nodes as control_nodes

# IPAdapter_plus
import ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapter

# DWPose
import comfyui_controlnet_aux.node_wrappers.dwpose as DWPose

def load_ckpt_model(ckpt_name):
    out=nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name, output_vae=True, output_clip=True)
    out_model = out[0]
    out_clip = out[1] 
    out_vae = out[2]

    # CLIPSetLastLayer
    stop_at_clip_layer = -1
    out_clip = nodes.CLIPSetLastLayer().set_last_layer(out_clip, stop_at_clip_layer)[0]

    return out_model, out_clip, out_vae

def ani_diff(ckpt_model, motion_model, context_options, context_length, context_overlap):
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

def PromptEncoding(clip, in_posi, in_nega):
    in_posi = "masterpiece, best quality, high resolution," + in_posi 
    in_nega = "deformed, bad quality, bad anatomy, bad hand, extra fingers," + in_nega

    out_posi_cond = nodes.CLIPTextEncode().encode(clip, in_posi)[0]
    out_nega_cond = nodes.CLIPTextEncode().encode(clip, in_nega)[0]    

    return out_posi_cond, out_nega_cond

def IPAd_apply(set_IPAdapter, in_model, attn_mask):
    ipadapter,weight,clip_vision,image,weight_type,noise,start_at,end_at=set_IPAdapter       

    # IPAdapter_plus PrepImageForClipVision
    interpolation = "LANCZOS"
    crop_position = "center"
    sharpening=0.0         
    t_img = IPAdapter.PrepImageForClipVision().prep_image(image, interpolation, 
                    crop_position, sharpening)[0]
  
    # Apply IPAdapter to model
    embeds = None
    unfold_batch=False
    out_model=IPAdapter.IPAdapterApply().apply_ipadapter(ipadapter,in_model,weight,
        clip_vision,t_img,weight_type,noise, embeds,attn_mask,start_at,end_at,unfold_batch)[0]

    return out_model

def cn_apply(posiCond, negaCond, control_net, cnImg, latent_keyframe):
    # TimestepKeyframe
    start_percent = 0.0
    strength = 1.0
    null_latent_kf_strength = 0.0
    cn_weights=control_net_weights=prev_timestep_kf=prev_timestep_keyframe=mask_optional=None
    inherit_missing=guarantee_usage=True
    interpolation='SI.NONE'
    timestep_kf=control_nodes.TimestepKeyframeNode().load_keyframe(
        start_percent,strength,cn_weights,control_net_weights,latent_keyframe,
        prev_timestep_kf,prev_timestep_keyframe,null_latent_kf_strength,
        inherit_missing,guarantee_usage,mask_optional,interpolation,)[0]
    
    # ApplyControlNet
    strength = 1.0  
    start_percent = 0.0
    end_percent = 1.0
    mask_optional=model_optional=latent_kf_override=weights_override=None
    posiCond,negaCond,_=control_nodes.AdvancedControlNetApply().apply_controlnet(
        posiCond,negaCond,control_net,cnImg,strength,start_percent,end_percent,
        mask_optional,model_optional,timestep_kf,latent_kf_override,weights_override)

    return (posiCond, negaCond,)

def yolo_detect(image_list, threshold):
        # from UltralyticsDetectorProvider
        model_name = "segm/yolov8m-seg.pt"
        model_path = folder_paths.get_full_path("ultralytics", model_name)
        model = subcore.load_yolo(model_path)
        segm_detector = subcore.UltraSegmDetector(model)

        dilation = 0
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
            p_segs = []
            o_segs = []
            for x in segs[1]:
                if x.label == 'person':
                    p_segs.append(x)
                else:    
                    o_segs.append(x)
            person_segs = (segs[0], p_segs)
            others_segs = (segs[0], o_segs)

            # SEGSOrderedFilter        
            target = "area(=w*h)"
            order = "descending"
            take_start = 0
            take_count = 1
            person_segs = segs_nodes.SEGSOrderedFilter().doit(person_segs, target, order, 
                        take_start, take_count)[0]

            person_masks = core.segs_to_combined_mask(person_segs)
            others_masks = core.segs_to_combined_mask(others_segs)

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
    
    del pre_sampler, model, vae, positive, negative, latent_image
    return images, samples

def img_scale_adjust(image, width, height):
    upscale_method = "nearest-exact"
    crop = "disabled"
    image = nodes.ImageScale().upscale(image, upscale_method, width, height, crop)[0]

    return image

def IPAdapter_set(IPAd_img, IPAd_name, set_ty):
    IPAd_model = IPAdapter.IPAdapterModelLoader().load_ipadapter_model(IPAd_name)[0]
    clip_name = "sd1.5 model.safetensors"
    clip_vision = nodes.CLIPVisionLoader().load_clip(clip_name)[0]   
    if set_ty == 1:
        weight = 1.0
        weight_type = "original"
        noise = 0.0
        start_at = 0.0
        end_at = 1.0
    else:
        weight = 0.7
        weight_type = "original"
        noise = 0.33
        start_at = 0.0
        end_at = 0.7

    set_IPAdapter = (IPAd_model, weight, clip_vision, IPAd_img, weight_type, noise, start_at, end_at)

    return set_IPAdapter 

def mask_composite(d, s, o):
    x = 0
    y = 0
    out_mask = nodes_mask.MaskComposite().combine(d, s, x, y, o)[0]
    return out_mask

def puz15_draw(image, crop_img, crop_w, crop_h, mask, puzzle, count_down):
    crop_w = int(crop_w)
    crop_h = int(crop_h)

    resize_source = False
    t_img = image
    for i in range(4):
        for j in range(4):
            x = j * crop_w
            y = i * crop_h

            d = t_img
            if puzzle.board[i][j] == 0:               
                if count_down == 0:    
                    s = crop_img[15]
                else:
                    d1 = crop_img[15]

                    text = str(count_down)
                    w = 90 
                    h = 80
                    size = 80
                    
                    bk_color = '#FFFFFF'
                    s, m = drawText(text, w, h, size, bk_color)

                    x1 = int((crop_w-w)/2)
                    y1 = int((crop_h-h)/2)
                    s = nodes_mask.ImageCompositeMasked().composite(d1, s, x1, y1, resize_source, m)[0]
            else:
                s = crop_img[puzzle.board[i][j]-1]
            
            t_img = nodes_mask.ImageCompositeMasked().composite(d, s, x, y, resize_source, mask)[0]

    return t_img 

def drawGridImage(image, w, h, color_backg):
    col_bk = []
    if color_backg != '':
        col_bk = color_backg.split(':')
        if col_bk[0] == 'r':
            col_bk[1] = list(map(int, col_bk[1].split(',')))
            col_bk = list(range(col_bk[1][0], col_bk[1][1]))
        else:
            col_bk = list(map(int, col_bk[0].split(',')))
   
    imgCnt = image.shape[0]
    q = imgCnt // 8
    t_h = 32    
    image = imgResize(image, w, h)
    d = nodes.EmptyImage().generate(w*8,(t_h+h+8)*q,1,color=0)[0]
    value = 1   
    mask = nodes_mask.SolidMask().solid(value, w, h)[0]     
    resize_source = False    
    
    for i in range(q):
        for j in range(8):  
            idx = i*8+j 
            text = str(idx+1)   
            size = 28

            bk_color = '#FFFFFF'
            if (idx+1) in col_bk:
                bk_color = '#00FF00'
                
            s, m = drawText(text, w, h, size, bk_color)

            x = w*j
            y = (t_h+h+8)*i
            d = nodes_mask.ImageCompositeMasked().composite(d,s,x,y,resize_source,m)[0]
            s = image[idx].unsqueeze(0)
            d = nodes_mask.ImageCompositeMasked().composite(d,s,x, y+t_h,resize_source,mask)[0]
    return d            

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

def makeDWPose(image, w, h):
    detect_hand=detect_body=detect_face="enable"
    resolution = w
    bbox_detector="yolox_l.onnx"
    pose_estimator="dw-ll_ucoco_384.onnx"
    kwargs={}
    dwpose = DWPose.DWPose_Preprocessor().estimate_pose(image,detect_hand,detect_body,
                detect_face,resolution,bbox_detector,pose_estimator,**kwargs)

    return dwpose['result'][0]

def detectPerson(image,scale_by,bbox_size_in,min_overlap_in,bbox_size_out,min_overlap_out):
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
    segs =  detectors.SimpleDetectorForEach().detect(bbox_detector, image, bbox_threshold, bbox_dilation, 
            crop_factor, drop_size, sub_threshold, sub_dilation, sub_bbox_expansion,
            sam_mask_hint_threshold, post_dilation, sam_model_opt, segm_detector_opt, detailer_hook)[0]
    
    # Upscale Image By
    upscale_method = "lanczos"
    image = nodes.ImageScaleBy().upscale(image, upscale_method, scale_by)[0]

    crop_factor = 1.5
    filter_segs_dilation = 30
    mask_irregularity = 0.70
    irregular_mask_mode = "Reuse fast"
    filter_in_segs_opt = segs
    filter_out_segs_opt = None
    segs_in = segs_nodes.MakeTileSEGS().doit(image, bbox_size_in, crop_factor, min_overlap_in, filter_segs_dilation, 
                mask_irregularity, irregular_mask_mode, filter_in_segs_opt, filter_out_segs_opt)[0]
    
    filter_in_segs_opt = None
    filter_out_segs_opt = segs
    segs_out = segs_nodes.MakeTileSEGS().doit(image, bbox_size_out, crop_factor, min_overlap_out, filter_segs_dilation, 
                mask_irregularity, irregular_mask_mode, filter_in_segs_opt, filter_out_segs_opt)[0]

    return image, segs_in, segs_out