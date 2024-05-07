import torch
import json

from .supplement import *

import comfy_extras.nodes_mask as nodes_mask
import comfy_extras.nodes_video_model as svd

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
from animatediff.utils_model import get_available_motion_models, BetaSchedules

import adv_control.nodes as control_nodes

# DWPose
import comfyui_controlnet_aux.node_wrappers.dwpose as DWPose

# IPAdapterplus
import ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapter

# import ComfyUI_YoloWorld_EfficientSAM.YOLO_WORLD_EfficientSAM as YoloESAM
import comfyui_segment_anything.node as segm_any

def loadCkptModel(ckpt_name):
    out=nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name, output_vae=True, output_clip=True)
    out_model = out[0]
    out_clip = out[1] 
    out_vae = out[2]

    # CLIPSetLastLayer
    stop_at_clip_layer = -1
    out_clip = nodes.CLIPSetLastLayer().set_last_layer(out_clip, stop_at_clip_layer)[0]

    return out_model, out_clip, out_vae

def condBatch(cond_from, cond_to, batch_size, strength_curve):
    c = cond_from[0][0]
    p = cond_from[0][1]['pooled_output']

    for i in range(1, batch_size):
        if strength_curve == "ease-in":
            to_weight = pow(i, 2)/pow(batch_size-1, 2)
        elif strength_curve == "ease-out": 
            to_weight = 1 - pow(batch_size-1-i, 2)/pow(batch_size-1, 2)
        else:       
            to_weight = i/(batch_size-1)
        t = nodes.ConditioningAverage().addWeighted(cond_to, cond_from, to_weight)[0]

        c = torch.cat([c, t[0][0]], dim=0)
        p = torch.cat([p, t[0][1]['pooled_output']], dim=0)   
    out_cond = [[c, {"pooled_output": p}]]

    return out_cond

def aniDiff(motion_model, context_options, context_length, context_overlap):
    # LoadAnimateDiffModel
    ad_settings = None
    motion_model = AD_gen2.LoadAnimateDiffModelNode().load_motion_model(motion_model,ad_settings)[0]

    # ApplyAnimateDiffModel(Adv)
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
    guarantee_steps = 1
    prev_context=None
    view_opts=None
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
    '''
    # UseEvolvedSampling
    sample_settings=None
    # out_model = AD_gen2.UseEvolvedSamplingNode().use_evolved_sampling(ckpt_model, beta_schedule, m_models, context_options,sample_settings,beta_schedule_override=None)[0]
    out_model = AD_gen2.UseEvolvedSamplingNode().use_evolved_sampling(ckpt_model, beta_schedule, m_models, context_options, sample_settings, beta_schedule_override=None)[0]
    return out_model
    '''

    return m_models, context_options

def contNetApply(contnet_img,control_net_name,img_size,posi_cond,nega_cond):
    contnet_img = imgResize_(contnet_img, img_size)
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

def Sampler(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise):
    model, _, vae, positive, negative = basic_pipe

    samples=nodes.KSampler().sample(model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,denoise)[0]  
    images=nodes.VAEDecode().decode(vae, samples)[0]
    '''
    if face_model != "none":
        images = reactorNode_(images, face_model)
    '''
    return images, samples

'''
def yoloESAM(image, categories, confidence_threshold, mask_combined):
    yolo_world_model = "yolo_world/l"
    yolo_world_model = YoloESAM.Yoloworld_ModelLoader_Zho().load_yolo_world_model(yolo_world_model)[0]
    device = "CUDA"
    esam_model = YoloESAM.ESAM_ModelLoader_Zho().load_esam_model(device)[0]
    iou_threshold=0.1
    box_thickness=text_thickness=2
    text_scale=1.0
    with_segmentation=with_confidence=returnmask_extracted=True
    with_class_agnostic_nms=False  
    mask_extracted_index=0

    return YoloESAM.Yoloworld_ESAM_Zho().yoloworld_esam_image(image, yolo_world_model, esam_model, categories, confidence_threshold, iou_threshold, box_thickness, text_thickness, text_scale, with_segmentation, mask_combined, with_confidence, with_class_agnostic_nms, returnmask_extracted, mask_extracted_index)
'''

def svdForSampler(ckpt_name, init_image, width, height, method, motion_bucket_id, augmentation_level, seed, steps, cfg, sampler_name, scheduler, basic_pipe):
    if basic_pipe == None:
        # ImageOnlyCheckpointLoader
        model, clip_vision, vae = svd.ImageOnlyCheckpointLoader().load_checkpoint(ckpt_name, output_vae=True, output_clip=True)
        # VideoLinearCFGGuidance
        min_cfg = 1.0
        model = svd.VideoLinearCFGGuidance().patch(model, min_cfg)[0]
    else:
        model, clip_vision, vae, _, _ = basic_pipe

    # SVD_img2vid_Conditioning
    if "xt" in ckpt_name:
        video_frames = 25
    else:
        video_frames = 14
    
    fps = 6
    denoise = 1.0

    if method =="origin":
        cond_posi, cond_nega, latent_image = svd.SVD_img2vid_Conditioning().encode(clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level)

        basic_pipe = model, clip_vision, vae, cond_posi, cond_nega
        out_imgs, _ = Sampler(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise)
    else:
        for i in range(init_image.shape[0]):
            cond_posi, cond_nega, latent_image = svd.SVD_img2vid_Conditioning().encode(clip_vision, init_image[i].unsqueeze(0), vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level)

            basic_pipe = model, clip_vision, vae, cond_posi, cond_nega
            images, _ = Sampler(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise)
            if i == 0:
                out_imgs = images
            else:
                out_imgs = torch.cat([out_imgs, images], dim=0)

    return out_imgs, basic_pipe


def yoloDetector(model_name, image_list, threshold, labels, segm_detector):
    if segm_detector == None:
        # from UltralyticsDetectorProvider
        model_path = folder_paths.get_full_path("ultralytics", model_name)
        model = subcore.load_yolo(model_path)
        segm_detector = subcore.UltraSegmDetector(model)

    if labels is not None and labels != '':
        labels = labels.split(',')

    dilation = 0
    crop_factor = 3.0
    drop_size = 10
    detailer_hook = None       

    masks = []
    for image in image_list:
        image = image.unsqueeze(0)
        segs = segm_detector.detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook)

        if len(labels) > 0:
            segs, _ = segs_nodes.SEGSLabelFilter.filter(segs, labels)

        mask = core.segs_to_combined_mask(segs)
        masks.append(mask)
    masks = torch.stack(masks, dim=0)  

    return masks, segm_detector

def drawTextList(text, font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, alignment, width, height):
    if ':' in text:
        s = text.split(':')
        s = list(map(str, [x for x in range(int(s[0]), int(s[1])+1)]))
    else:    
        s = text.split(',')

    out_imgs = []
    out_masks = []
    for t in s:
        i, m = essentials.DrawText().execute(t, font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, alignment, width, height)
        out_imgs.append(i)
        out_masks.append(m)

    out_imgs = torch.cat(out_imgs, dim=0)
    out_masks = torch.cat(out_masks, dim=0)

    return out_imgs, out_masks

def maskColorValue(image, mask, method):
    mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    mask_area = (image * 255).round().to(torch.int).clamp(min=1) * mask_image
    c1 = mask_area[mask_area>0].view(-1,3) # [r,g,b] list of masked area

    m = torch.mean(c1, dim=0).round().to(torch.int)
    m1 = m.numpy()

    if method == "std":
        s = torch.std(c1, dim=0).round().to(torch.int)
        upper = (m+s).clamp(max=255)
        lower = (m-s).clamp(min=0)
    else:
        upper = torch.max(c1, dim=0).values
        lower = torch.min(c1, dim=0).values

    m = m.view(1, 1, 1, 3)
    upper = upper.view(1, 1, 1, 3)
    lower = lower.view(1, 1, 1, 3)
    color_value = [m, upper, lower]

    return color_value, m1[0], m1[1], m1[2] 

def maskAndImaging(source_image,color_value,threshold,invert_mask,b_mask,b_mask_op,to_image,to_x,to_y):
    resize_source = False
    # create masks 
    x=y=0
    masks = []
    out_imgs = []
    for i, s in enumerate(source_image):
        s = s.unsqueeze(0)
        if threshold > 0:
            upper = (color_value[0]+threshold).clamp(max=255)
            lower = (color_value[0]-threshold).clamp(min=0)
        else:
            upper = color_value[1]
            lower = color_value[2]

        i1 = (s * 255).round().to(torch.int)
        mask = (i1 >= lower) & (i1 <= upper)
        mask = mask.all(dim=-1)
        mask = mask.float()

        if invert_mask:
            mask = 1-mask

        if b_mask != None:
            mask = nodes_mask.MaskComposite().combine(mask, b_mask[i].unsqueeze(0), x, y, b_mask_op)[0]
        masks.append(mask)

        # create out_img after mask adjusted
        d = torch.zeros([1, s.shape[1], s.shape[2], 3])
        out_img = nodes_mask.ImageCompositeMasked().composite(d,s,x,y,resize_source,mask)[0]
        out_imgs.append(out_img)
    masks = torch.cat(masks, dim=0)
    out_imgs = torch.cat(out_imgs, dim=0)

    # create output_image if there are destination image
    output_images = []
    if to_image != None:
        if len(to_image) < len(out_imgs):
            quotient, remainder = divmod(len(out_imgs), len(to_image))
            temp_image = to_image
            for t in temp_image:
                to_image = torch.cat((to_image, t.repeat((quotient-1,1,1,1))), dim=0)
            if remainder > 0:
                to_image = torch.cat((to_image, t.repeat((remainder,1,1,1))), dim=0)

        output_images = []
        for i in range(len(to_image)):
            d = to_image[i].unsqueeze(0)
            s = out_imgs[i].unsqueeze(0)
            m = masks[i].unsqueeze(0)
            output_image=nodes_mask.ImageCompositeMasked().composite(d,s,to_x,to_y,resize_source,m)[0]
            output_images.append(output_image)
        output_images = torch.cat(output_images, dim=0)

    return out_imgs, masks, output_images
