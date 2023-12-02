import torch

import json

import nodes
import comfy.utils
import folder_paths

import impact.core as core
import impact.subcore as subcore
from segment_anything import sam_model_registry
from ultralytics import YOLO

import os       
import sys

# AnimateDiff, ControlNet, import
comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")
AniDiff_path = os.path.join(custom_nodes_path, "ComfyUI-AnimateDiff-Evolved")
Controlnet_path  = os.path.join(custom_nodes_path, "ComfyUI-Advanced-ControlNet")
sys.path.append(AniDiff_path )
sys.path.append(Controlnet_path )
import animatediff.nodes as AD_nodes
import control.nodes as control_nodes

# IPAdapter_plus import
import ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapter

MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

MAX_RESOLUTION=8192

def read_prompt_type_list(file_path):   
    file_path = os.path.join(file_path, 'promp_list_ep01.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_type_list = json.load(file)
        type_list = list()     
        for item in prompt_type_list:
            type_list.append(item['TYPE'])
            
    return (prompt_type_list, type_list)        

class PreSampler:
    def __init__(self):    
        pass

    @classmethod
    def INPUT_TYPES(self):  
        file_path = os.path.dirname(os.path.realpath(__file__))     
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  
        
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "input_positive": ("STRING", {"default": "", "multiline": True}),
                             "input_negative": ("STRING", {"default": "", "multiline": True}),
                             "prompt_type": (type_list, {"default": "NONE"}),
                             "lora_name": (["None"] + folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                             "face_positive": ("STRING", {"default": "", "multiline": True}),
                             "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
                            }}

    RETURN_TYPES=("PRESET01", "FACELORA")
    RETURN_NAMES=("preset_sampler", "preset_face")    
    FUNCTION = "todo"
    CATEGORY = "TestEp01"

    def todo(self, ckpt_name, input_positive, input_negative, prompt_type,
               lora_name, strength_model, strength_clip, face_positive, width, height, batch_size): 

        # test

        # apply prompt type
        for item in self.prompt_type_list:  
            if item['TYPE'] == prompt_type:
                output_positive = "masterpiece, best quality, high resolution," + item['POSITIVE'] + input_positive
                output_negative = "deformed, bad quality, bad anatomy," + item['NEGATIVE'] + input_negative

                break

        # preset_sampler
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, 
                output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
 
        ckpt_model = out[0]
        ckpt_clip = out[1] 
        vae = out[2]

        tokens = ckpt_clip.tokenize(output_positive)
        ckpt_positive_cond, ckpt_positive_pooled = ckpt_clip.encode_from_tokens(tokens, return_pooled=True)

        tokens = ckpt_clip.tokenize(output_negative)
        ckpt_negative_cond, ckpt_negative_pooled = ckpt_clip.encode_from_tokens(tokens, return_pooled=True)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])

        preset_sampler = (ckpt_model, vae, [[ckpt_positive_cond, {"pooled_output": ckpt_positive_pooled}]], 
                [[ckpt_negative_cond, {"pooled_output": ckpt_negative_pooled}]], {"samples":latent},)

        # preset_face
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
            face_model, face_clip = comfy.sd.load_lora_for_models(ckpt_model, ckpt_clip, lora_model, 
                                                                  strength_model, strength_clip)
        else:
            face_model = ckpt_model
            face_clip = ckpt_clip

        tokens = face_clip.tokenize(face_positive)
        face_positive_cond, face_positive_pooled = face_clip.encode_from_tokens(tokens, return_pooled=True)

        tokens = face_clip.tokenize("")
        face_negative_cond, face_negative_pooled = face_clip.encode_from_tokens(tokens, return_pooled=True)

        preset_face = (face_model, [[face_positive_cond, {"pooled_output": face_positive_pooled}]], 
                [[face_negative_cond, {"pooled_output": face_negative_pooled}]],)

        return(preset_sampler, preset_face,)

class PreSampler_video:
    def __init__(self):    
        pass

    @classmethod
    def INPUT_TYPES(self):  
        file_path = os.path.dirname(os.path.realpath(__file__))     
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "input_positive": ("STRING", {"default": "", "multiline": True}),
                             "input_negative": ("STRING", {"default": "", "multiline": True}),
                             "prompt_type": (type_list, {"default": "NONE"}),
                             "lora_name": (["None"] + folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                             "face_positive": ("STRING", {"default": "", "multiline": True}),
                             "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "load_image": ("IMAGE",),
                             "ipadapter_image": ("IMAGE",),
                            }}

    RETURN_TYPES=("PRESET01", "FACELORA")
    RETURN_NAMES=("preset_sampler", "preset_face")    
    FUNCTION = "todo"
    CATEGORY = "TestEp01"

    def todo(self, ckpt_name, input_positive, input_negative, prompt_type,
               lora_name, strength_model, strength_clip, face_positive, width, height, batch_size, load_image, ipadapter_image): 

        # apply prompt type
        for item in self.prompt_type_list:  
            if item['TYPE'] == prompt_type:
                output_positive = "masterpiece, best quality, high resolution," + item['POSITIVE'] + input_positive
                output_negative = "deformed, bad quality, bad anatomy," + item['NEGATIVE'] + input_negative

                break

        # preset_sampler
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, 
                output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
 
        ckpt_model = out[0]
        ckpt_clip = out[1] 
        vae = out[2]

        # AnimateDiffUniformContextOptions
        context_length = 16
        context_stride = 1
        context_overlap = 4
        context_schedule = 'uniform'
        closed_loop = False
        context_options = AD_nodes.AnimateDiffUniformContextOptions().create_options(context_length,
                             context_stride, context_overlap, context_schedule, closed_loop)[0]

        # AnimatteDiff Loader
        model_name = 'mm_sd_v15_v2.ckpt'
        beta_schedule = 'sqrt_linear (AnimateDiff)'
        motion_lora = None
        motion_model_settings = None
        motion_scale = 1
        apply_v2_models_properly = False

        AD_model = AD_nodes.AnimateDiffLoaderWithContext().load_mm_and_inject_params( 
                       ckpt_model, model_name, beta_schedule, context_options, motion_lora, 
                       motion_model_settings, motion_scale, apply_v2_models_properly)[0]

        # Apply IPAdapter
        # IPAdapterModelLoader
        ipadapter_file = 'ip-adapter-plus_sd15.bin'
        ipadapter = IPAdapter.IPAdapterModelLoader().load_ipadapter_model(ipadapter_file)[0]

        # CLIPVisionLoader
        clip_name = 'sd1.5 model.safetensors'
        clip_vision = nodes.CLIPVisionLoader().load_clip(clip_name)[0]
        weight = 1
        weight_type="original"
        noise = 0.0
        embeds = None
        attn_mask = None
        start_at = 0.0
        end_at = 1.0
        unfold_batch=False
        sampler_model = IPAdapter.IPAdapterApply().apply_ipadapter(ipadapter, AD_model, weight, 
                               clip_vision, ipadapter_image, weight_type, noise, embeds, 
                               attn_mask, start_at, end_at, unfold_batch)[0]

        # CLIPSetLastLayer
        last_layer_clip = ckpt_clip.clone()
        last_layer_clip.clip_layer = (-2)

        tokens = last_layer_clip.tokenize(output_positive)
        last_layer_positive_cond, last_layer_positive_pooled = last_layer_clip.encode_from_tokens(
                                                    tokens, return_pooled=True)
        last_layer_positive = [[last_layer_positive_cond, {"pooled_output": last_layer_positive_pooled}]]

        tokens = last_layer_clip.tokenize(output_negative)
        last_layer_negative_cond, last_layer_negative_pooled = last_layer_clip.encode_from_tokens(
                                                    tokens, return_pooled=True)
        last_layer_negative = [[last_layer_negative_cond, {"pooled_output": last_layer_negative_pooled}]]

        #CLIPVisionEncode
        clip_vision_output = clip_vision.encode_image(ipadapter_image)

        # unCLIPConditioning
        strength = 1
        noise_augmentation = 0
        IPAdapter_positive = nodes.unCLIPConditioning().apply_adm(last_layer_positive,
                            clip_vision_output, strength, noise_augmentation)[0]
        
        latent = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]
              
        # Load ControlNet Model (Advanced)
        control_net_name = "control_v11p_sd15_openpose.pth"
        timestep_keyframe = None
        control_net = control_nodes.ControlNetLoaderAdvanced().load_controlnet( 
                         control_net_name, timestep_keyframe)[0]
        
        # Apply ControlNet
        strength = 1.0
        start_percent = 0.0
        end_percent = 1.0
        mask_optional = None
        controlnet_positive, controlnet_negative = control_nodes.AdvancedControlNetApply().apply_controlnet(
                            IPAdapter_positive, last_layer_negative, control_net, load_image, strength, 
                            start_percent, end_percent, mask_optional)

        print(f"ckpt_model={ckpt_model}")
        print(f"AD_model={AD_model}")
        print(f"sampler_model={sampler_model}")
        print(f"last_layer_positive_cond={last_layer_positive_cond}")
        print(f"IPAdapter_positive={IPAdapter_positive}")
        print(f"controlnet_positive={controlnet_positive}")

        preset_sampler = (sampler_model, vae, controlnet_positive, controlnet_negative, latent)

        # preset_face
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
            face_model, face_clip = comfy.sd.load_lora_for_models(ckpt_model, ckpt_clip, lora_model, 
                                                                  strength_model, strength_clip)
        else:
            face_model = ckpt_model
            face_clip = ckpt_clip

        tokens = face_clip.tokenize(face_positive)
        face_positive_cond, face_positive_pooled = face_clip.encode_from_tokens(tokens, return_pooled=True)

        tokens = face_clip.tokenize("")
        face_negative_cond, face_negative_pooled = face_clip.encode_from_tokens(tokens, return_pooled=True)

        preset_face = (face_model, [[face_positive_cond, {"pooled_output": face_positive_pooled}]], 
                [[face_negative_cond, {"pooled_output": face_negative_pooled}]],)
        
        return(preset_sampler, preset_face,)

class PreRegionalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "preset_sampler": ("PRESET01",),   
                              "preset_face": ("FACELORA",),
                              "input_image": ("IMAGE", ),
                              "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                              "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                              "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                              }}

    RETURN_TYPES = ("LATENT", "KSAMPLER_ADVANCED", "REGIONAL_PROMPTS", "VAE", "IMAGE")
    RETURN_NAMES = ("samples", "base_sampler", "regional_prompts", "vae", "face_mask_image",)
    FUNCTION = "todo"
    CATEGORY = "TestEp01"

    def todo(self, preset_sampler, preset_face, input_image, cfg, sampler_name, scheduler):
      ckpt_model, vae, positive, negative, samples = preset_sampler
      face_model, face_positive, face_negative = preset_face
   
      base_sampler = core.KSamplerAdvancedWrapper(ckpt_model, cfg, sampler_name, scheduler, positive, negative)

      # from UltralyticsDetectorProvider
      model_name = "bbox/face_yolov8m.pt"
      model_path = folder_paths.get_full_path("ultralytics", model_name)
      model = subcore.load_yolo(model_path)
      bbox_detector = subcore.UltraBBoxDetector(model)

      # from SAMLoader
      model_name = "sam_vit_b_01ec64.pth"
      model_name = folder_paths.get_full_path("sams", model_name)
      model_kind = 'vit_b'
      sam_model_opt = sam_model_registry[model_kind](checkpoint=model_name)
      sam_model_opt.is_auto_mode = "AUTO"

      # from SimpleDetectorForEach
      bbox_threshold = 0.5
      bbox_dilation = 0
      crop_factor = 3
      drop_size = 20
      sub_threshold = 0.5
      sub_dilation = 0
      sub_bbox_expansion = 0
      sam_mask_hint_threshold = 0.7
      segs = bbox_detector.detect(input_image, bbox_threshold, bbox_dilation, crop_factor, drop_size)
      mask = core.make_sam_mask(sam_model_opt, segs, input_image, "center-1", sub_dilation,
                                sub_threshold, sub_bbox_expansion, sam_mask_hint_threshold, False)
      segs = core.segs_bitwise_and_mask(segs, mask)
      mask = core.segs_to_combined_mask(segs)
      face_mask = mask
      
      advanced_sampler = core.KSamplerAdvancedWrapper(face_model, cfg, sampler_name, scheduler, 
                                                      face_positive, face_negative)
      regional_prompt = core.REGIONAL_PROMPT(face_mask, advanced_sampler)
      regional_prompts = []
      regional_prompts += [regional_prompt]

      face_image = face_mask.reshape((-1, 1, face_mask.shape[-2], 
                                      face_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
      
      return (samples, base_sampler, regional_prompts, vae, face_image)

class FromPresetSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "preset_sampler": ("PRESET01",),   
                            }}
    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("model", "vae", "positive", "negative", "samples")
    FUNCTION = "todo"
    CATEGORY = "TestEp01/etc"

    def todo(self, preset_sampler):
        model, vae, positive, negative, samples = preset_sampler
        return model, vae, positive, negative, samples