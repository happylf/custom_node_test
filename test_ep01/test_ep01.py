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

# N-Nodes (GPT Test )
from llama_cpp import Llama
N_Nodes_path = os.path.join(custom_nodes_path, "ComfyUI-N-Nodes\py")
sys.path.append(N_Nodes_path)
import gptcpp_node as N_nodes

# Impact-pack
Impact_path = os.path.join(custom_nodes_path, "ComfyUI-Impact-Pack\modules\impact")
sys.path.append(Impact_path)
import impact_pack as Impact

import comfy_extras.nodes_mask as nodes_mask

# Inspire-Pack for Regional Color Mask
Inspire_path = os.path.join(custom_nodes_path, "ComfyUI-Inspire-Pack")
sys.path.append(Inspire_path)
import inspire.regional_nodes as Regional

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

class ToPresetSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "vae": ("VAE",),
                             "positive": ("CONDITIONING",),
                             "negative": ("CONDITIONING",),
                             "samples":  ("IMAGE",),
                            }}
    RETURN_TYPES = ("PRESET01",)
    RETURN_NAMES = ("preset_sampler",)
    FUNCTION = "todo"
    CATEGORY = "TestEp01/etc"

    def todo(self, model, vae, positive, negative, samples):
        preset_sampler = (model, vae, positive, negative, samples) 
        return (preset_sampler,)
    
### N-Nodes GPT 
class GptPrompt:
    @classmethod
    def INPUT_TYPES(s):         
        return {"required": {"gpt_model": (folder_paths.get_filename_list("GPTcheckpoints"),),
                             "gpt_question": ("STRING", {"default": "", "multiline": True}),
                            }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gpt_positive_list",) 
    FUNCTION = "todo"
    CATEGORY = "TestEp01"

    def todo(self, gpt_model, gpt_question):
        # GPTLoaderSimple
        gpu_layers = 27
        n_threads = 8
        max_ctx = 2048
        gpt_model, gpt_model_path = N_nodes.GPTLoaderSimple().load_gpt_checkpoint(
            gpt_model, gpu_layers, n_threads, max_ctx)
               
        # GPTSampler
        max_tokens = 2048
        temperature = 0.7
        top_p = 0.5
        logprobs = 0
        echo = "disable"
        stop_token = "STOPTOKEN"
        frequency_penalty = 0.0
        presence_penalty = 0.0
        repeat_penalty = 1.17647
        top_k = 40
        tfs_z = 1.0
        print_output = "disable"
        cached = "NO"
        prefix = "### Instruction: "
        suffix = "### Response: "

        gpt_answer = N_nodes.GPTSampler().generate_text(gpt_question, max_tokens, temperature, 
                top_p, logprobs, echo, stop_token, frequency_penalty, presence_penalty, repeat_penalty, 
                top_k, tfs_z, gpt_model, gpt_model_path, print_output, cached, prefix, suffix)
       
        gpt_positive_list = []
        gpt_positive = gpt_answer["result"][0] 
        gpt_positive_list = gpt_positive.split('###')   
        gpt_positive_list = [x for x in gpt_positive_list if x!= '']

        return (gpt_positive_list,)
    
class PreSampler_GPT:
    @classmethod
    def INPUT_TYPES(s):  
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "gpt_positive_list": ("STRING", {"forceInput": True}),
                             "input_positive": ("STRING", {"default": "", "multiline": True}),
                             "input_negative": ("STRING", {"default": "", "multiline": True}),
                             "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                            }}

    RETURN_TYPES=("PRESET01",)
    RETURN_NAMES=("preset_sampler",)    
    FUNCTION = "todo"
    CATEGORY = "TestEp01"

    def todo(self, ckpt_name, gpt_positive_list, input_positive, input_negative, width, height, batch_size): 

        # preset_sampler
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, 
                output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
 
        ckpt_model = out[0]
        ckpt_clip = out[1] 
        vae = out[2]

        # apply prompt type
        ckpt_positive_list = []

        for gpt_positive in gpt_positive_list:
            output_positive = "masterpiece, best quality, high resolution," + input_positive + gpt_positive

            tokens = ckpt_clip.tokenize(output_positive)
            ckpt_positive_cond, ckpt_positive_pooled = ckpt_clip.encode_from_tokens(
                                 tokens, return_pooled=True)
            ckpt_positive_list.append([[ckpt_positive_cond, {"pooled_output": ckpt_positive_pooled}]])

        tokens = ckpt_clip.tokenize(input_negative)
        ckpt_negative_cond, ckpt_negative_pooled = ckpt_clip.encode_from_tokens(tokens, return_pooled=True)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])

        preset_sampler = (ckpt_model, vae, ckpt_positive_list, 
                [[ckpt_negative_cond, {"pooled_output": ckpt_negative_pooled}]], {"samples":latent},)

        return(preset_sampler,)

class Sampler_GPT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "preset_for_sampler": ("PRESET01",),   
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                              "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, 
                                                "step":0.1, "round": 0.01}),
                              "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                              "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                              "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),                                                      
                              }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "todo"
    CATEGORY = "TestEp01/Sampler"

    def todo(self, preset_for_sampler, seed, steps, cfg, sampler_name, scheduler, denoise=1.0):  
        model, vae, positive_list, negative, latent_image = preset_for_sampler
        samples_list = []
        for positive in positive_list:
            samples = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                            positive, negative, 
                                        latent_image, denoise=denoise)           
            samples_list.append(vae.decode(samples[0]["samples"]))
 
        output_image = torch.cat(samples_list, 0)
        return (output_image,)

class FaceEnhance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "basic_pipe": ("BASIC_PIPE",),
                              "input_image": ("IMAGE", ),
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                              "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                              "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                              "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                              "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_img",)
    FUNCTION = "todo"
    CATEGORY = "TestEp01/etc"

    def todo(self, face_detailer, input_image, seed, steps, cfg, sampler_name, scheduler, denoise):

        print(f"image={input_image}")

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
        # mask = core.segs_to_combined_mask(segs)
        # face_mask = mask

        model, clip, vae, positive, negative = face_detailer
        guide_size = 256
        guide_size_for = 'bbox'
        max_size = 768
        feather = 5
        noise_mask = "enabled"
        force_inpaint = "enabled"
        wildcard = ""
        detailer_hook=None
        enhanced_img = Impact.DetailerForEach().doit(input_image, segs, model, clip, vae, guide_size,
                guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, positive, 
                negative, denoise, feather, noise_mask, force_inpaint, wildcard, detailer_hook)
      
        return (enhanced_img,)
    
class Yolo_Detector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image_list": ("IMAGE", ),
                              "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "labels": ("STRING", {"multiline": True, "default": "all", 
                              "placeholder": "List the types of segments to be allowed, separated by commas"}),                            
                              }}

    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("person_masks", "others_masks", "mask_color_image")
    FUNCTION = "todo"
    CATEGORY = "TestEp01/Detector"

    def todo(self, image_list, threshold, labels):
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
        mask_color_image_list = []

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
            d = nodes.EmptyImage().generate(width, height, batch_size=1, color=255)[0]      #blue
            s = nodes.EmptyImage().generate(width, height, batch_size=1, color=16711680)[0] #red - person
            temp_image = nodes_mask.ImageCompositeMasked().composite(d, s, x, y, 
                                      resize_source, person_masks)[0]
            s = nodes.EmptyImage().generate(width, height, batch_size=1, color=65280)[0]    #green - others
            mask_color_image = nodes_mask.ImageCompositeMasked().composite(temp_image, s, x, y, 
                                      resize_source, others_masks)[0]
                 
            mask_color_image_list.append(mask_color_image)
        
        output_person_masks = torch.cat(person_masks_list, 0)
        output_others_masks = torch.cat(others_masks_list, 0) 
        output_mask_color_image = torch.cat(mask_color_image_list, 0)

        return (output_person_masks, output_others_masks, output_mask_color_image)
    
class PreSampler_IPAdapter:
    @classmethod
    def INPUT_TYPES(s):  
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "input_positive": ("STRING", {"default": "", "multiline": True}),
                             "input_negative": ("STRING", {"default": "", "multiline": True}),
                             "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "load_image": ("IMAGE",),
                             "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "background_image": ("IMAGE", ),
                             "person_image": ("IMAGE", ),
                             "others_image": ("IMAGE", ),
                            }}

    RETURN_TYPES=("PRESET01", "IMAGE")
    RETURN_NAMES=("preset_sampler", "color_mask_list")    

    FUNCTION = "todo"
    CATEGORY = "TestEp01"

    def todo(self, ckpt_name, input_positive, input_negative, width, height, batch_size, load_image, threshold,
             background_image, person_image, others_image): 

        # checkpoint model
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, 
                output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
 
        ckpt_model = out[0]
        ckpt_clip = out[1] 
        vae = out[2]

        # prompt encoding
        output_positive = "masterpiece, best quality, high resolution," + input_positive
        tokens = ckpt_clip.tokenize(output_positive)
        ckpt_positive_cond, ckpt_positive_pooled = ckpt_clip.encode_from_tokens(
                                 tokens, return_pooled=True)
        tokens = ckpt_clip.tokenize(input_negative)
        ckpt_negative_cond, ckpt_negative_pooled = ckpt_clip.encode_from_tokens(tokens, return_pooled=True)
        latent_image = torch.zeros([batch_size, 4, height // 8, width // 8])
        
        upscale_method = "nearest-exact"
        crop = "disabled"
        load_image = nodes.ImageScale().upscale(load_image, upscale_method, width, height, crop)[0]

        # Load ControlNet Model (Advanced)
        control_net_name = "control_v11p_sd15_lineart_fp16.safetensors"
        timestep_keyframe = None
        control_net = control_nodes.ControlNetLoaderAdvanced().load_controlnet( 
                         control_net_name, timestep_keyframe)[0]
        
        # Apply ControlNet
        strength = 1.0
        start_percent = 0.0
        end_percent = 1.0
        mask_optional = None

        controlnet_positive_list = []
        controlnet_negative_list = []
        for image in load_image:
            image = image.unsqueeze(0)
            controlnet_positive, controlnet_negative = control_nodes.AdvancedControlNetApply().apply_controlnet(
                                [[ckpt_positive_cond, {"pooled_output": ckpt_positive_pooled}]], 
                                [[ckpt_negative_cond, {"pooled_output": ckpt_negative_pooled}]], 
                                control_net, image, strength, start_percent, end_percent, mask_optional)
            controlnet_positive_list.append(controlnet_positive) 
            controlnet_negative_list.append(controlnet_negative)

        # Yolo_Detector (color mask)
        labels = 'all'
        _, _, color_mask_list = Yolo_Detector().todo(load_image, threshold, labels)

        output_model = []

        for color_mask in color_mask_list:
            color_mask = color_mask.unsqueeze(0)    

            # IPAdapterToModel
            weight = 0.7
            noise = 0.5
            weight_type = "original"
            mask_color = "#0000FF"
            cond_b, _ = Regional.RegionalIPAdapterColorMask().doit(color_mask, mask_color, 
                                            background_image, weight, noise, weight_type)
            mask_color = "#FF0000"
            cond_p, _ = Regional.RegionalIPAdapterColorMask().doit(color_mask, mask_color, 
                                            person_image, weight, noise, weight_type)
            mask_color = "#00FF00"
            cond_o, _ = Regional.RegionalIPAdapterColorMask().doit(color_mask, mask_color, 
                                            others_image, weight, noise, weight_type)

            # Apply IPAdapter
            # IPAdapterModelLoader
            ipadapter_file = 'ip-adapter-plus_sd15.bin'
            ipadapter = IPAdapter.IPAdapterModelLoader().load_ipadapter_model(ipadapter_file)[0]

            # CLIPVisionLoader
            clip_name = 'sd1.5 model.safetensors'
            clip_vision = nodes.CLIPVisionLoader().load_clip(clip_name)[0]

            # ToIPAdapterPipe:
            pipe = Regional.ToIPAdapterPipe().doit(ipadapter, clip_vision, ckpt_model)[0]

            # ApplyRegionalIPAdapters
            kwargs = ({'ipadapter_pipe': pipe, 'regional_ipadapter1': cond_b, 'regional_ipadapter2': cond_p, 
                    'regional_ipadapter3': cond_o})
            
            output_model.append(Regional.ApplyRegionalIPAdapters().doit(**kwargs)[0])
      
        preset_sampler = (output_model, vae, controlnet_positive_list, controlnet_negative_list, {"samples":latent_image},)

        return(preset_sampler, color_mask_list)
    
class Sampler_IPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "preset_for_sampler": ("PRESET01",),   
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                              "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, 
                                                "step":0.1, "round": 0.01}),
                              "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                              "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                              "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),                                                      
                              }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "todo"
    CATEGORY = "TestEp01/Sampler"

    def todo(self, preset_for_sampler, seed, steps, cfg, sampler_name, scheduler, denoise=1.0):  
        model_list, vae, positive_list, negative_list, latent_image = preset_for_sampler
        samples_list = []
        
        i = 0
        for model in model_list:
            samples = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                           positive_list[i], negative_list[i], latent_image, denoise=denoise)           
            samples_list.append(vae.decode(samples[0]["samples"]))
            i = i + 1
 
        output_image = torch.cat(samples_list, 0)
        return (output_image,)    