import torch

import os       
import json

import nodes
import comfy.utils
import folder_paths

import impact.core as core
import impact.subcore as subcore
from segment_anything import sam_model_registry
from ultralytics import YOLO

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
    