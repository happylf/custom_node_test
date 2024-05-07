import torch

import os       
import json

import nodes
import comfy.utils
import folder_paths

MAX_RESOLUTION=8192

class cl_TestNode01:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "input_text": ("STRING", {"default": "", "multiline": True})
                               
                              }}

    RETURN_TYPES = ("STRING", )
    FUNCTION = "test01"
    CATEGORY = "TestNode/Begin"

    def test01(self, input_text):
      output_text = input_text + ", mountain"
      print(f"output_text = {output_text}")
      return (output_text,)

def read_prompt_type_list(file_path):    #N02-05 read prompt type from jason file
    file_path = os.path.join(file_path, 'prompt_type_list.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_type_list = json.load(file)
        type_list = list()      #02-06 for #N02-02
        for item in prompt_type_list:
            type_list.append(item['TYPE'])
            
    return (prompt_type_list, type_list)        
        
class cl_TestNode02:        #N02-01 copy from test01, chaange '01' to '02'
    def __init__(self):     #02-07 python requirement
        pass
    
    @classmethod
    def INPUT_TYPES(self):  #02-08 change to 'self' - python requirement
        file_path = os.path.dirname(os.path.realpath(__file__))     #02-09 (to #02-05)    
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  #02-09 (from #02-05)
        
        return {"required": {"input_text": ("STRING", {"default": "", "multiline": True}),
                             "select_type": (type_list, {"default": "BATTLE"})  #N02-02 add 1 more input
                              }}

    RETURN_TYPES = ("STRING", )
    FUNCTION = "test02"
    CATEGORY = "TestNode/Begin"

    def test02(self, input_text, select_type): #N02-03 add 1 more input
        for item in self.prompt_type_list:
            if item['TYPE'] == select_type:
                output_text = input_text + item['PROMPT'] 
                break
            
        return (output_text,)
    
class cl_TestNode03:        
    def __init__(self):    
        pass
    
    @classmethod
    def INPUT_TYPES(self):  
        file_path = os.path.dirname(os.path.realpath(__file__))     
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  
        
        return {"required": { "clip": ("CLIP", ),
                             "input_text": ("STRING", {"default": "", "multiline": True}),
                             "select_type": (type_list, {"default": "BATTLE"})  
                              }}

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "test03"
    CATEGORY = "TestNode/Begin"

    def test03(self, clip, input_text, select_type): 
        for item in self.prompt_type_list:
            if item['TYPE'] == select_type:
                output_text = input_text + item['PROMPT'] 
                break
 
        tokens = clip.tokenize(output_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        print(f"prompt={output_text}")
        print(f"tokenize={tokens}")
        print(f"encode={cond}")
              
        return ([[cond, {"pooled_output": pooled}]], )

class cl_TestNode04:        
    def __init__(self):    
        pass
    
    @classmethod
    def INPUT_TYPES(self):  
        file_path = os.path.dirname(os.path.realpath(__file__))     
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  
        
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "input_positive": ("STRING", {"default": "", "multiline": True}),
                             "input_negative": ("STRING", {"default": "", "multiline": True}),
                             "select_type": (type_list, {"default": "BATTLE"})  
                            }}

    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "VAE", "positive", "negative",)    
    FUNCTION = "test04"
    CATEGORY = "TestNode/Begin"

    def test04(self, ckpt_name, input_positive, input_negative, select_type): 
        for item in self.prompt_type_list:
            if item['TYPE'] == select_type:
                output_positive = input_positive + item['PROMPT'] 
                break
        output_positive = "masterpiece, best quality, high resolution," + output_positive

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, 
                output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        clip = out[1]  
        tokens = clip.tokenize(output_positive)
        positive_cond, positive_pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        tokens = clip.tokenize(input_negative)
        negative_cond, negative_pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return (out[0], out[2], [[positive_cond, {"pooled_output": positive_pooled}]], 
                [[negative_cond, {"pooled_output": negative_pooled}]], )
    
class cl_TestNode05:        
    def __init__(self):    
        pass
    @classmethod
    def INPUT_TYPES(self):  
        file_path = os.path.dirname(os.path.realpath(__file__))     
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  
        
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "input_positive": ("STRING", {"default": "", "multiline": True}),
                             "input_negative": ("STRING", {"default": "", "multiline": True}),
                             "select_type": (type_list, {"default": "BATTLE"}),
                             "lora_name": (folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                            }}

    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "VAE", "positive", "negative",)    
    FUNCTION = "test05"
    CATEGORY = "TestNode/Begin"

    def test05(self, ckpt_name, input_positive, input_negative, select_type,
               lora_name, strength_model, strength_clip): 
        for item in self.prompt_type_list:
            if item['TYPE'] == select_type:
                output_positive = input_positive + item['PROMPT'] 
                break
        output_positive = "masterpiece, best quality, high resolution," + output_positive

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, 
                output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        # load lora everytime(check original source in nodes.py)
        if not (strength_model == 0 and strength_clip == 0):
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(out[0], out[1], lora, strength_model, strength_clip)
        else:
            model_lora = out[0]
            clip_lora = out[1]

        tokens = clip_lora.tokenize(output_positive)
        positive_cond, positive_pooled = clip_lora.encode_from_tokens(tokens, return_pooled=True)

        tokens = clip_lora.tokenize(input_negative)
        negative_cond, negative_pooled = clip_lora.encode_from_tokens(tokens, return_pooled=True)


        return (model_lora, out[2], [[positive_cond, {"pooled_output": positive_pooled}]], 
                [[negative_cond, {"pooled_output": negative_pooled}]], )
    
class cl_TestNode06:        
    def __init__(self):    
        pass
    @classmethod
    def INPUT_TYPES(self):  
        file_path = os.path.dirname(os.path.realpath(__file__))     
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  
        
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             "input_positive": ("STRING", {"default": "", "multiline": True}),
                             "input_negative": ("STRING", {"default": "", "multiline": True}),
                             "select_type": (type_list, {"default": "BATTLE"}),
                             "lora_name": (folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                             "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
                            }}

    RETURN_TYPES = ("PRESET01",)
    RETURN_NAMES = ("preset_for_sampler",)    
    FUNCTION = "test06"
    CATEGORY = "TestNode/Begin"

    def test06(self, ckpt_name, input_positive, input_negative, select_type,
               lora_name, strength_model, strength_clip, width, height, batch_size=1): 
        for item in self.prompt_type_list:
            if item['TYPE'] == select_type:
                output_positive = input_positive + item['PROMPT'] 
                break
        output_positive = "masterpiece, best quality, high resolution," + output_positive

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, 
                output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        # load lora everytime(check original source in nodes.py)
        if not (strength_model == 0 and strength_clip == 0):
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(out[0], out[1], lora, strength_model, strength_clip)
        else:
            model_lora = out[0]
            clip_lora = out[1]

        tokens = clip_lora.tokenize(output_positive)
        positive_cond, positive_pooled = clip_lora.encode_from_tokens(tokens, return_pooled=True)

        tokens = clip_lora.tokenize(input_negative)
        negative_cond, negative_pooled = clip_lora.encode_from_tokens(tokens, return_pooled=True)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])            
        preset_for_sampler = (model_lora, out[2], [[positive_cond, {"pooled_output": positive_pooled}]], 
                [[negative_cond, {"pooled_output": negative_pooled}]], {"samples":latent},)
        
        return(preset_for_sampler, )

class cl_TestSampler01:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "preset_for_sampler": ("PRESET01",),   
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                              "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                              "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                              "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                              "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),                                                      
                              }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "testsampler01"
    CATEGORY = "TestNode/Begin"

    def testsampler01(self, preset_for_sampler, seed, steps, cfg, sampler_name, scheduler, denoise=1.0): 
    
        model, vae, positive, negative, latent_image = preset_for_sampler
        print(f"positive = {positive}")
        samples = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
                                        latent_image, denoise=denoise) 

        return (vae.decode(samples[0]["samples"]),)
        
        
