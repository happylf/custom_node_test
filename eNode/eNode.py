from .common import *

MAX_RESOLUTION=4096

def readPromptType(f_path):   
    f_path = os.path.join(f_path, 'promp_type.json')
    with open(f_path, 'r', encoding='utf-8') as f:
        prompt_type_list = json.load(f)
        type_list = list()     
        for item in prompt_type_list:
            type_list.append(item['TYPE'])
            
    return (prompt_type_list, type_list)        

class BasicSetting:
    def __init__(self):    
        pass

    @classmethod
    def INPUT_TYPES(self):
        f_path = os.path.dirname(os.path.realpath(__file__))     
        self.prompt_type_list, type_list = readPromptType(f_path)  

        return {
            "required": {  
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "posi_from": ("STRING", {"default": "", "multiline": True}),
                "posi_to": ("STRING", {"default": "", "multiline": True}),
                "app_text": ("STRING", {"default": "", "multiline": True}),                
                "negative": ("STRING", {"default": "", "multiline": True}),
                "prompt_type": (type_list, {"default": "NONE"}),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("basic_pipe", "model", "clip", "vae", "positive", "negative", "latent_image")
    FUNCTION = "todo"

    CATEGORY = "eNode"

    def todo(self,ckpt_name,posi_from,posi_to,app_text,negative,prompt_type,lora_name,vae_name,width,height,batch_size):
        # load checkpoint model
        model, clip, vae = loadCkptModel(ckpt_name)

        # load vae
        if vae_name != "None":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)

        # load lora
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
            strength_model=strength_clip=1.0
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora_model, strength_model, strength_clip)
        
        # apply prompt type
        for item in self.prompt_type_list:  
            if item['TYPE'] == prompt_type:
                posi_from="masterpiece,best quality,high resolution,"+item['POSITIVE']+posi_from+app_text
                if posi_to != "":
                    posi_to="masterpiece,best quality,high resolution,"+item['POSITIVE']+posi_to+app_text
                negative="deformed,bad quality,bad anatomy," + item['NEGATIVE']+negative
                break

        cond_from = nodes.CLIPTextEncode().encode(clip, posi_from)[0]
        if posi_to == "":
            cond_posi = cond_from
        else:            
            cond_to = nodes.CLIPTextEncode().encode(clip, posi_to)[0]
            cond_posi = condBatch(cond_from, cond_to, batch_size)
        cond_nega = nodes.CLIPTextEncode().encode(clip, negative)[0]

        latent_image = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]
        basic_pipe = model, clip, vae, cond_posi, cond_nega

        return basic_pipe, model, clip, vae, cond_posi, cond_nega, latent_image

class ApplyAniDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "motion_model": (get_available_motion_models(), {"default": 'v3_sd15_mm.ckpt'}),              "context_options": (['StandardUniform','StandardStatic','LoopedUniform','Batched [Non-AD]'],{"default": 'StandardStatic'}),
                "context_length": ("INT", {"default": 16, "min": 1, "max": 128}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 128}),
            },
            "optional": {
                "basic_pipe": ("BASIC_PIPE",),  
                "model": ("MODEL",),                
            }
        }
    
    RETURN_TYPES = ("BASIC_PIPE", "MODEL",)
    RETURN_NAMES = ("basic_pipe", "model",)
    FUNCTION = "todo"

    CATEGORY = "eNode"

    def todo(self,motion_model,context_options,context_length,context_overlap,basic_pipe=None,model=None):
        if basic_pipe != None:
            model, clip, vae, posi_cond, nega_cond = basic_pipe
            model = aniDiff(model,motion_model,context_options,context_length,context_overlap)
            basic_pipe = model, clip, vae, posi_cond, nega_cond     
        else:
            model = aniDiff(model,motion_model,context_options,context_length,context_overlap)

        return (basic_pipe, model)

class ApplyContNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "contnet_img": ("IMAGE",),  
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "basic_pipe": ("BASIC_PIPE",),  
                "posi_cond": ("CONDITIONING", ),
                "nega_cond": ("CONDITIONING", ),
            }
        }
    
    RETURN_TYPES = ("BASIC_PIPE","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("basic_pipe", "posiCond", "negaCond",)
    FUNCTION = "todo"

    CATEGORY = "eNode"

    def todo(self,contnet_img,control_net_name,width,height,basic_pipe=None,posi_cond=None,nega_cond=None):
        if basic_pipe != None:
            model, clip, vae, posi_cond, nega_cond = basic_pipe
            posi_cond, nega_cond = contNetApply(contnet_img,control_net_name,width,height,posi_cond,nega_cond)
            basic_pipe = model, clip, vae, posi_cond, nega_cond     
        else:
            posi_cond, nega_cond = contNetApply(contnet_img,control_net_name,width,height,posi_cond,nega_cond)

        return (basic_pipe, posi_cond, nega_cond)
    
WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer (SDXL)', 'composition (SDXL)']

class ApplyIPadapterAdv():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                #"preset": (['None', 'PLUS (high strength)'], {"default": 'PLUS (high strength)'}),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                #"combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                #"embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "basic_pipe": ("BASIC_PIPE",),  
                "model": ("MODEL",),                
                "ipadapter": ("IPADAPTER", ),
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                #"clip_vision": ("CLIP_VISION",),
            }
        }
    
    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "IPADAPTER",)
    RETURN_NAMES = ("basic_pipe", "model", "ipadapter",)
    FUNCTION = "todo"

    CATEGORY = "eNode"

    def todo(self, image,weight,weight_type,start_at,end_at,basic_pipe=None,model=None,ipadapter=None,image_negative=None,attn_mask=None):
        if basic_pipe != None:
            model, clip, vae, posi_cond, nega_cond = basic_pipe
            model,ipadapter=applyIPAdapAdv(image,weight,weight_type,start_at,end_at,model,ipadapter,image_negative,attn_mask)
            basic_pipe = model, clip, vae, posi_cond, nega_cond     
        else:
            model,ipadapter=applyIPAdapAdv(image,weight,weight_type,start_at,end_at,model,ipadapter,image_negative,attn_mask)

        return (basic_pipe,model,ipadapter,)
    
class MakeTileSegs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "images": ("IMAGE",),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "bbox_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "min_overlap": ("INT", {"default": 100, "min": 0, "max": 512, "step": 1}),
                "option": (["None", "aniDiff"], {"default": "None"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "SEGS", "SEGS",)
    RETURN_NAMES = ("images", "segs_in", "segs_out",)
    FUNCTION = "todo"

    CATEGORY = "eNode/etc"

    def todo(self,images,scale_by,bbox_size,min_overlap,option):
        images, segs_in, segs_out = detectPerson(images,scale_by,bbox_size,min_overlap,option)
        return (images, segs_in, segs_out,)

class TileSegsDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "images": ("IMAGE",),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                #"bbox_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                #"min_overlap": ("INT", {"default": 100, "min": 0, "max": 512, "step": 1}),
                "option": (["None", "aniDiff"], {"default": "None"}),
                #"guide_size": ("FLOAT", {"default": 384, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise_in": ("FLOAT", {"default": 0.4, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "denoise_out": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "basic_pipe": ("BASIC_PIPE", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "SEGS", "SEGS")
    RETURN_NAMES = ("images", "segs_in", "segs_out")
    FUNCTION = "todo"

    CATEGORY = "eNode"

    def todo(self,images,scale_by,option,seed,steps,cfg,sampler_name,scheduler,denoise_in,denoise_out,basic_pipe):
        images, segs_in, segs_out = detectPerson(images,scale_by,option)

        return (segsDetailer(images,segs_in,segs_out,option,seed,steps,cfg,sampler_name,scheduler,denoise_in,denoise_out,basic_pipe))

class Sampler01:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "basic_pipe": ("BASIC_PIPE",),   
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "latent_image": ("LATENT", ),                
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_model": (reactor_nodes.get_model_names(reactor_nodes.get_facemodels),),
            }
        }
    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "todo"
    CATEGORY = "eNode"

    def todo(self, basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise, face_model): 
        images, samples = Sampler(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise, face_model)

        return (images, samples)    

