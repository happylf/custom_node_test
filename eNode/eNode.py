from .common import *

from PIL import Image
MAX_RESOLUTION=4096

# from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

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
                "conditioning_to_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_curve": (["linear", "ease-in", "ease-out"], {"default": "linear",}),
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
    
    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE_SIZE")
    RETURN_NAMES = ("basic_pipe", "model", "clip", "vae", "positive", "negative", "latent_image", "img_size")
    FUNCTION = "todo"

    CATEGORY = "eNode"

    def todo(self,ckpt_name,posi_from,posi_to,conditioning_to_strength,strength_curve,app_text,negative,prompt_type,lora_name,vae_name,width,height,batch_size):
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
            if batch_size == 1:
                cond_posi = nodes.ConditioningAverage().addWeighted(cond_to, cond_from, conditioning_to_strength)[0]
            else:
                cond_posi = condBatch(cond_from, cond_to, batch_size, strength_curve)
        cond_nega = nodes.CLIPTextEncode().encode(clip, negative)[0]

        latent_image = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]
        basic_pipe = model, clip, vae, cond_posi, cond_nega
        img_size = width, height

        return basic_pipe, model, clip, vae, cond_posi, cond_nega, latent_image, img_size

class ApplyAniDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "motion_model": (get_available_motion_models(), {"default": 'v3_sd15_mm.ckpt'}), 
                "context_options": (['StandardUniform','StandardStatic','LoopedUniform','Batched [Non-AD]'],{"default": 'StandardStatic'}),
                "context_length": ("INT", {"default": 16, "min": 1, "max": 128}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("M_MODELS", "CONTEXT_OPTIONS")
    RETURN_NAMES = ("m_models", "context_options")
    FUNCTION = "todo"

    CATEGORY = "eNode"

    def todo(self,motion_model,context_options,context_length,context_overlap):
        '''
        if basic_pipe != None:
            model, clip, vae, posi_cond, nega_cond = basic_pipe
            model = aniDiff(model,motion_model,context_options,context_length,context_overlap,beta_schedule)
            basic_pipe = model, clip, vae, posi_cond, nega_cond     
        else:
            model = aniDiff(model,motion_model,context_options,context_length,context_overlap,beta_schedule)
        '''
        return aniDiff(motion_model, context_options, context_length, context_overlap)

class ApplyContNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "contnet_img": ("IMAGE",),  
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "img_size": ("IMAGE_SIZE",),
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

    def todo(self,contnet_img,control_net_name,img_size,basic_pipe=None,posi_cond=None,nega_cond=None):
        if basic_pipe != None:
            model, clip, vae, posi_cond, nega_cond = basic_pipe
            posi_cond, nega_cond = contNetApply(contnet_img,control_net_name,img_size,posi_cond,nega_cond)
            basic_pipe = model, clip, vae, posi_cond, nega_cond     
        else:
            posi_cond, nega_cond = contNetApply(contnet_img,control_net_name,img_size,posi_cond,nega_cond)

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
                "noise_type": (["none", "fade", "dissolve", "gaussian", "shuffle"], {"default": 'none'} ),
                "strength": ("FLOAT", { "default": 1.0, "min": 0, "max": 1, "step": 0.05 }),
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

    def todo(self, image,weight,weight_type,start_at,end_at,noise_type,strength,basic_pipe=None,model=None,ipadapter=None,image_negative=None,attn_mask=None):
        if noise_type != "none":
            blur = 0
            image_optional=image_negative
            image_negative = IPAdapter.IPAdapterNoise().make_noise(noise_type, strength, blur, image_optional)[0]

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

    CATEGORY = "eNode/Detailer"

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
                # "face_model": (reactor_nodes.get_model_names(reactor_nodes.get_facemodels),),
            }
        }
    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "todo"
    CATEGORY = "eNode/Sampler"

    def todo(self, basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise): 
        images, samples = Sampler(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise)
        
        return (images, samples)    

class MaskColorValue:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (["range", "std"], {"default": "range"},),
            }
        }

    RETURN_TYPES = ("COLOR_VALUE", "INT", "INT", "INT")
    RETURN_NAMES = ("color_value", "red", "green", "blue")
    FUNCTION = "todo"
    CATEGORY = "eNode/Mask"

    def todo(self, image, mask, method):

        return maskColorValue(image, mask, method)

class MaskingAndImaging:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "color_value": ("COLOR_VALUE",),
                "threshold": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
                "invert_mask": ("BOOLEAN", { "default": False }),
            },
            "optional": {
                "b_mask": ("MASK",),
                "b_mask_op": (["multiply", "add", "subtract", "and", "or", "xor"],),
                "to_image": ("IMAGE",),
                "to_x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "to_y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "output_image")
    FUNCTION = "todo"
    CATEGORY = "eNode/Mask"

    def todo(self,source_image,color_value,threshold,invert_mask,b_mask=None,b_mask_op=None,to_image=None,to_x=None,to_y=None):
        return maskAndImaging(source_image,color_value,threshold,invert_mask,b_mask,b_mask_op,to_image,to_x,to_y)                

class CreateMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "mask_type": (["box", "circle"], {"default": "box"}),
                "mask_x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "mask_y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),                
                "mask_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),                
                "box_width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "box_height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "circle_radius": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "todo"
    CATEGORY = "eNode/Mask"

    def todo(self, width, height, mask_type, mask_x, mask_y, mask_value, box_width, box_height, circle_radius):
        if mask_type == "box":
            mask = torch.full((1, height, width), 0, dtype=torch.float32, device="cpu")
            mask[:, mask_y:mask_y+box_height, mask_x:mask_x+box_width] = mask_value
        elif mask_type == "circle":
            y_axis, x_axis = torch.meshgrid(torch.arange(height), torch.arange(width))
            mask = (((y_axis - mask_y)**2 + (x_axis - mask_x)**2) <= circle_radius**2) * mask_value
            mask.float()
            mask = mask.unsqueeze(0)

        mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        return (mask, mask_image)

class CreateBatchMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "mask_type": (["box", "circle"], {"default": "box"}),
                "mask_x": ("STRING", {"default": "0,0", }),
                "mask_y": ("STRING", {"default": "0,0", }),                
                "mask_value": ("STRING", {"default": "1.0,1.0", }),                
                "box_width": ("STRING", {"default": "512,512", }),
                "box_height": ("STRING", {"default": "512,512", }),
                "circle_radius": ("STRING", {"default": "512,512", }),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "todo"
    CATEGORY = "eNode/Mask"

    def todo(self, width, height, mask_type, mask_x, mask_y, mask_value, box_width, box_height, circle_radius,batch_size):
        def interval(x):
            x = list(map(float, x.split(','))) 
            x = [x[0]+(x[1]-x[0])*k/(batch_size-1) for k in range(batch_size)]
            return x
        mask_x = list(map(int, interval(mask_x)))    
        mask_y = list(map(int, interval(mask_y)))  
        mask_value = list(map(float, interval(mask_value)))    
        box_width = list(map(int, interval(box_width)))      
        box_height = list(map(int, interval(box_height)))   
        circle_radius = list(map(int, interval(circle_radius)))   

        masks = []
        mask_images = []
        for i in range(batch_size):
            mask, mask_image = CreateMask().todo(width, height, mask_type, mask_x[i], mask_y[i], mask_value[i], box_width[i], box_height[i], circle_radius[i])
            masks.append(mask)
            mask_images.append(mask_image)
        masks = torch.cat(masks, 0)   
        mask_images = torch.cat(mask_images, 0)   

        return (masks, mask_images)
'''
class YoloESAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "categories": ("STRING", {"default": "person", "multiline": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step":0.01}),
                "mask_combined": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "todo"
    CATEGORY = "eNode/detect"

    def todo(self, image, categories, confidence_threshold, mask_combined):
        return yoloESAM(image, categories, confidence_threshold, mask_combined)
'''
class SegmAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": (segm_any.list_sam_model(), ),
                "dino_model": (segm_any.list_groundingdino_model(), ),
                "image": ("IMAGE",),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {"default": 0.3,"min": 0,"max": 1.0,"step": 0.01}),
            },
            "optional": {
                "sam_dino_model": ("SAM_DINO_MODEL",),              
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "SAM_DINO_MODEL")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "sam_dino_model")
    FUNCTION = "todo"
    CATEGORY = "eNode/Detector"

    def todo(self, sam_model, dino_model, image, prompt, threshold, sam_dino_model=None):
        if sam_dino_model == None:
            sam_model = segm_any.SAMModelLoader().main(sam_model)[0]
            dino_model = segm_any.GroundingDinoModelLoader().main(dino_model)[0]
            sam_dino_model = sam_model, dino_model
        else:
            sam_model, dino_model = sam_dino_model

        image, mask = segm_any.GroundingDinoSAMSegment().main(dino_model, sam_model, image, prompt, threshold)
        mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        return (image, mask, mask_image, sam_dino_model)

class ConditionalMath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": (any,),
                "b": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "condition": (['a>=b', 'a<=b', "True"],),
                "true_value": ("STRING", { "multiline": False, "default": "" }),
                "false_value": ("STRING", { "multiline": False, "default": "" }),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT")
    RETURN_NAMES = ("INT", "FLOAT", "sel_num")    
    FUNCTION = "todo"
    CATEGORY = "eNode/etc"

    def todo(self, a, b, condition, true_value, false_value):
        if eval(condition):
            return (round(eval(true_value)), eval(true_value), 1)
        else:
            return (round(eval(false_value)), eval(false_value), 2)

class BatchControl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cur": ("INT", {"default": 0, "min": 1, "max": 10000}),
                "split_batch": ("STRING", {"default": "1,4,5", }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("sel_num",)    
    FUNCTION = "todo"
    CATEGORY = "eNode/etc"

    def todo(self, cur, split_batch):
        x = list(map(float, split_batch.split(',')))
        for i in range(len(x)):
            if cur <= x[i]:
                return (i+1,)

class SVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),    
                "init_image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 576, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "method": (["origin", "each"], {"default": "origin"},),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}, ),
            },
            "optional": {
                "basic_pipe": ("BASIC_PIPE",),              
            }
        }

    RETURN_TYPES = ("IMAGE", "BASIC_PIPE")
    RETURN_NAMES = ("images", "basic_pipe")    
    FUNCTION = "todo"
    CATEGORY = "eNode/Sampler"

    def todo(self, ckpt_name, init_image, width, height, method, motion_bucket_id, augmentation_level, seed, steps, cfg, sampler_name, scheduler, basic_pipe=None):
        return svdForSampler(ckpt_name, init_image, width, height, method, motion_bucket_id, augmentation_level, seed, steps, cfg, sampler_name, scheduler, basic_pipe)

class SplitImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "split_format": ("STRING", {"default": "2,3,1", }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("img1", "img2", "img3")    
    FUNCTION = "todo"
    CATEGORY = "eNode/etc"

    def todo(self, images, split_format):
        return splitImages_(images, split_format)

class YoloDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_name": (["segm/"+x for x in folder_paths.get_filename_list("ultralytics_segm")], {"default":"segm/person_yolov8m-seg.pt"},),             
                "image_list": ("IMAGE", ),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "labels": ("STRING", {"default": "person"}, ),                            
            },
            "optional": {
                "segm_detector": ("SEGM_DETECTOR", ),
            }
        }
    RETURN_TYPES = ("MASK", "SEGM_DETECTOR")
    RETURN_NAMES = ("masks", "segm_detector")
    FUNCTION = "todo"
    CATEGORY = "eNode/Detector"

    def todo(self, model_name, image_list, threshold, labels, segm_detector=None,):      
        return yoloDetector(model_name, image_list, threshold, labels, segm_detector)

FONTS_DIR = os.path.join(custom_nodes_path, "ComfyUI_essentials", "fonts")
class DrawTextList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", { "multiline": True, "dynamicPrompts": True, "default": "" }),
                "font": ([f for f in os.listdir(FONTS_DIR) if f.endswith('.ttf') or f.endswith('.otf')], ),
                "size": ("INT", { "default": 56, "min": 1, "max": 9999, "step": 1 }),
                "color": ("STRING", { "multiline": False, "default": "#FFFFFF" }),
                "background_color": ("STRING", { "multiline": False, "default": "#00000000" }),
                "shadow_distance": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_blur": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_color": ("STRING", { "multiline": False, "default": "#000000" }),
                "alignment": (["left", "center", "right"],),
                "width": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
                "height": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "todo"
    CATEGORY = "eNode/etc"

    def todo(self, text, font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, alignment, width, height):
        return drawTextList(text, font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, alignment, width, height)

