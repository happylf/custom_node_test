from .test_ep01 import*

NODE_CLASS_MAPPINGS = {
    "PreSampler":           PreSampler,
    "PreRegionalSampler":   PreRegionalSampler,
    "PreSampler_video":     PreSampler_video,
    "FromPresetSampler":    FromPresetSampler,
    "ToPresetSampler":      ToPresetSampler,
    "GptPrompt":            GptPrompt,
    "PreSampler_GPT":       PreSampler_GPT,
    "Sampler_GPT":          Sampler_GPT,
    "FaceEnhance":          FaceEnhance,
    "Yolo_Detector":        Yolo_Detector,
    "PreSampler_IPAdapter": PreSampler_IPAdapter,
    "Sampler_IPAdapter":    Sampler_IPAdapter,
    "Pre_PromptSchedule":   Pre_PromptSchedule,
    "Sampler_lcm":          Sampler_lcm    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreSampler":           "PreSampler",
    "PreRegionalSampler":   "PreRegionalSampler",
    "PreSampler_video":     "PreSampler_video",
    "FromPresetSampler":    "FromPresetSampler",
    "ToPresetSampler":      "ToPresetSampler",    
    "GptPrompt":            "GptPrompt",
    "PreSampler_GPT":       "PreSampler_GPT",
    "Sampler_GPT":          "Sampler_GPT",
    "FaceEnhance":          "FaceEnhance",
    "Yolo_Detector":        "Yolo_Detector",
    "PreSampler_IPAdapter": "PreSampler_IPAdapter",
    "Sampler_IPAdapter":    "Sampler_IPAdapter",
    "Pre_PromptSchedule":   "Pre_PromptSchedule",
    "Sampler_lcm":          "Sampler_lcm"    
}

