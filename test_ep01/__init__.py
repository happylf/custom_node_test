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
    "FaceEnhance":          FaceEnhance
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
    "FaceEnhance":          "FaceEnhance"
}

