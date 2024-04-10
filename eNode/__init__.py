from .eNode import *

NODE_CLASS_MAPPINGS = {
    "BasicSetting":             BasicSetting,
    "ApplyAniDiff":             ApplyAniDiff,
    "ApplyContNet":             ApplyContNet,
    "ApplyIPadapterAdv":        ApplyIPadapterAdv,
    "TileSegsDetailer":         TileSegsDetailer,
    "Sampler01":                Sampler01
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BasicSetting":             "BasicSetting",
    "ApplyAniDiff":             "ApplyAniDiff", 
    "ApplyContNet":             "ApplyContNet",
    "ApplyIPadapterAdv":        "ApplyIPadapterAdv",
    "TileSegsDetailer":         "TileSegsDetailer", 
    "Sampler01":                "Sampler01"
}