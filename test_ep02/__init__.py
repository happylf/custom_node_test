from .test_ep02         import *
from .compartment       import *

NODE_CLASS_MAPPINGS = {
    "PreSampler_CN":            PreSampler_CN,
    "set_IPAdapter":            set_IPAdapter,
    "Yolo_Detector":            Yolo_Detector,
    "Sampler01":                Sampler01,
    "Edit_pre_sampler":         Edit_pre_sampler,
    "CN_IPAd1_Backg":           CN_IPAd1_Backg,
    "LoopDecision01":           LoopDecision01
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreSampler_CN":            "PreSampler_CN",    
    "set_IPAdapter":            "set_IPAdapter",
    "Yolo_Detector":            "Yolo_Detector",
    "Sampler01":                "Sampler01",
    "Edit_pre_sampler":         "Edit_pre_sampler",
    "CN_IPAd1_Backg":           "CN_IPAd1_Backg",
    "LoopDecision01":           "LoopDecision01"
}

