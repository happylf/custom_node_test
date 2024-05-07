from .eNode import *

NODE_CLASS_MAPPINGS = {
    "BasicSetting":             BasicSetting,
    "ApplyAniDiff":             ApplyAniDiff,
    "ApplyContNet":             ApplyContNet,
    "ApplyIPadapterAdv":        ApplyIPadapterAdv,
    "TileSegsDetailer":         TileSegsDetailer,
    "Sampler01":                Sampler01,
    "MaskColorValue":           MaskColorValue,
    "MaskingAndImaging":        MaskingAndImaging,
    "CreateMask":               CreateMask,
    "CreateBatchMask":          CreateBatchMask,
    # "YoloESAM":                 YoloESAM,
    "SegmAnything":             SegmAnything,
    "ConditionalMath":          ConditionalMath,
    "BatchControl":             BatchControl,
    "SVDSampler":               SVDSampler,
    "SplitImages":              SplitImages,
    "YoloDetector":             YoloDetector,
    "DrawTextList":             DrawTextList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BasicSetting":             "BasicSetting",
    "ApplyAniDiff":             "ApplyAniDiff", 
    "ApplyContNet":             "ApplyContNet",
    "ApplyIPadapterAdv":        "ApplyIPadapterAdv",
    "TileSegsDetailer":         "TileSegsDetailer", 
    "Sampler01":                "Sampler01",
    "MaskColorValue":           "MaskColorValue",
    "MaskingAndImaging":        "MaskingAndImaging",
    "CreateMask":               "CreateMask",
    "CreateBatchMask":          "CreateBatchMask",
    # "YoloESAM":                 "YoloESAM",
    "SegmAnything":             "SegmAnything",
    "ConditionalMath":          "ConditionalMath",
    "BatchControl":             "BatchControl",
    "SVDSampler":               "SVDSampler",
    "SplitImages":              "SplitImages",
    "YoloDetector":             "YoloDetector",
    "DrawTextList":             "DrawTextList"
}