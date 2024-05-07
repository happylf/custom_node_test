import torch

class ColorValueForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("color", "red", "green", "blue")
    FUNCTION = "todo"
    CATEGORY = "eNode"

    def todo(self, image, mask):
        mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        mask_area = (image * 255).round().to(torch.int).clamp(min=1) * mask_image
        c1 = mask_area[mask_area>0].view(-1,3) # [r,g,b] list of masked area

        m = torch.mean(c1, dim=0).round().to(torch.int)
        m1 = m.numpy()

        color = m1[0]*256*256+m1[1]*256+m1[2]

        return color, m1[0], m1[1], m1[2]         

NODE_CLASS_MAPPINGS = {
    "ColorValueForMask":        ColorValueForMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorValueForMask":        "ColorValueForMask",
}   
