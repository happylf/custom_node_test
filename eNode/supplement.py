import os
import sys

import nodes
import folder_paths
import comfy.utils

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

# essentials
import ComfyUI_essentials.essentials as essentials

'''
# reactor node
reactor_path = os.path.join(custom_nodes_path, "comfyui-reactor-node")
sys.path.append(reactor_path)
import reactor_nodes
'''

def imgResize_(image, img_size):
    w, h = img_size
    keep_proportion = False
    interpolation="nearest"
    condition="always"
    out_img = essentials.ImageResize().execute(image,w,h,keep_proportion,interpolation,condition)[0]

    return out_img

def splitImages_(images, split_format):
    x = list(map(int, split_format.split(',')))
    img1 = images[:x[0],]
    img2 = images[x[0]:x[0]+x[1],]
    img3 = images[x[0]+x[1]:,]
    return img1, img2, img3

def imageCoordinate_(image):
    #1 create image_coordinate
    font = "arial.ttf"
    size = 16
    color = "#000000"
    background_color = "#FFFFFF"
    shadow_distance=shadow_blur=width=height=0
    shadow_color = "#000000"
    alignment = "left" 
    resize_source = False

    image_coordinate = image[0].unsqueeze(0)
    for y1 in range(100, image.shape[2], 100):
        for x1 in range(100, image.shape[1], 100):
            t, m = essentials.DrawText().execute(f"{str(int(x1/100)):s},{str(int(y1/100)):s}", font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, alignment, width, height)
            image_coordinate = nodes_mask.ImageCompositeMasked().composite(image_coordinate,t,x1,y1,resize_source,m)[0]

    return image_coordinate
'''
def reactorNode_(images, face_model):
    face_model = reactor_nodes.LoadFaceModel().load_model(face_model)[0]
    enabled = True
    input_image = images
    swap_model = "inswapper_128.onnx"
    detect_gender_source=detect_gender_input="no"
    source_faces_index=input_faces_index="0"
    console_log_level=1
    face_restore_model = "GFPGANv1.4.pth"
    face_restore_visibility = 1.0
    codeformer_weight = 0.50
    facedetection = "retinaface_resnet50"
    source_image=None
    images,_=reactor_nodes.reactor().execute(enabled,input_image,swap_model, detect_gender_source, detect_gender_input, source_faces_index,input_faces_index,console_log_level,face_restore_model,face_restore_visibility,codeformer_weight,facedetection,source_image,face_model)

    return images
'''