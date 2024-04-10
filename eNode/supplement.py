import os
import sys

import nodes
import folder_paths
import comfy.utils

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

# essentials
import ComfyUI_essentials.essentials as essentials

# reactor node
reactor_path = os.path.join(custom_nodes_path, "comfyui-reactor-node")
sys.path.append(reactor_path)
import reactor_nodes

def imgResize_(image, w, h):
    keep_proportion = False
    interpolation="nearest"
    condition="always"
    out_img = essentials.ImageResize().execute(image,w,h,keep_proportion,interpolation,condition)[0]

    return out_img

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
