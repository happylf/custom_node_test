import os       
import sys

# Reactor
'''
# fail---------------
custom_node_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ComfyUI_path = "D:\ComfyUI_windows_portable-test_01\ComfyUI"
sys.path.remove(ComfyUI_path)
Reactor_path = os.path.join(custom_node_path, "comfyui-reactor-node")
sys.path.insert(0, Reactor_path)
print(sys.path)
from nodes import reactor, LoadFaceModel
# fail---------------
'''
custom_node_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
Reactor_path = os.path.join(custom_node_path, "comfyui-reactor-node")
sys.path.append(Reactor_path)
from reactor_nodes import reactor, LoadFaceModel

def Reactor_apply(image):
    model_name = "Tamiya.safetensors"
    face_model = LoadFaceModel().load_model(model_name)[0]

    enabled = "ON"
    swap_model = "inswapper_128.onnx"
    facedetection = "retinaface_resnet50"
    face_restore_model = "GFPGANv1.4.pth"
    face_restore_visibility = 1.0
    codeformer_weight = 0.5
    detect_gender_source = "no"
    detect_gender_input = "no"
    source_faces_index = "0"
    input_faces_index = "0"
    console_log_level = "1"
    source_image=None    
    image, _ = reactor().execute(enabled,image, swap_model, detect_gender_source,
            detect_gender_input, source_faces_index, input_faces_index, console_log_level, 
            face_restore_model, face_restore_visibility, codeformer_weight, facedetection,
            source_image, face_model)
    
    return image
