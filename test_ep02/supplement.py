import os       
import sys
import math

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

def moving_calc(in_text, batch_size):
    in_text_list = [list(i.split(':')) for i in in_text.split('>')]
    mov_list = []
    for j in in_text_list:
        moving_list = [list(map(int, k.split(','))) for k in j[1].split('/')]
        for m in moving_list:
            if m[0] == 0:
                b1, x1, y1, r1 = m       
                continue 
            b2, x2, y2, r2 = m

            if b1 > batch_size:
                break
            if b2 == batch_size:
                b2 += 1
            for b in range(b1, b2): 
                match(j[0]):
                    case '1':       
                        x = x1 + int((x2-x1)*math.sqrt(b-b1)/math.sqrt(b2-b1))
                        y = y1 + int((y2-y1)*math.sqrt(b-b1)/math.sqrt(b2-b1))
                        r = r1 + int((r2-r1)*math.sqrt(b-b1)/math.sqrt(b2-b1))
                    case '2':       
                        x = x1 + int((x2-x1)*math.sqrt(b-b1)/math.sqrt(b2-b1))
                        y = y1 + int((y2-y1)*abs(math.sin((b-b1)/5)))
                        r = r1 + int((r2-r1)*math.sqrt(b-b1)/math.sqrt(b2-b1))
                    case _:
                        x = x1
                        y = y1
                        r = r1         

                if b == batch_size:
                    x=x2
                    y=y2
                    r=r2
                    mov_list.append([b, x, y, r])              
                    break
                mov_list.append([b, x, y, r])  
            b1 = b2
            x1 = x2
            y1 = y2
            r1 = r2 

    return mov_list 