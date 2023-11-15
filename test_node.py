import os       #N02-04 for reading jason file
import json


class cl_TestNode01:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "input_text": ("STRING", {"default": "", "multiline": True})
                               
                              }}

    RETURN_TYPES = ("STRING", )
    FUNCTION = "test01"
    CATEGORY = "TestNode"

    def test01(self, input_text):
      output_text = input_text + ", mountain"
      print(f"output_text = {output_text}")
      return (output_text,)

def read_prompt_type_list(file_path):    #N02-05 read prompt type from jason file
    file_path = os.path.join(file_path, 'prompt_type_list.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_type_list = json.load(file)
        type_list = list()      #02-06 for #N02-02
        for item in prompt_type_list:
            type_list.append(item['TYPE'])
            
    return (prompt_type_list, type_list)        
        
class cl_TestNode02:        #N02-01 copy from test01, chaange '01' to '02'
    def __init__(self):     #02-07 python requirement
        pass
    
    @classmethod
    def INPUT_TYPES(self):  #02-08 change to 'self' - python requirement
        file_path = os.path.dirname(os.path.realpath(__file__))     #02-09 (to #02-05)    
        self.prompt_type_list, type_list = read_prompt_type_list(file_path)  #02-09 (from #02-05)
        
        return {"required": {"input_text": ("STRING", {"default": "", "multiline": True}),
                             "select_type": (type_list, {"default": "BATTLE"}),  #N02-02 add 1 more input
                              }}

    RETURN_TYPES = ("STRING", )
    FUNCTION = "test02"
    CATEGORY = "TestNode"

    def test02(self, input_text, select_type): #N02-03 add 1 more input
        for item in self.prompt_type_list:
            if item['TYPE'] == select_type:
                output_text = input_text + item['PROMPT'] 
                break
            
        return (output_text,)
