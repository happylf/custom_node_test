import os       
import sys
import math

# essentials
import ComfyUI_essentials.essentials as essentials

# 15 Puzzle
from .Puzzle15.model import *
from .Puzzle15.ai import *
from PIL import Image

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

def puz15_init():
    # ai
    ai_init()
    # aiMoveIndex = 0
    aiMoves = []

    puzzle = Puzzle()
    print("initial state")
    print(puzzle)
    aiMoves = idaStar(puzzle)
    print(aiMoves)

    return puzzle, aiMoves

def crop(image, width, height, x, y):
    x = int(x)
    y = int(y)
    to_x = int(width + x)
    to_y = int(height + y)
    img = image[:,y:to_y, x:to_x, :]
    return img

def puz15_crop(image, crop_w, crop_h):
    crop_img = []
    for i in range(4):
        for j in range(4):
            x = j * crop_w
            y = i * crop_h
            t_img = crop(image, crop_w, crop_h, x, y)
            crop_img.append(t_img)
    return crop_img

def drawText(text, w, h, size, bk_color):
    font = 'arialbd'
    color = '#000000'
    sd_distance = 0
    sd_blur = 0
    sd_color = '#000000'
    alignment = "center"
    return essentials.DrawText().execute(text,font,size,color,bk_color,sd_distance,sd_blur,sd_color,alignment,w,h)

def imgResize(image, w, h):
    keep_proportion = False
    interpolation="nearest"
    condition="always"
    out_img = essentials.ImageResize().execute(image,w,h,keep_proportion,interpolation,condition)[0]

    return out_img