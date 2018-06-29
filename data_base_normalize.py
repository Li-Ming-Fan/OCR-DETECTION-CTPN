# -*- coding: utf-8 -*-


import os
from PIL import Image

import math


#
dir_images_base_prenorm = './images_base_prenorm'
dir_images_base = './images_base'
#
str_dot_img_ext = '.png'
#
width_norm = 800
height_norm = 600
#

#
if not os.path.exists(dir_images_base): os.mkdir(dir_images_base)
#

#
def getFilesInDirect(path, str_dot_ext):
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)  
        if os.path.splitext(file_path)[1] == str_dot_ext:  
            file_list.append(file_path)
            #print(file_path)
        #
    return file_list;
    #
#
def extractNormalizedFrom(img_file):
    #    
    img_pre = Image.open(img_file)
    img_size = img_pre.size  # (width, height)
    ratio_w = width_norm * 1.0/img_size[0]
    ratio_h = height_norm * 1.0/img_size[1]
    #
    if ratio_w > 1:
        if ratio_h > ratio_w:
            #h_target = height_norm
            img_pre = img_pre.resize((max(width_norm, int(img_size[0] * ratio_h)), height_norm))            
        else:
            #w_target = width_norm
            img_pre = img_pre.resize((width_norm, max(height_norm, int(img_size[1] * ratio_w))))  
        #
    #
    img_size = img_pre.size  # (width, height)
    rate_w = img_size[0] * 1.0/width_norm
    rate_h = img_size[1] * 1.0/height_norm
    #
    NumW = math.ceil(rate_w)
    NumH = math.ceil(rate_h)
    #
    StrideW = 0
    StrideH = 0
    if NumW > 1:
        StrideW = (img_size[0] - width_norm) * 1.0 / (NumW-1)
    if NumH > 1:
        StrideH = (img_size[1] - height_norm) * 1.0 / (NumH-1) 
    #
    filename = os.path.basename(img_file)
    arr_str = os.path.splitext(filename)
    filename = os.path.join(dir_images_base, arr_str[0])
    #
    num = 0
    #
    y_s = 0
    y_e = height_norm
    #
    for j in range(0, NumH):
        x_s = 0
        x_e = width_norm
        #
        for i in range(0, NumW):
            bbox = (x_s,y_s,x_e,y_e)
            region = img_pre.crop(bbox)
            #
            region_file = filename + '_' + str(num) + arr_str[1]
            region.save(region_file)
            #
            #print(bbox)
            #print(region_file)
            #
            num += 1
            #
            x_s += StrideW
            x_e += StrideW
            #
        y_s += StrideH
        y_e += StrideH
        # 
    #
    return num
    #
    
#
#
img_list = getFilesInDirect(dir_images_base_prenorm, str_dot_img_ext)
#
NumImages = len(img_list)
count = 0
#
for img_file in img_list:
    count += 1
    #  
    print('current: %d / %d' %(count, NumImages))
    #
    extractNormalizedFrom(img_file)
    #
#
print('finished')
#
    
    
    
    