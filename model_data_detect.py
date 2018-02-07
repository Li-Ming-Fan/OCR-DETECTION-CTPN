# -*- coding: utf-8 -*-

import os

from PIL import Image,ImageDraw
import numpy as np

from math import ceil


'''
#
dir_data = './data_generated'
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
#
'''

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
def getImageSize(img_file):
    #
    img = Image.open(img_file)
    return img.size  # (width, height)
    #
#
def getTargetTxtFile(img_file):
    #
    pre_dir = os.path.abspath(os.path.dirname(img_file)+os.path.sep+"..")
    txt_dir = os.path.join(pre_dir, 'contents')
    #
    filename = os.path.basename(img_file)
    arr_split = os.path.splitext(filename)
    filename = arr_split[0] + '.txt'
    #
    txt_file = os.path.join(txt_dir, filename)
    #
    return txt_file
    #
#
def getListContents(content_file):
    #
    contents = []
    #
    with open(content_file, 'r') as fp:
        lines = fp.readlines()
    #
    for line in lines:
        arr_str = line.split('|')
        item = list(map(lambda x: int(x), arr_str[0].split('-')))
        #
        contents.append([item, arr_str[1]])
        #
    return contents
#
#
#
def calculateTargetsAt(anchor_center, txt_list, anchor_heights):
    #
    # anchor_center = [hc, wc]
    #
    #
    # anchor width: 8,
    # anchor height: 8, 16, 32, 64, ...
    #
    # anchor stride: 8,8
    #

    #
    anchor_width = 8
    #
    
    #
    hc = anchor_center[0]
    wc = anchor_center[1]
    #
    maxIoU = 0
    anchor_posi = 0
    text_bbox = []
    #
    for item in txt_list:
        #
        # width: if more than half of the anchor is text, positive;
        # height: if more than half of the anchor is text, positive;        
        # heigth_IoU: of the 4 anchors, choose the one with max height_IoU;
        #
        bbox = item[0]
        #
        # horizontal
        flag = 0    
        #
        if (bbox[0] < wc and wc <= bbox[2]):
            flag = 1
        elif (wc < bbox[0] and bbox[2] < wc+8):
            if (bbox[0] - wc < wc+8 - bbox[2]):
                flag = 1
        elif (wc-8 < bbox[0] and bbox[2] < wc):
            if (bbox[2] - wc <= wc-8 - bbox[0]):
                flag = 1
        #
        if flag == 0: continue
        #
        # vertical
        #
        bcenter = (bbox[1] + bbox[3]) / 2.0
        #
        d0 = abs(hc - bcenter)
        dm = abs(hc-8 - bcenter)
        dp = abs(hc+8 - bcenter)
        #
        if (d0 < 8 and d0 <= dm and d0 < dp):
            pass
        else:
            continue        
        #
        #
        posi = 0
        #
        for ah in anchor_heights:
            #
            hah = ah //2  # half_ah
            #
            IoU = 1.0* (min(hc+hah, bbox[3])-max(hc-hah, bbox[1])) \
                      /(max(hc+hah, bbox[3])-min(hc-hah, bbox[1]))
            #
            if IoU > maxIoU:
                maxIoU = IoU
                anchor_posi = posi
                text_bbox = bbox
            #
            posi += 1
            #
        #
        break
    #
    # no text
    if maxIoU <= 0:  #
        #
        num_anchors = len(anchor_heights)
        #
        cls = [0, 0] * num_anchors
        ver = [0, 0] * num_anchors
        hor = [0, 0] * num_anchors
        #
        return cls, ver, hor
    #
    # text
    cls = []
    ver = []
    hor = []
    #
    for idx, ah in enumerate(anchor_heights):
        #
        if not idx == anchor_posi:
            cls.extend([0, 0])  #
            ver.extend([0, 0])
            hor.extend([0, 0])
            continue
        #
        cls.extend([1, 1])  #
        #
        half_ah = ah //2
        half_aw = anchor_width //2
        #
        anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]
        #
        ratio_bbox = [0, 0, 0, 0]
        #
        ratio = (text_bbox[0]-anchor_bbox[0]) /anchor_width
        if abs(ratio) < 1: 
            ratio_bbox[0] = ratio
        #
        # print(ratio)
        #
        ratio = (text_bbox[2]-anchor_bbox[2]) /anchor_width
        if abs(ratio) < 1:
            ratio_bbox[2] = ratio
        #
        # print(ratio)
        #
        ratio_bbox[1] = (text_bbox[1]-anchor_bbox[1]) /ah
        ratio_bbox[3] = (text_bbox[3]-anchor_bbox[3]) /ah
        #
        # print(ratio_bbox)
        #
        ver.extend([ratio_bbox[1], ratio_bbox[3]])
        hor.extend([ratio_bbox[0], ratio_bbox[2]]) 
        #
    #
    return cls, ver, hor
    #
#
# util function
#
def getImageAndTargets(img_file, anchor_heights):
    
    # img_data
    img = Image.open(img_file)
    img_data = np.array(img, dtype = np.float32)/255
    # height, width, channel
    #
    img_data = img_data[:,:,0:3]  # rgba
    #
    
    # texts
    txt_file = getTargetTxtFile(img_file)
    txt_list = getListContents(txt_file)
    #
    
    # targets
    img_size = getImageSize(img_file)
    # width, height    
    #
    # ///2, ///2, ///2,
    #
    width_feat = ceil(ceil(ceil(img_size[0]/2.0)/2.0)/2.0)
    height_feat = ceil(ceil(ceil(img_size[1]/2.0)/2.0)/2.0)
    #
    
    #
    num_anchors = len(anchor_heights)
    #
    target_cls = np.zeros((height_feat, width_feat, 2*num_anchors))
    target_ver = np.zeros((height_feat, width_feat, 2*num_anchors))
    target_hor = np.zeros((height_feat, width_feat, 2*num_anchors))
    #
    
    #
    # detection
    #
    # [3,3; 1,1],
    # [6,6; 2,2], [8,8; 2,2], [10,10; 2,2]
    # [20,20; 4,4], [22,22; 4,4], [24,24; 4,4]
    # [48,48; 8,8], [50,50; 8,8], [52,52; 8,8], 
    #
    # anchor width: 8,
    # anchor height: 8, 16, 32, 64,
    #
    # feature_layer --> receptive_field
    # [0,0] --> [0:52, 0:52]
    # [0,1] --> [0:52, 8:52+8]
    # [i,j] --> [8*i:52+8*i, 8*j:52+8*j]
    #
    # feature_layer --> anchor_center
    # [0,0] --> [26, 26]
    # [0,1] --> [26, 26+8]
    # [i,j] --> [26+8*i, 26+8*j]
    #
    
    for h in range(height_feat):
        #
        hc = 26 + 8*h  # anchor height center
        #
        for w in range(width_feat):
            #
            cls,ver,hor = calculateTargetsAt([hc, 26 + 8*w], txt_list, anchor_heights)
            #
            target_cls[h, w] = cls
            target_ver[h, w] = ver
            target_hor[h, w] = hor
            #
    #
    return [img_data], [height_feat, width_feat], target_cls, target_ver, target_hor
    #
#
def transResults(r_cls, r_ver, r_hor, anchor_heights, threshold):
    #
    # anchor width: 8,
    #
    
    #
    anchor_width = 8
    #
    
    #
    aw = anchor_width
    #
        
    #
    list_bbox = []
    #
    feat_shape = r_cls.shape
    #print(feat_shape)
    #
    for h in range(feat_shape[0]):
        #
        for w in range(feat_shape[1]):
            #
            if max(r_cls[h,w,:]) < threshold: continue
            #
            anchor_posi = np.argmax(r_cls[h,w,:])  # in r_cls
            anchor_id = anchor_posi //2  # in anchor_heights
            #
            #print(anchor_id)
            #print(r_cls[h,w,:])
            #
            #
            ah = anchor_heights[anchor_id]  #
            anchor_posi = anchor_id *2   # for retrieve in r_ver, r_hor
            #
            hc = 26 + 8*h  # anchor center
            wc = 26 + 8*w  # anchor center
            #
            half_ah = ah //2
            half_aw = anchor_width //2
            #
            anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]
            #
            text_bbox = [0, 0, 0, 0]
            #
            text_bbox[0] = anchor_bbox[0] + aw * r_hor[h,w,anchor_posi]
            text_bbox[1] = anchor_bbox[1] + ah * r_ver[h,w,anchor_posi]
            text_bbox[2] = anchor_bbox[2] + aw * r_hor[h,w,anchor_posi+1]
            text_bbox[3] = anchor_bbox[3] + ah * r_ver[h,w,anchor_posi+1]
            #
            list_bbox.append(text_bbox)
            #
    #
    return list_bbox
    #
#
def drawTextBox(img_file, text_bbox):
    #
    #打开图片，画图
    img_draw = Image.open(img_file)
    #
    draw = ImageDraw.Draw(img_draw)
    #
    for item in text_bbox:
        #
        xs = item[0]
        ys = item[1]
        xe = item[2]
        ye = item[3]
        #
        line_width = 1 # round(text_size/10.0)
        draw.line([(xs,ys),(xs,ye),(xe,ye),(xe,ys),(xs,ys)],
                   width=line_width, fill=(255,0,0))
        #
    #
    img_draw.save(img_file)
    #

#
if __name__ == '__main__':
    #
    print('draw target bbox ... ')
    #
    import model_config as meta
    #
    list_imgs = getFilesInDirect(meta.dir_images_valid, meta.str_dot_img_ext)
    #
    curr = 0
    NumImages = len(list_imgs)
    #
    # valid_result save-path
    if not os.path.exists(meta.dir_results_valid): os.mkdir(meta.dir_results_valid)
    #
    for img_file in list_imgs:
        #
        img_data, feat_size, target_cls, target_ver, target_hor = \
        getImageAndTargets(img_file, meta.anchor_heights)
        #
        # img_size = getImageSize(img_file) # width, height
        #
        curr += 1
        print('curr: %d / %d' % (curr, NumImages))
        #
        filename = os.path.basename(img_file)
        arr_str = os.path.splitext(filename)
        #
        # image
        r = Image.fromarray(img_data[0][:,:,0] *255).convert('L')
        g = Image.fromarray(img_data[0][:,:,1] *255).convert('L')
        b = Image.fromarray(img_data[0][:,:,2] *255).convert('L')
        #
        file_target = os.path.join(meta.dir_results_valid, 'target_' +arr_str[0] + '.png')
        img_target = Image.merge("RGB", (r, g, b))
        img_target.save(file_target)
        #
        # trans
        text_bbox = transResults(target_cls, target_ver, target_hor, meta.anchor_heights, meta.threshold)
        #
        drawTextBox(file_target, text_bbox)
        #
    #
    print('draw end.')
    #


