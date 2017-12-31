# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:22:44 2017

@author: mingfan.li
"""

import os

from PIL import Image
import numpy as np

import random


'''
#
dir_data = './data_generated'
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
#
'''

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
        contents.append([item, arr_str[1][:-1]])
        #
    return contents
#
#
def getDataBatch(img_file, height_norm, batch_dir = './data_batch'):
    #
    txt_file = getTargetTxtFile(img_file)
    content_list = getListContents(txt_file)
    #
    img = Image.open(img_file)
    #img_data = np.array(img, dtype = np.float32)/255
    #
    #len_str_max = max(map(lambda x: len(x[1]), content_list))
    #
    labels = []
    images = []
    #
    # 将ASCII字符转换为对应的数值，即‘a’-->97，使用ord函数,ord('a')
    # 反之，使用chr函数，将数值转换为对应的ASCII字符，chr(97)
    #
    chosen = random.choice(range(len(content_list)))
    #
    # labels
    list_chars = list(map(lambda x: ord(x)-97, content_list[chosen][1]))
    labels.append(list_chars)
    #
    # images
    rect = img.crop(content_list[chosen][0])
    rect_size = rect.size
    #
    w = int(rect_size[0] * height_norm *1.0/rect_size[1])
    rect = rect.resize((w, height_norm))
    #
    images.append( np.array(rect, dtype = np.float32)/255 )
    #
    images[0] = images[0][:,:,0:3] #np.array(images[0])[:,:,0:3]    
    #
    # save
    r = Image.fromarray(images[0][:,:,0] *255).convert('L')
    g = Image.fromarray(images[0][:,:,1] *255).convert('L')
    b = Image.fromarray(images[0][:,:,2] *255).convert('L')
    #
    file_target = os.path.join(batch_dir, 'curr_' + str(0) + '.png')
    img_target = Image.merge("RGB", (r, g, b))
    img_target.save(file_target)
    #
    # to list
    images[0] = images[0].tolist()
    #
    
    #
    return labels, np.array(images), w, len(labels)
    #
    
    '''
    width_list = []
    #
    for item in content_list:
        #
        list_chars = list(map(lambda x: ord(x)-97, item[1]))
        #        
        # len norm
        #d = len_str_max - len(list_chars)
        #if d: list_chars.extend([0] * d)
        #
        # one-hot encoder
        # list_onehot = [np.eye(27)[x] for x in list_chars]
        # labels.append((np.array(list_onehot)).tolist())
        #
        #print(list_chars)
        labels.append(list_chars)
        #
        # images
        rect = img.crop(item[0])
        rect_size = rect.size
        #        
        w = int(rect_size[0] * height_norm *1.0/rect_size[1])   
        rect = rect.resize((w, height_norm))
        #
        images.append( np.array(rect, dtype = np.float32)/255 )
        #
        # width
        width_list.append(w)
        #
    #
    # norm width
    width_max = max(width_list)
    #
    if not os.path.exists(batch_dir): os.mkdir(batch_dir)
    #
    for i in range(len(width_list)):
        #
        if width_list[i] == width_max:
            #
            images[i] = np.array(images[i])[:,:,0:3]
            #           
        else:
            ap = np.ones( (height_norm, width_max - width_list[i], 3), dtype=np.float32 )
            #
            data = np.array(images[i])[:,:,0:3]
            #
            images[i] = np.concatenate((data, ap), 1)
        #  
        # save
        r = Image.fromarray(images[i][:,:,0] *255).convert('L')
        g = Image.fromarray(images[i][:,:,1] *255).convert('L')
        b = Image.fromarray(images[i][:,:,2] *255).convert('L')
        #
        file_target = os.path.join(batch_dir, str(i) + '.png')
        img_target = Image.merge("RGB", (r, g, b))
        img_target.save(file_target)
        #
        images[i] = images[i].tolist()
        #
    #    
    #return labels, images
    return labels, np.array(images), width_max, len(content_list)
    '''
    #
#
def transResultsRNN(results):
    #
    # to batch-major
    trans = np.transpose(results, (1, 0, 2))
    #
    trans = trans.tolist()
    #
    for idx, seq in enumerate(trans):
        str_seq = ''
        for idy, item in enumerate(seq):
            #
            str_seq += chr(np.argmax(item) + 97)
            #
            #print(item)
            #
        #
        trans[idx] = str_seq
    #
    return np.array(trans)
    #


#
#file_list = getFilesInDirect('./data_test/images', '.PNG')
#texts, images, width, batch = getDataBatch(file_list[0], 32)


#print(texts)
#print(list_content.shape)
#print(len(list_content))

#print('images')
#print(images.shape)
#print(width)
#print(batch)
