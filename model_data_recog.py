# -*- coding: utf-8 -*-

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
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet += ''' ,./<>?;':"[]\{}|-=_+~!@#$%^&*()
            '''.strip()
#
alphabet_blank = '`'
#


def define_alphabet():
    #
    pass

def mapChar2Order(char): return alphabet.index(char)
def mapOrder2Char(order):
    if order == len(alphabet):
        return alphabet_blank
    else:
        return alphabet[order]
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

def getImageSize(img_file):
    #
    img = Image.open(img_file)
    return img.size  # (width, height)
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
# util function
#
def getDataBatch(img_file, height_norm, batch_dir = './data_batch'):
    #
    if not os.path.exists(batch_dir): os.mkdir(batch_dir)
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
    
    
    #
    chosen = random.choice(range(len(content_list)))
    #
    # labels
    list_chars = list(map(mapChar2Order, content_list[chosen][1]))
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
        list_chars = list(map(mapChar2Order, item[1]))
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

'''
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
            str_seq += mapOrder2Char(np.argmax(item))
            #
            #print(item)
            #
        #
        trans[idx] = str_seq
    #
    return np.array(trans)
    #

'''

