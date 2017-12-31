# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:50:37 2017

@author: mingfan.li
"""

import os
import shutil 
import time

import tensorflow as tf
from tensorflow.contrib import learn

from PIL import Image
import numpy as np

import model_detect
import model_detect_data


# data for validation
dir_data_valid = './data_test'
#
dir_images_valid = dir_data_valid + '/images'
dir_contents_valid = dir_data_valid + '/contents'
dir_results_valid = dir_data_valid + '/results'
#
str_dot_img_ext = '.PNG'
#
model_dir = './model_detect'
model_name = 'model_detect'
#
anchor_heights = [18, 27, 36, 45, 54, 72]
#
threshold = 0.5
#

#
# input-output, graph
x = tf.placeholder(tf.float32, (1, None, None, 3), name = 'x-input')
w = tf.placeholder(tf.int32, (None,), name = 'w-input') # width
t_cls = tf.placeholder(tf.float32, (None, None, None), name = 'c-input')
t_ver = tf.placeholder(tf.float32, (None, None, None), name = 'v-input')
t_hor = tf.placeholder(tf.float32, (None, None, None), name = 'h-input')
#
features, sequence_length = model_detect.conv_feat_layers(x, w, learn.ModeKeys.TRAIN) #INFER
rnn_cls, rnn_ver, rnn_hor = model_detect.rnn_detect_layers(features, sequence_length, len(anchor_heights))
#
loss = model_detect.detect_loss(rnn_cls, rnn_ver, rnn_hor, t_cls, t_ver, t_hor)
#
print('graph defined')
#
# get test images
list_images_valid = model_detect_data.getFilesInDirect(dir_images_valid, str_dot_img_ext)
#
# test_result save-path
if os.path.exists(dir_results_valid): shutil.rmtree(dir_results_valid)
time.sleep(0.1)
os.mkdir(dir_results_valid)
#
# to process
saver = tf.train.Saver()
with tf.Session() as sess:
    #
    tf.global_variables_initializer().run()
    #
    # restore with saved data
    ckpt = tf.train.get_checkpoint_state(model_dir)
    #
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    #
    # test
    NumImages = len(list_images_valid)
    curr = 0
    for img_file in list_images_valid:
        #
        # input data
        img_data, feat_size, target_cls, target_ver, target_hor = \
              model_detect_data.getImageAndTargets(img_file, anchor_heights)
        #
        img_size = model_detect_data.getImageSize(img_file) # width, height
        #
        w_arr = np.ones((feat_size[0],), dtype = np.int32) * img_size[0]
        #
        
        '''
        #
        print(feat_size)
        #
        
        feed_dict = {x: img_data, w: w_arr}
        feat, seq_len = sess.run([features, sequence_length], feed_dict)
        #
        #
        print('feat')
        print(feat)
        print('feat shape')
        print(feat.shape)
        print('seq_len')
        print(seq_len)
        #
        '''
        
        #        
        feed_dict = {x: img_data, w: w_arr, \
                     t_cls: target_cls, t_ver: target_ver, t_hor: target_hor}
        #
        r_cls, r_ver, r_hor, loss_value = sess.run([rnn_cls, rnn_ver, rnn_hor, loss], feed_dict)
        #        
        #
        curr += 1
        print('curr: %d / %d, loss: %f' % (curr, NumImages, loss_value))
        #
        #print(r_cls.shape)
        #print(r_cls)
        #
        filename = os.path.basename(img_file)
        arr_str = os.path.splitext(filename)
        #
        # image
        r = Image.fromarray(img_data[0][:,:,0] *255).convert('L')
        g = Image.fromarray(img_data[0][:,:,1] *255).convert('L')
        b = Image.fromarray(img_data[0][:,:,2] *255).convert('L')
        #
        # target
        file = os.path.join(dir_results_valid, arr_str[0] + '_target.png')
        img = Image.merge("RGB", (r, g, b))
        img.save(file)
        #
        text_bbox = model_detect_data.transResults(target_cls, target_ver, target_hor, \
                                                   anchor_heights, threshold)
        #
        model_detect_data.drawTextBox(file, text_bbox)
        #
        # result
        file = os.path.join(dir_results_valid, arr_str[0] + '_result.png')
        img = Image.merge("RGB", (r, g, b))
        img.save(file)
        #
        text_bbox = model_detect_data.transResults(r_cls, r_ver, r_hor, \
                                                   anchor_heights, threshold)
        #
        model_detect_data.drawTextBox(file, text_bbox)
        #
        #
        #break
        #
    #

    