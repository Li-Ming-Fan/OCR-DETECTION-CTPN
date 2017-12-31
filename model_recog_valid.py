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

import model_recog
import model_recog_data


# data for test
dir_data = './data_test'
#
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
dir_results = dir_data + '/results'
#
str_dot_img_ext = '.PNG'
#
model_dir = './model'
model_name = 'model_detect'
#
num_classes = 27 #
#
height_norm = 32
#

#
# input-output, graph
x = tf.placeholder(tf.float32, (None, None, None, 3), name = 'x-input')
yT = tf.sparse_placeholder(tf.int32, shape = (None, None), name = 'y-input') 
w = tf.placeholder(tf.int32, (None,), name = 'w-input')
#
features, sequence_length = model_recog.conv_feat_layers(x, w, learn.ModeKeys.TRAIN) #INFER
result_logits = model_recog.rnn_recog_layers(features, sequence_length, num_classes)
#
loss = model_recog.ctc_loss_layer(yT, result_logits, sequence_length)
#
print('graph loaded')
#
# get test images
list_images = model_recog_data.getFilesInDirect(dir_images, str_dot_img_ext)
#
# test_result save-path
if os.path.exists(dir_results): shutil.rmtree(dir_results)
time.sleep(0.1)
os.mkdir(dir_results)
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
    NumImages = len(list_images)
    curr = 0
    for img_file in list_images:
        #
        # input data
        targets, images, width, batch = model_recog_data.getDataBatch(img_file, height_norm)
        #
        # targets_sparse_value
        tsv = model_recog.convert2SparseTensorValue(targets)
        #
        # results
        #w_arr = tf.constant(width, shape =(batch,))
        #w_arr = sess.run(w_arr)
        #
        w_arr = np.ones((batch,), dtype = np.int32) * width
        #

        #
        #print(w_arr)
        #        
        feed_dict = {x: images, w: w_arr}
        feat, seq_len = sess.run([features, sequence_length], feed_dict)
        #
        #
        print('targets')
        print(targets)
        #print('feat')
        #print(feat)
        print('feat shape')
        print(feat.shape)
        #print('seq_len')
        #print(seq_len)
        #print('width')
        #print(width)
        #

        #        
        feed_dict = {x: images, yT: tsv, w: w_arr}
        #
        results, loss_value = sess.run([result_logits, loss], feed_dict)
        #
        
        #print('results')
        #print(results)
        #print(results[0][0])
        print('results shape')
        print(results.shape)
        
        #
        curr += 1
        print('curr: %d / %d, loss: %f' % (curr, NumImages, loss_value))
        #
        # text result
        trans = model_recog_data.transResultsRNN(results)
        #
        #print(trans)
        #
        filename = os.path.basename(img_file)
        arr_str = os.path.splitext(filename)
        result_file = os.path.join(dir_results, arr_str[0] + '.txt')
        #
        with open(result_file, 'w') as fp:
            for seq in trans: fp.write(seq+'\n')
        #
        # image
        r = Image.fromarray(images[0][:,:,0] *255).convert('L')
        g = Image.fromarray(images[0][:,:,1] *255).convert('L')
        b = Image.fromarray(images[0][:,:,2] *255).convert('L')
        #
        file_target = os.path.join(dir_results, arr_str[0] + '.png')
        img_target = Image.merge("RGB", (r, g, b))
        img_target.save(file_target)
        #
    #

    