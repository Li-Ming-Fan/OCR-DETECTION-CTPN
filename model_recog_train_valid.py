# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:50:37 2017

@author: mingfan.li
"""

import os
import shutil 
import time
import random

import tensorflow as tf
from tensorflow.contrib import learn

from PIL import Image
import numpy as np

import model_recog
import model_recog_data


# data for train
dir_data = './data_generated'
#
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
#
str_dot_img_ext = '.PNG'
#
# data for test
dir_data_valid = './data_test'
#
dir_images_valid = dir_data_valid + '/images'
dir_contents_valid = dir_data_valid + '/contents'
#
dir_results = dir_data_valid + '/results'

#
#
model_dir = './model'
model_name = 'model_recog'
#
num_classes = 27  #
#
height_norm = 32
#

#
TRAINING_STEPS = 2**20
#
LEARNING_RATE_BASE = 1e-5
DECAY_STEPS = 2**10
DECAY_RATE = 0.9
DECAY_STAIRCASE = False
MOMENTUM = 0.9
#


#
# input-output, graph
x = tf.placeholder(tf.float32, (None, None, None, 3), name = 'x-input')
yT = tf.sparse_placeholder(tf.int32, shape = (None, None), name = 'y-input') 
w = tf.placeholder(tf.int32, (None,), name = 'w-input')
#
features, sequence_length = model_recog.conv_feat_layers(x, w, learn.ModeKeys.TRAIN)
result_logits = model_recog.rnn_recog_layers(features, sequence_length, num_classes)
#
loss = model_recog.ctc_loss_layer(yT, result_logits, sequence_length)
#
# train-related
global_step = tf.contrib.framework.get_or_create_global_step()
#
# Update batch norm stats [http://stackoverflow.com/questions/43234667]
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#
with tf.control_dependencies(extra_update_ops):
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               tf.train.get_global_step(),
                                               DECAY_STEPS,
                                               DECAY_RATE,
                                               staircase = DECAY_STAIRCASE,
                                               name = 'learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                       beta1 = MOMENTUM)
    train_op = tf.contrib.layers.optimize_loss(loss = loss,
                                               global_step = tf.train.get_global_step(),
                                               learning_rate = learning_rate,
                                               optimizer = optimizer)
                                               #variables = rnn_vars)
#
print('graph loaded')
#
# get training images
list_images = model_recog_data.getFilesInDirect(dir_images, str_dot_img_ext)
#
# model save-path
if not os.path.exists(model_dir): os.mkdir(model_dir)
#
# get validation images
list_images_valid = model_recog_data.getFilesInDirect(dir_images_valid, str_dot_img_ext)
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
    print('initialized, or restored')
    #
    # start training
    start_time = time.time()
    begin_time = start_time 
    #
    for i in range(TRAINING_STEPS):
        #
        img_file = random.choice(list_images)
        #
        #print(img_file)
        #
        # input data
        targets, images, width, batch = model_recog_data.getDataBatch(img_file, height_norm)
        #
        # targets_sparse_value
        tsv = model_recog.convert2SparseTensorValue(targets)
        #
        w_arr = np.ones((batch,), dtype = np.int32) * width
        #
        
        #print(targets)
        #print(w_arr)
        
        '''
        feed_dict = {x: images, w: w_arr}
        feat, seq_len = sess.run([features, sequence_length], feed_dict)
        #
        print('feat shape')
        print(feat.shape)
        print('width')
        print(width)
        '''
        
        # run     
        feed_dict = {x: images, yT: tsv, w: w_arr}
        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict)
        #
        if i % 1 == 0:
            #
            saver.save(sess, os.path.join(model_dir, model_name), global_step = step)
            #
            curr_time = time.time()
            #
            print('step: %d, loss: %g, sect_time: %f, total_time: %f, %s' %
                  (step, loss_value, 
                   curr_time - begin_time,
                   curr_time - start_time,
                   os.path.basename(img_file)))
            #
            begin_time = curr_time
            #
        # validation
        if step % 10 == 0:
            #
            NumImages = len(list_images_valid)
            curr = 0
            for img_file in list_images_valid:
                #
                # input data
                targets, images, width, batch = model_recog_data.getDataBatch(img_file, height_norm)
                #
                # targets_sparse_value
                tsv = model_recog.convert2SparseTensorValue(targets)
                #
                w_arr = np.ones((batch,), dtype = np.int32) * width
                #
                feed_dict = {x: images, yT: tsv, w: w_arr}
                results, loss_value = sess.run([result_logits, loss], feed_dict)
                #
                curr += 1
                print('curr: %d / %d, loss: %f' % (curr, NumImages, loss_value))
                #
                trans = model_recog_data.transResultsRNN(results)
                #
                #print(trans)
                #
                filename = os.path.basename(img_file)
                arr_str = os.path.splitext(filename)
                result_file = os.path.join(dir_results, str(step) + '_' + arr_str[0] + '.txt')
                #
                with open(result_file, 'w') as fp:
                    for seq in trans: fp.write(seq+'\n')
                # image
                r = Image.fromarray(images[0][:,:,0] *255).convert('L')
                g = Image.fromarray(images[0][:,:,1] *255).convert('L')
                b = Image.fromarray(images[0][:,:,2] *255).convert('L')
                #
                file_target = os.path.join(dir_results, str(step) + '_' + arr_str[0] + '.png')
                img_target = Image.merge("RGB", (r, g, b))
                img_target.save(file_target)
                #
        #
    
    #

    