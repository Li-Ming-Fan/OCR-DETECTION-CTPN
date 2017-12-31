# -*- coding: utf-8 -*-

import os
import shutil 
import time
import random

import tensorflow as tf
from tensorflow.contrib import learn

from PIL import Image
import numpy as np

import model_detect
import model_detect_data


# data for train
dir_data_train = './data_generated'
#
dir_images_train = dir_data_train + '/images'
dir_contents_train = dir_data_train + '/contents'

# data for validation
dir_data_valid = './data_test'
#
dir_images_valid = dir_data_valid + '/images'
dir_contents_valid = dir_data_valid + '/contents'
#
dir_results_valid = dir_data_valid + '/results'
#
str_dot_img_ext = '.png'
#
model_dir = './model_detect'
model_name = 'model_detect'
#
anchor_heights = [18, 27, 36, 45, 54, 72]
#
threshold = 0.5
#

#
TRAINING_STEPS = 2**20
#
LEARNING_RATE_BASE = 1e-5
DECAY_RATE = 0.9
DECAY_STAIRCASE = True
DECAY_STEPS = 2**9
MOMENTUM = 0.9
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
# train-related
#global_step = tf.contrib.framework.get_or_create_global_step()
global_step = tf.train.get_or_create_global_step()
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
print('graph defined')
#
# get training images
list_images_train = model_detect_data.getFilesInDirect(dir_images_train, str_dot_img_ext)
#
# model save-path
if not os.path.exists(model_dir): os.mkdir(model_dir)
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
#
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #
    tf.global_variables_initializer().run()
    #
    # restore with saved data
    ckpt = tf.train.get_checkpoint_state(model_dir)
    #
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    #
    # start training
    start_time = time.time()
    begin_time = start_time 
    #
    for i in range(TRAINING_STEPS):
        #
        img_file = random.choice(list_images_train)
        #
        #print(img_file)
        #
        # input data
        img_data, feat_size, target_cls, target_ver, target_hor = \
              model_detect_data.getImageAndTargets(img_file, anchor_heights)
        #
        img_size = model_detect_data.getImageSize(img_file) # width, height
        #
        w_arr = np.ones((feat_size[0],), dtype = np.int32) * img_size[0]
        #
        #        
        feed_dict = {x: img_data, w: w_arr, \
                     t_cls: target_cls, t_ver: target_ver, t_hor: target_hor}
        #
        _, loss_value, step, lr = sess.run([train_op, loss, global_step, learning_rate],
                                           feed_dict)
        #
        if i % 1 == 0:
            #
            saver.save(sess, os.path.join(model_dir, model_name), global_step = step)
            #
            curr_time = time.time()            
            #
            print('step: %d, loss: %g, lr: %g, sect_time: %.1f, total_time: %.1f, %s' %
                  (step, loss_value, lr, 
                   curr_time - begin_time,
                   curr_time - start_time,
                   os.path.basename(img_file)))
            #
            begin_time = curr_time
            #
        # validation
        if step % 10 == 0:
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
                feed_dict = {x: img_data, w: w_arr, \
                             t_cls: target_cls, t_ver: target_ver, t_hor: target_hor}
                #
                r_cls, r_ver, r_hor, loss_value = sess.run([rnn_cls, rnn_ver, rnn_hor, loss], feed_dict)
                #
                
                #
                curr += 1
                print('curr: %d / %d, loss: %f' % (curr, NumImages, loss_value))
                #
                filename = os.path.basename(img_file)
                arr_str = os.path.splitext(filename)
                #
                # image
                r = Image.fromarray(img_data[0][:,:,0] *255).convert('L')
                g = Image.fromarray(img_data[0][:,:,1] *255).convert('L')
                b = Image.fromarray(img_data[0][:,:,2] *255).convert('L')
                #
                file_target = os.path.join(dir_results_valid, str(step) + '_' +arr_str[0] + '.png')
                img_target = Image.merge("RGB", (r, g, b))
                img_target.save(file_target)
                #
                # trans
                text_bbox = model_detect_data.transResults(r_cls, r_ver, r_hor, anchor_heights, threshold)
                #
                model_detect_data.drawTextBox(file_target, text_bbox)
                #
                #
        #

'''

3.训练

训练的时候需要注意两点，(1)输入参数training=True，
(2)计算loss时，要添加以下代码（即添加update_ops到最后的train_op中）。
这样才能计算μ和σ的滑动平均（测试时会用到）

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
 

4.测试

测试时需要注意一点，输入参数training=False，其他就没了

 

5.预测

预测时比较特别，因为这一步一般都是从checkpoint文件中读取模型参数，然后做预测。
一般来说，保存checkpoint的时候，不会把所有模型参数都保存下来，因为一些无关数据会增大模型的尺寸，
常见的方法是只保存那些训练时更新的参数（可训练参数），如下：

var_list = tf.trainable_variables()
saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

但使用了batch_normalization，γ和β是可训练参数没错，μ和σ不是，它们仅仅是通过滑动平均计算出的，
如果按照上面的方法保存模型，在读取模型预测时，会报错找不到μ和σ。
更诡异的是，利用tf.moving_average_variables()也没法获取bn层中的μ和σ（也可能是我用法不对），
不过好在所有的参数都在tf.global_variables()中，因此可以这么写：

var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

按照上述写法，即可把μ和σ保存下来，读取模型预测时也不会报错，当然输入参数training=False还是要的。

注意上面有个不严谨的地方，因为我的网络结构中只有bn层包含moving_mean和moving_variance，
因此只根据这两个字符串做了过滤，如果你的网络结构中其他层也有这两个参数，但你不需要保存，
建议使用诸如bn/moving_mean的字符串进行过滤。

'''
   
