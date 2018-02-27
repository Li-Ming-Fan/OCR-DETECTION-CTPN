# -*- coding: utf-8 -*-


import os
import time
import random

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util

import model_comm_def as model_def
import model_comm_meta as meta

import model_detect_data



#
TRAINING_STEPS = 10000
#
LEARNING_RATE_BASE = 0.001
DECAY_RATE = 0.9
DECAY_STAIRCASE = True
DECAY_STEPS = 1000
#
MOMENTUM = 0.9
#


class ModelDetect():
    #
    def __init__(self):
        #
        self.z_pb_file = os.path.join(meta.model_detect_dir, meta.model_detect_pb_file)
        #        
        self.z_sess_config = tf.ConfigProto()
        # self.z_sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #
        self.z_valid_freq = 100
        self.z_valid_option = False
        #
        
    def load_pb_for_prediction(self, pb_file_path = None):
        #
        if pb_file_path == None: pb_file_path = self.z_pb_file 
        #
        if not os.path.exists(pb_file_path):
            print('ERROR: %s NOT exists, when load_pb_for_predict()' % pb_file_path)
            return -1
        #
        self.graph = tf.Graph()
        #
        with self.graph.as_default():
            #
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                #
                tf.import_graph_def(graph_def, name="")
                #
            #
            # change the input/output variables
            #
            self.x = self.graph.get_tensor_by_name('x-input:0')
            self.w = self.graph.get_tensor_by_name('w-input:0')
            #
            self.rnn_cls = self.graph.get_tensor_by_name('rnn_cls:0')
            self.rnn_ver = self.graph.get_tensor_by_name('rnn_ver:0')
            self.rnn_hor = self.graph.get_tensor_by_name('rnn_hor:0')
            #
                
        #
        print('graph loaded for prediction')
        #
        return 0
        #
        
    def create_session_for_prediction(self):
        #
        with self.graph.as_default():
            sess = tf.Session(config = self.z_sess_config)
            
            return sess
        #

    def predict(self, sess, img_file, out_dir = './results_prediction'):
        #
        # input data
        img_data, feat_size, target_cls, target_ver, target_hor = \
        model_detect_data.getImageAndTargets(img_file, meta.anchor_heights)
        #
        img_size = model_detect_data.getImageSize(img_file) # width, height
        #
        w_arr = np.ones((feat_size[0],), dtype = np.int32) * img_size[0]
        #
        # predication_result save-path
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        #
        with self.graph.as_default():              
            #
            feed_dict = {self.x: img_data, self.w: w_arr}
            #
            r_cls, r_ver, r_hor = sess.run([self.rnn_cls, self.rnn_ver, self.rnn_hor], feed_dict)
            #
            #
            filename = os.path.basename(img_file)
            arr_str = os.path.splitext(filename)
            #
            # image
            r = Image.fromarray(img_data[0][:,:,0] *255).convert('L')
            g = Image.fromarray(img_data[0][:,:,1] *255).convert('L')
            b = Image.fromarray(img_data[0][:,:,2] *255).convert('L')
            #
            file_target = os.path.join(out_dir, arr_str[0] + '_predict.png')
            img_target = Image.merge("RGB", (r, g, b))
            img_target.save(file_target)
            #
            # trans
            text_bbox = model_detect_data.transResults(r_cls, r_ver, r_hor, \
                                                       meta.anchor_heights, meta.threshold)
            #
            model_detect_data.drawTextBox(file_target, text_bbox)
            #
    
    @staticmethod
    def z_define_graph_all(graph, train = True): # learn.ModeKeys.TRAIN  INFER
        #
        with graph.as_default():
            #
            x = tf.placeholder(tf.float32, (1, None, None, 3), name = 'x-input')
            w = tf.placeholder(tf.int32, (None,), name = 'w-input') # width
            #
            conv_feat, sequence_length = model_def.conv_feat_layers(x, w, train)   # train
            rnn_cls, rnn_ver, rnn_hor = model_def.rnn_detect_layers(conv_feat, sequence_length, len(meta.anchor_heights))
            #
            # print(rnn_cls.op.name)
            #
            print('forward graph defined, training = %s' % train)
            #
            #
            t_cls = tf.placeholder(tf.float32, (None, None, None), name = 'c-input')
            t_ver = tf.placeholder(tf.float32, (None, None, None), name = 'v-input')
            t_hor = tf.placeholder(tf.float32, (None, None, None), name = 'h-input')
            #            
            # print(self.graph.get_operations())
            #
            loss = model_def.detect_loss(rnn_cls, rnn_ver, rnn_hor, t_cls, t_ver, t_hor)
            #
            # print(loss.op.name)
            #
            global_step = tf.train.get_or_create_global_step()
            #
            # Update batch norm stats [http://stackoverflow.com/questions/43234667]
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # 
            with tf.control_dependencies(extra_update_ops):
                #
                learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           tf.train.get_global_step(),
                                                           DECAY_STEPS,
                                                           DECAY_RATE,
                                                           staircase = DECAY_STAIRCASE,
                                                           name = 'learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                                   beta1 = MOMENTUM)
                # train_op =                     
                train_op = tf.contrib.layers.optimize_loss(loss = loss,
                                                           global_step = tf.train.get_global_step(),
                                                           learning_rate = learning_rate,
                                                           optimizer = optimizer,
                                                           name = 'train_op')  #variables = rnn_vars)
            #
            print('train graph defined, training = %s' % train)
            #
            print('global_step.op.name: ' + global_step.op.name)
            print('train_op.op.name: ' + train_op.op.name)                
            #
            #
                
    def validate(self, step, training):
        #
        # get validation images
        list_images_valid = model_detect_data.getFilesInDirect(meta.dir_images_valid, meta.str_dot_img_ext)
        #
        # valid_result save-path
        if not os.path.exists(meta.dir_results_valid): os.mkdir(meta.dir_results_valid)
        #
        # if os.path.exists(dir_results): shutil.rmtree(dir_results)
        # time.sleep(0.1)
        # os.mkdir(dir_results)
        #
        # validation graph
        self.graph = tf.Graph()
        #
        self.z_define_graph_all(self.graph, training)
        #
        with self.graph.as_default():
            #
            saver = tf.train.Saver()
            #
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            #
            with tf.Session(config = self.z_sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(meta.model_detect_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # pb
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names = \
                                                                           ['rnn_cls','rnn_ver','rnn_hor'])
                with tf.gfile.FastGFile(self.z_pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                #
                # variables
                #
                x = self.graph.get_tensor_by_name('x-input:0')
                w = self.graph.get_tensor_by_name('w-input:0')
                #
                rnn_cls = self.graph.get_tensor_by_name('rnn_cls:0')
                rnn_ver = self.graph.get_tensor_by_name('rnn_ver:0')
                rnn_hor = self.graph.get_tensor_by_name('rnn_hor:0')
                #
                t_cls = self.graph.get_tensor_by_name('c-input:0')
                t_ver = self.graph.get_tensor_by_name('v-input:0')
                t_hor = self.graph.get_tensor_by_name('h-input:0')
                #
                loss = self.graph.get_tensor_by_name('loss:0')
                #
                # test
                NumImages = len(list_images_valid)
                curr = 0
                for img_file in list_images_valid:
                    #
                    # input data
                    img_data, feat_size, target_cls, target_ver, target_hor = \
                    model_detect_data.getImageAndTargets(img_file, meta.anchor_heights)
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
                    file_target = os.path.join(meta.dir_results_valid, str(step) + '_' +arr_str[0] + '.png')
                    img_target = Image.merge("RGB", (r, g, b))
                    img_target.save(file_target)
                    #
                    # trans
                    text_bbox = model_detect_data.transResults(r_cls, r_ver, r_hor, \
                                                               meta.anchor_heights, meta.threshold)
                    #
                    model_detect_data.drawTextBox(file_target, text_bbox)
                    #
                #
                print('validation finished')
                #
        
    
    def train_and_valid(self):
        #
        # get training images
        list_images_train = model_detect_data.getFilesInDirect(meta.dir_images_train, meta.str_dot_img_ext)
        #      
        # model save-path
        if not os.path.exists(meta.model_detect_dir): os.mkdir(meta.model_detect_dir)
        #                   
        # training graph
        self.z_graph = tf.Graph()
        #
        self.z_define_graph_all(self.z_graph, True)
        #
        with self.z_graph.as_default():
            #
            saver = tf.train.Saver()
            #
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95, allocator_type = 'BFC')
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            #
            with tf.Session(config = self.z_sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(meta.model_detect_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                
                #
                # variables
                #
                x = self.z_graph.get_tensor_by_name('x-input:0')
                w = self.z_graph.get_tensor_by_name('w-input:0')
                #
                t_cls = self.z_graph.get_tensor_by_name('c-input:0')
                t_ver = self.z_graph.get_tensor_by_name('v-input:0')
                t_hor = self.z_graph.get_tensor_by_name('h-input:0')
                #
                loss = self.z_graph.get_tensor_by_name('loss:0')
                #
                global_step = self.z_graph.get_tensor_by_name('global_step:0')
                learning_rate = self.z_graph.get_tensor_by_name('learning_rate:0')
                train_op = self.z_graph.get_tensor_by_name('train_op/control_dependency:0')
                #
                
                #
                print('begin to train ...')
                #
                # start training
                start_time = time.time()
                begin_time = start_time 
                #
                for i in range(TRAINING_STEPS):
                    #
                    img_file = random.choice(list_images_train)
                    #
                    # print(img_file)
                    #
                    # input data
                    img_data, feat_size, target_cls, target_ver, target_hor = \
                    model_detect_data.getImageAndTargets(img_file, meta.anchor_heights)
                    #
                    img_size = model_detect_data.getImageSize(img_file) # width, height
                    #
                    w_arr = np.ones((feat_size[0],), dtype = np.int32) * img_size[0]
                    #
                    #
                    feed_dict = {x: img_data, w: w_arr, \
                                 t_cls: target_cls, t_ver: target_ver, t_hor: target_hor}
                    #                    
                    #print(img_data.size)
                    #print(feat_size)
                    #print(w_arr)
                    #
                    #rnn_cls_v = sess.run(seq_len, feed_dict)
                    #print(len(rnn_cls_v))
                    #
                    #loss_value = sess.run(loss, feed_dict)
                    #print('sess.run, loss = %g' % loss_value)
                    #
                    _, loss_value, step, lr = sess.run([train_op, loss, global_step, learning_rate], \
                                                        feed_dict)
                    #
                    if i % 1 == 0:
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
                    #
                    # validation
                    if step % self.z_valid_freq == 0:
                        #
                        # ckpt
                        saver.save(sess, os.path.join(meta.model_detect_dir, meta.model_detect_name), \
                                   global_step = step)
                        #
                        self.validate(step, self.z_valid_option)
                        #
        #

#
'''
#
graph 相关的操作

add_to_collection(name,value)
as_default()
device(*args,**kwds)

with g.device('/gpu:0'):
  # All operations constructed in this context will be placed
  # on GPU 0.
  with g.device(None):
    # All operations constructed in this context will have no
    # assigned device.

# Defines a function from `Operation` to device string.
def matmul_on_gpu(n):
  if n.type == "MatMul":
    return "/gpu:0"
  else:
    return "/cpu:0"

with g.device(matmul_on_gpu):
  # All operations of type "MatMul" constructed in this context
  # will be placed on GPU 0; all other operations will be placed
  # on CPU 0.
  
finalize()
get_all_collection_keys()
get_operation_by_name(name)
get_operations()
get_tensor_by_name(name)
is_feedable(tensor)           # 作用：要是一个tensor能够被feed的话，返回True。
is_fetchable(tensor_or_op)
name_scope(*args,**kwds) 

with tf.Graph().as_default() as g:
  c = tf.constant(5.0, name="c")
  assert c.op.name == "c"
  c_1 = tf.constant(6.0, name="c")
  assert c_1.op.name == "c_1"

  # Creates a scope called "nested"
  with g.name_scope("nested") as scope:
    nested_c = tf.constant(10.0, name="c")
    assert nested_c.op.name == "nested/c"

    # Creates a nested scope called "inner".
    with g.name_scope("inner"):
      nested_inner_c = tf.constant(20.0, name="c")
      assert nested_inner_c.op.name == "nested/inner/c"

    # Create a nested scope called "inner_1".
    with g.name_scope("inner"):
      nested_inner_1_c = tf.constant(30.0, name="c")
      assert nested_inner_1_c.op.name == "nested/inner_1/c"

      # Treats `scope` as an absolute name scope, and
      # switches to the "nested/" scope.
      with g.name_scope(scope):
        nested_d = tf.constant(40.0, name="d")
        assert nested_d.op.name == "nested/d"

        with g.name_scope(""):
          e = tf.constant(50.0, name="e")
          assert e.op.name == "e"

'''

