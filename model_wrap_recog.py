# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import os
import time
import random

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util

import model_define
import model_config as meta

import model_data_recog


class ModelRecog():
    #
    def __init__(self):
        #
        self.z_pb_file = os.path.join(meta.model_recog_dir, meta.model_recog_pb_file)
        #
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #    
        self.z_sess_config = config
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
            self.seq_len = self.graph.get_tensor_by_name('seq_len:0')
            self.result_logits = self.graph.get_tensor_by_name('recog_logits/Sigmoid:0')
            #
            self.result_i = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
            self.result_v = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
            self.result_s = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
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
        targets, img_data, width, batch = model_data_recog.getDataBatch(img_file, meta.height_norm)
        #
        w_arr = np.ones((batch,), dtype = np.int32) * width
        #
        # predication_result save-path
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        #
        with self.graph.as_default():              
            #
            feed_dict = {self.x: img_data, self.w: w_arr}
            #
            results, seq_length, d_i, d_v, d_s = sess.run([self.result_logits, self.seq_len, \
                                                           self.result_i, self.result_v, self.result_s], feed_dict)
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
            #
            decoded = tf.SparseTensorValue(indices = d_i, values = d_v, dense_shape = d_s)
            #
            trans = model_define.convert2ListLabels(decoded)
            # 
            #print(trans)
            #
            filename = os.path.basename(img_file)
            arr_str = os.path.splitext(filename)
            result_file = os.path.join(out_dir, arr_str[0] + '_predict.txt')
            #
            with open(result_file, 'w') as fp:
                for label in targets:
                    seq = list(map(model_data_recog.mapOrder2Char, label))
                    print(''.join(seq))
                    fp.write(''.join(seq) + '\n')
                    #
                for item in trans:
                    seq = list(map(model_data_recog.mapOrder2Char, item))
                    print(''.join(seq))
                    fp.write(''.join(seq) + '\n')
                    #
            #
        
    def z_define_graph_all(self, graph, train = True): # learn.ModeKeys.TRAIN  INFER
        #
        with graph.as_default():
            #
            x = tf.placeholder(tf.float32, (1, None, None, 3), name = 'x-input')
            w = tf.placeholder(tf.int32, (None,), name = 'w-input') # width
            #
            conv_feat, sequence_length = model_define.conv_feat_layers(x, w, False, train)   # train
            #
            rnn_feat = model_define.rnn_feat_layers(conv_feat, sequence_length)
            #
            result_logits = model_define.recog_results(rnn_feat, len(model_data_recog.alphabet) + 1)
            #
            result_decoded_list = model_define.decode_rnn_results_ctc_beam(result_logits, sequence_length)
            #
            print(' ')
            #print(result_logits.op.name)    # recog_logits/Sigmoid
            #print(sequence_length)          # seq_len
            print(result_decoded_list[0])   # CTCBeamSearchDecoder
            print(' ')
            #
            # SparseTensor(indices=Tensor("CTCBeamSearchDecoder:0", shape=(?, 2), dtype=int64),
            #              values=Tensor("CTCBeamSearchDecoder:1", shape=(?,), dtype=int64),
            #              dense_shape=Tensor("CTCBeamSearchDecoder:2", shape=(2,), dtype=int64))
            #
            
            #
            print('forward graph defined, training = %s' % train)
            #
            #
            y = tf.sparse_placeholder(tf.int32, shape = (None, None), name = 'y-input') 
            #
            # <tf.Operation 'y-input/shape' type=Placeholder>,
            # <tf.Operation 'y-input/values' type=Placeholder>,
            # <tf.Operation 'y-input/indices' type=Placeholder>]
            #            
            #print(graph.get_operations())
            #
            loss = model_define.ctc_loss_layer(y, result_logits, sequence_length)
            #
            #print(loss.op.name)  # loss
            #
            # train
            global_step = tf.train.get_or_create_global_step()
            #
            # Update batch norm stats [http://stackoverflow.com/questions/43234667]
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # 
            with tf.control_dependencies(extra_update_ops):
                #
                learning_rate = tf.train.exponential_decay(meta.LEARNING_RATE_BASE,
                                                           tf.train.get_global_step(),
                                                           meta.DECAY_STEPS,
                                                           meta.DECAY_RATE,
                                                           staircase = meta.DECAY_STAIRCASE,
                                                           name = 'learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                                   beta1 = meta.MOMENTUM)
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
        list_images_valid = model_data_recog.getFilesInDirect(meta.dir_images_valid, meta.str_dot_img_ext)
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
                ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # pb
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names = \
                                                                           ['recog_logits/Sigmoid','seq_len',\
                                                                            'CTCBeamSearchDecoder'])
                with tf.gfile.FastGFile(self.z_pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                #
                # variables
                #
                x = self.graph.get_tensor_by_name('x-input:0')
                w = self.graph.get_tensor_by_name('w-input:0')
                #
                seq_len = self.graph.get_tensor_by_name('seq_len:0')
                result_logits = self.graph.get_tensor_by_name('recog_logits/Sigmoid:0')
                #
                result_i = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
                result_v = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
                result_s = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
                #
                y_s = self.graph.get_tensor_by_name('y-input/shape:0')
                y_i = self.graph.get_tensor_by_name('y-input/indices:0')
                y_v = self.graph.get_tensor_by_name('y-input/values:0')
                #
                # <tf.Operation 'y-input/shape' type=Placeholder>,
                # <tf.Operation 'y-input/values' type=Placeholder>,
                # <tf.Operation 'y-input/indices' type=Placeholder>]
                #
                #
                loss_ts = self.graph.get_tensor_by_name('loss:0')
                #
                # test
                NumImages = len(list_images_valid)
                curr = 0
                print(' ')
                #
                for img_file in list_images_valid:
                    #
                    # input data
                    targets, images, width, batch = model_data_recog.getDataBatch(img_file, meta.height_norm)
                    #                    
                    w_arr = np.ones((batch,), dtype = np.int32) * width
                    #
                    # targets_sparse_value
                    tsv = model_define.convert2SparseTensorValue(targets)
                    #
                    feed_dict = {x: images, w: w_arr, y_s: tsv.dense_shape, y_i: tsv.indices, y_v: tsv.values}
                    results, loss, seq_length, d_i, d_v, d_s = sess.run([result_logits, loss_ts, seq_len, \
                                                                         result_i, result_v, result_s], feed_dict)
                    #
                    decoded = tf.SparseTensorValue(indices = d_i, values = d_v, dense_shape = d_s)
                    #
                    curr += 1
                    print('curr: %d / %d, loss: %f' % (curr, NumImages, loss))
                    #
                    #print(targets)               
                    #print(results)
                    #print(decoded)
                    #
                    trans = model_define.convert2ListLabels(decoded)
                    #            
                    #print(trans)
                    #
                    filename = os.path.basename(img_file)
                    arr_str = os.path.splitext(filename)
                    result_file = os.path.join(meta.dir_results_valid, str(step) + '_' + arr_str[0] + '.txt')
                    #
                    with open(result_file, 'w') as fp:
                        for label in targets:
                            seq = list(map(model_data_recog.mapOrder2Char, label))
                        print(''.join(seq))
                        fp.write(''.join(seq) + '\n')
                        #
                        for item in trans:
                            seq = list(map(model_data_recog.mapOrder2Char, item))
                            print(''.join(seq))
                            fp.write(''.join(seq) + '\n')
                    #
                    # image
                    r = Image.fromarray(images[0][:,:,0] *255).convert('L')
                    g = Image.fromarray(images[0][:,:,1] *255).convert('L')
                    b = Image.fromarray(images[0][:,:,2] *255).convert('L')
                    #
                    file_target = os.path.join(meta.dir_results_valid, str(step) + '_' + arr_str[0] + '.png')
                    img_target = Image.merge("RGB", (r, g, b))
                    img_target.save(file_target)
                    #
                    #
                
                #
                print('validation finished')
                print(' ')
                #
        
    
    def train_and_valid(self, load_from_pretrained = True):
        #
        # get training images
        list_images_train = model_data_recog.getFilesInDirect(meta.dir_images_train, meta.str_dot_img_ext)
        #
        # model save-path
        if not os.path.exists(meta.model_recog_dir): os.mkdir(meta.model_recog_dir)
        #
        # training graph
        self.z_graph = tf.Graph()
        #
        self.z_define_graph_all(self.z_graph, True)
        #
        # load from pretained
        list_ckpt = model_data_recog.getFilesInDirect(meta.model_recog_dir, '.meta')
        #
        print(' ')
        #
        if len(list_ckpt) > 0:
            print('model_recog ckpt already exists, no need to load common tensors.')            
        elif load_from_pretrained == False:
            print('not to load common tensors, by manual setting.')
        else:
            print('load common tensors from pretrained detection model.')
            self.z_load_from_pretrained_detection_model()
        print(' ')
        #
        # restore and train
        with self.z_graph.as_default():
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
                ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # variables
                #
                x = self.z_graph.get_tensor_by_name('x-input:0')
                w = self.z_graph.get_tensor_by_name('w-input:0')
                #
                y_s = self.z_graph.get_tensor_by_name('y-input/shape:0')
                y_i = self.z_graph.get_tensor_by_name('y-input/indices:0')
                y_v = self.z_graph.get_tensor_by_name('y-input/values:0')
                #
                # <tf.Operation 'y-input/shape' type=Placeholder>,
                # <tf.Operation 'y-input/values' type=Placeholder>,
                # <tf.Operation 'y-input/indices' type=Placeholder>]
                #
                #
                loss = self.z_graph.get_tensor_by_name('loss:0')
                #
                global_step = self.z_graph.get_tensor_by_name('global_step:0')
                learning_rate = self.z_graph.get_tensor_by_name('learning_rate:0')
                train_op = self.z_graph.get_tensor_by_name('train_op/control_dependency:0')
                #
                print('begin to train ...')
                #
                # start training
                start_time = time.time()
                begin_time = start_time 
                #
                for i in range(meta.TRAINING_STEPS):
                    #
                    img_file = random.choice(list_images_train)
                    #
                    #print(img_file)
                    #
                    # input data
                    targets, images, width, batch = model_data_recog.getDataBatch(img_file, meta.height_norm)
                    #                    
                    w_arr = np.ones((batch,), dtype = np.int32) * width
                    #
                    # targets_sparse_value
                    tsv = model_define.convert2SparseTensorValue(targets)
                    #
                    #
                    feed_dict = {x: images, w: w_arr, y_s: tsv.dense_shape, y_i: tsv.indices, y_v: tsv.values}
                    #
                    # sess.run
                    _, loss_value, step, lr = sess.run([train_op, loss, global_step, learning_rate], \
                                                        feed_dict)
                    #
                    if i % 1 == 0:
                        #
                        # ckpt
                        saver.save(sess, os.path.join(meta.model_recog_dir, meta.model_recog_name), \
                                   global_step = step)
                        #
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
                        self.validate(step, self.z_valid_option)
                        #
        #
    
    def z_load_from_pretrained_detection_model(self):
        #
        # model save-path
        # if not os.path.exists(meta.model_recog_dir): os.mkdir(meta.model_recog_dir)
        #
        # training graph
        # self.z_graph = tf.Graph()
        # self.z_define_graph_all(self.z_graph, True)
        #
        print('check common tensors to load ...')
        #
        comm_op_list = []
        #
        op_list = self.z_graph.get_operations()
        for op in op_list:
            if 'comm' in op.name and 'train' not in op.name and 'Variable' in op.type:
                #
                #print(op)
                #print(op.name)
                #print(op.type)
                #
                op_tensor = self.z_graph.get_tensor_by_name(op.name + ':0')
                #
                comm_op_list.append(op_tensor)
                #
        #
        print('checked.')
        #
        with self.z_graph.as_default():
            #
            saver = tf.train.Saver(comm_op_list)
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
                    print('loading comm_tensors of detection and recognition ...')
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('loaded.')
                else:
                    print('NO pretrained detection model.')
                #
        #

#
