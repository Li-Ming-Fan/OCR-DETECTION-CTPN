# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

#
'''
tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1),
                 padding='valid', data_format='channels_last',
                 dilation_rate=(1,1), use_bias=True,
                 kernel_initializer=None, bias_initializer=init_ops.zeros_initializer(), 
                　kernel_regularizer=None, bias_regularizer=None, 
                 activation=None, activity_regularizer=None,
                 trainable=True,　name=None, reuse=None)
'''
#
def conv_layer(inputs, params, training):
    '''define a convolutional layer with layer_params'''
    #
    # 输入数据维度为 4-D tensor: [batch_size, width, height, channels]
    #                           [batch_size, height, width, channels]
    #
    #  Layer params:   Filts  K  Strides  Padding   Name   BatchNorm?
    # layer_params = [[  64, 3,  (1,1),  'same',  'conv1',  False], 
    #                 [  64, 3,  (3,3),  'same',  'conv2',  True]]
    #
    batch_norm = params[5] # Boolean
    #
    if batch_norm:
        activation = None
    else:
        activation = tf.nn.relu
    #
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    outputs = tf.layers.conv2d(inputs, 
                               filters = params[0],
                               kernel_size = params[1],
                               strides = params[2],
                               padding = params[3],
                               activation = activation,
                               kernel_initializer = kernel_initializer,
                               bias_initializer = bias_initializer,
                               name = params[4])
    #
    if batch_norm:
        outputs = norm_layer(outputs, training, name = params[4]+'/batch_norm')
        outputs = tf.nn.relu(outputs, name = params[4]+'/relu')
    #
    return outputs
#
def norm_layer_alter(inputs, training, name):
    '''define a batch-norm layer'''
    return tf.layers.batch_normalization(inputs, axis = 3, # batch first, channels last,
                                         training = training,
                                         name = name)
#
def norm_layer(x, train, eps = 1e-05, decay = 0.9, affine = True, name = None):
    #
    with tf.variable_scope(name, default_name='batch_norm'):
        #
        params_shape = [x.shape[-1]]              #
        batch_dims = list(range(0,len(x.shape) - 1))    #
        #
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer(),
                                          trainable=False)
        #
        def mean_var_with_update():
            #
            # axis = list(np.arange(len(x.shape) - 1))
            batch_mean, batch_variance = tf.nn.moments(x, batch_dims, name='moments')
            #
            with tf.control_dependencies([assign_moving_average(moving_mean, batch_mean, decay),
                                          assign_moving_average(moving_variance, batch_variance, decay)]):
                return tf.identity(batch_mean), tf.identity(batch_variance)
        #
        #mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        #
        if train:
            mean,variance = mean_var_with_update()
        else:
            mean,variance = moving_mean, moving_variance        
        #
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=True)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        #
        return x
        #
#
'''
tf.layers.max_pooling2d(inputs, pool_size, strides,
                        padding='valid', data_format='channels_last', name=None)
'''
#
def maxpool_layer(inputs, size, stride, padding, name):
    '''define a max-pooling layer'''
    return tf.layers.max_pooling2d(inputs, size, stride,
                                   padding = padding,
                                   name = name)
#
'''
tf.layers.average_pooling2d(inputs, pool_size, strides,
                            padding='valid', data_format='channels_last', name=None)
'''
def averpool_layer(inputs, size, stride, padding, name):
    '''define a average-pooling layer'''
    return tf.layers.average_pooling2d(inputs, size, stride,
                                   padding = padding,
                                   name = name)
#
#
def block_resnet(inputs, layer_params, training, name):
    '''define resnet block'''
    #    
    shape_in = inputs.get_shape().as_list()
    #
    short_cut = inputs   # 1，图像大小不缩小，深度不加深
    #                    # 2，图像大小只能降，1/2, 1/3, 1/4, ...
    #                    # 3，深度，卷积修改
    #
    with tf.variable_scope(name):
        #
        for item in layer_params:
            inputs = conv_layer(inputs, item, training)
        #
        shape_out = inputs.get_shape().as_list()
        #
        # 图片大小，缩小
        if shape_in[1] != shape_out[1] or shape_in[2] != shape_out[2]:
            #
            size = [shape_in[1]//shape_out[1], shape_in[2]//shape_out[2]]
            #
            short_cut = maxpool_layer(short_cut, size, size, 'valid', 'shortcut_pool')
            #
        #
        # 深度
        if shape_in[3] != shape_out[3]:
            #
            item = [shape_out[3], 1, (1,1), 'same', 'shortcut_conv', False]
            #
            short_cut = conv_layer(short_cut, item, training)
            #
        #
        outputs = tf.nn.relu(inputs + short_cut, 'last_relu')                
    #    
    return outputs
#
def bottleneck_block(inputs, depth_arr, name):
    '''define bottleneck block'''
    #
    #shape_in = inputs.get_shape().as_list()
    #
    #short_cut = inputs
    #
    with tf.variable_scope(name):
        #
        out = tf.layers.conv2d(inputs, depth_arr[0], 1, (1,1), 'same',
                               activation = tf.nn.relu, name = 'conv1')
        out = tf.layers.conv2d(out, depth_arr[1], 3, (1,1), 'same',
                               activation = tf.nn.relu, name = 'conv2')
        out = tf.layers.conv2d(out, depth_arr[2], 1, (1,1), 'same',
                               activation = None, name = 'conv3')
        #
        outputs = tf.nn.relu(inputs + out, 'last_relu')            
    #    
    return outputs
#
#
def incept_block(inputs, K, depth_arr, training, name):
    ''' define inception-like block '''
    #
    with tf.variable_scope(name):
        #
        params_1 = [depth_arr[0], [1, K], (1,1), 'same',  'branch1', False]
        params_2 = [depth_arr[1], [K, 1], (1,1), 'same',  'branch2', False]
        params_3_1 = [depth_arr[2], [1, K], (1,1), 'same',  'branch3_1', False]
        params_3_2 = [depth_arr[3], [K, 1], (1,1), 'same',  'branch3_2', False]
        params_4 = [depth_arr[4], [K, K], (1,1), 'same',  'branch4', False]
        #
        branch_1 = conv_layer(inputs, params_1, training)
        branch_2 = conv_layer(inputs, params_2, training)
        branch_3 = conv_layer(inputs, params_3_1, training)
        branch_3 = conv_layer(branch_3, params_3_2, training)
        branch_4 = conv_layer(inputs, params_4, training)
        #
        outputs = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)
    #
    return outputs
#
'''
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                              sequence_length = None, # 输入序列的实际长度（可选，默认为输入序列的最大长度）
                                                      # sequence_length must be a vector of length batch_size
                              initial_state_fw = None,  # 前向的初始化状态（可选）
                              initial_state_bw = None,  # 后向的初始化状态（可选）
                              dtype = None, # 初始化和输出的数据类型（可选）
                              parallel_iterations = None,
                              swap_memory = False,
                              time_major = False, 
                              # 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`. 
                              # 如果为false, tensor的形状必须为`[batch_size, max_time, depth]`. 
                              scope = None)
返回值：一个(outputs, output_states)的元组
其中，
1. outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。
假设time_major = false, tensor的shape为[batch_size, max_time, depth]。
实验中使用tf.concat(outputs, 2)将其拼接。
2. output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
output_state_fw和output_state_bw的类型为LSTMStateTuple。
LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
'''
def rnn_layer(input_sequence, sequence_length, rnn_size, scope):
    '''build bidirectional (concatenated output) lstm layer'''
    #
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    #
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    #
    # Include?
    #cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw, 
    #                                         input_keep_prob=dropout_rate )
    #cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw, 
    #                                         input_keep_prob=dropout_rate )
    
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = True,
                                                    dtype = tf.float32,
                                                    scope = scope)
    
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    #rnn_output_stack = tf.concat(rnn_output, 2, name = 'output_stack')
    rnn_output_stack = rnn_output[0] + rnn_output[1]
    
    return rnn_output_stack
#
def gru_layer(input_sequence, sequence_length, rnn_size, scope):
    '''build bidirectional (concatenated output) lstm layer'''
    
    # Default activation is tanh
    cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
    cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
    #
    # tf.nn.rnn_cell.GRUCell(num_units, input_size=None, activation=<function tanh>).
    # tf.contrib.rnn.GRUCell
    #
    # Include?
    #cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw, 
    #                                         input_keep_prob=dropout_rate )
    #cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw, 
    #                                         input_keep_prob=dropout_rate )
    
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = True,
                                                    dtype = tf.float32,
                                                    scope = scope)
    
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    #rnn_output_stack = tf.concat(rnn_output, 2, name = 'output_stack')
    rnn_output_stack = rnn_output[0] + rnn_output[1]
    
    return rnn_output_stack
#
# model
#
def conv_feat_layers(inputs, width, training):
    #
    # inputs: [1 h w d], 
    #

    #
    # [3,3; 1,1]
    # [9,9; 3,3], [9,9; 3,3], [11,11; 3,3]
    # [33,33; 9,9], [33,33; 9,9], [35,35; 9,9]
    # [70,70; 18,18], [70,70; 18,18], [72,72; 18,18]
    #
    # anchor width: 18,
    # anchor height: 9, 18, 36, 72,
    #
    
    #
    #training = (mode == learn.ModeKeys.TRAIN)
    #
    layer_params = [ [  64, 3, (1,1), 'valid',  'conv1', False], 
                     [  64, 3, (1,1),  'same',  'conv2',  True],
                     [  64, 2, (2,2), 'valid',  'pool2', False], # for pool
                     [ 128, 3, (1,1), 'valid',  'conv3', False], 
                     [ 128, 3, (1,1),  'same',  'conv4',  True],
                     [ 128, 3, (3,3), 'valid',  'pool4', False], # for pool
                     [ 256, 3, (1,1), 'valid',  'conv5', False],
                     [ 256, 3, (1,1),  'same',  'conv6',  True],
                     [ 256, 3, (3,3), 'valid',  'pool6', False], # for pool
                     [ 512, 3, (1,1), 'valid',  'feat6',  True]] # for feat
    #
    with tf.variable_scope("detect_conv"):
        #
        #for item in layer_params:
        #    inputs = conv_layer(inputs, item, training)
        #
        #inputs = incept_block(inputs, 3, [16,16,8,16,16], training, 'incept')
        #
        inputs = conv_layer(inputs, layer_params[0], training)
        inputs = conv_layer(inputs, layer_params[1], training)
        inputs = conv_layer(inputs, layer_params[2], training)
        #inputs = maxpool_layer(inputs, 2, (2,2), 'valid', 'pool2')
        #        
        item1 = [ 64, 3, (1,1),  'same',  'conv1', False]
        item2 = [ 64, 3, (1,1),  'same',  'conv2', False]
        inputs = block_resnet(inputs, [item1, item2], training, 'res1')
        #
        inputs = conv_layer(inputs, layer_params[3], training)
        inputs = conv_layer(inputs, layer_params[4], training)
        inputs = conv_layer(inputs, layer_params[5], training)
        #inputs = maxpool_layer(inputs, 3, (3,3), 'valid', 'pool4')
        #
        item1 = [ 128, 3, (1,1),  'same',  'conv1', False]
        item2 = [ 128, 3, (1,1),  'same',  'conv2', False]
        inputs = block_resnet(inputs, [item1, item2], training, 'res2')
        #
        inputs = conv_layer(inputs, layer_params[6], training)
        inputs = conv_layer(inputs, layer_params[7], training)
        inputs = conv_layer(inputs, layer_params[8], training)
        #inputs = maxpool_layer(inputs, 3, (3,3), 'valid', 'pool6')
        #
        #inputs = bottleneck_block(inputs, [256, 256, 256], 'bottleneck')
        #
        inputs = conv_layer(inputs, layer_params[9], training)
        #
        features = tf.squeeze(inputs, axis = 0, name = 'features') # squeeze
        # tf.expand_dims()
        #
        #print(features.shape)
        #
        
        #
        # Calculate resulting sequence length from original image widths
        #
        # -2, -2, //2-1,
        # //2-3, //2-3, //2//3-1,
        # //2//3-3, //2//3-3, //2//3//3-1,
        # //2//3//3-3,
        #
        two = tf.constant(2, dtype=tf.int32, name='two')
        three = tf.constant(3, dtype=tf.int32, name='three')
        #
        w = tf.floor_div(width, two)
        w = tf.floor_div(w, three)
        w = tf.floor_div(w, three)
        w = tf.subtract(w, three)        
        #
        # Vectorize
        sequence_length = tf.reshape(w, [-1], name='seq_len') 
        #
        
    #
    return features, sequence_length
    #
#
def rnn_detect_layers(features, sequence_length, num_anchors):
    #
    # Input features is [batchSize paddedSeqLen numFeatures]
    #
    #
    rnn_size = 512
    fc_size = 512  # 256, 384, 512
    #
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    with tf.variable_scope("detect_rnn"):
        #
        # Transpose to time-major order for efficiency
        #  --> [paddedSeqLen batchSize numFeatures]
        #
        rnn_sequence = tf.transpose(features, perm = [1, 0, 2], name = 'time_major')
        #
        rnn1 = rnn_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
        rnn2 = rnn_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
        #rnn3 = rnn_layer(rnn2, sequence_length, rnn_size, 'bdrnn3')
        #
        fc = tf.layers.dense(rnn2, fc_size,
                             activation = tf.nn.relu,
                             kernel_initializer = weight_initializer,
                             bias_initializer = bias_initializer,
                             name = 'fc')
        #
        # out
        rnn_cls = tf.layers.dense(fc, num_anchors * 2,
                                  activation = tf.nn.sigmoid,
                                  kernel_initializer = weight_initializer,
                                  bias_initializer = bias_initializer,
                                  name = 'text_cls')
        #
        rnn_ver = tf.layers.dense(fc, num_anchors * 2,
                                  activation = tf.nn.tanh,
                                  kernel_initializer = weight_initializer,
                                  bias_initializer = bias_initializer,
                                  name = 'text_ver')
        #
        rnn_hor = tf.layers.dense(fc, num_anchors * 2,
                                  activation = tf.nn.tanh,
                                  kernel_initializer = weight_initializer,
                                  bias_initializer = bias_initializer,
                                  name = 'text_hor')
        #
        # dense operates on last dim
        #
        
        #
        rnn_cls = tf.transpose(rnn_cls, perm = [1, 0, 2], name = 'rnn_cls')
        rnn_ver = tf.transpose(rnn_ver, perm = [1, 0, 2], name = 'rnn_ver')
        rnn_hor = tf.transpose(rnn_hor, perm = [1, 0, 2], name = 'rnn_hor')
    #
    return rnn_cls, rnn_ver, rnn_hor
    #
#
# loss
#
def detect_loss(rnn_cls, rnn_ver, rnn_hor, target_cls, target_ver, target_hor):        
    #
    # loss_cls    
    #
    rnn_cls_posi = rnn_cls * target_cls
    rnn_cls_neg = rnn_cls - rnn_cls_posi
    #
    pow_posi = tf.square(rnn_cls_posi - target_cls)
    pow_neg = tf.square(rnn_cls_neg)
    #
    mod_posi = tf.pow(pow_posi / 0.24, 5)    # 0.3, 0.2,     0.5,0.4
    mod_neg = tf.pow(pow_neg / 0.24, 5)      # 0.7, 0.6,
    mod_con = tf.pow(0.25 / 0.2, 5)
    #
    num_posi = tf.reduce_sum(target_cls) /2
    num_neg = tf.reduce_sum(target_cls + 1) /2 - num_posi * 2
    #
    loss_cls_posi = tf.reduce_sum(pow_posi * mod_posi) /2
    loss_cls_neg = tf.reduce_sum(pow_neg * mod_neg) /2
    #
    loss_cls = loss_cls_posi/num_posi + loss_cls_neg/num_neg
    #
    # loss reg
    #
    rnn_ver_posi = rnn_ver * target_cls
    rnn_hor_posi = rnn_hor * target_cls
    #
    rnn_ver_neg = rnn_ver - rnn_ver_posi
    rnn_hor_neg = rnn_hor - rnn_hor_posi
    #
    pow_ver_posi = tf.square(rnn_ver_posi - target_ver)
    pow_hor_posi = tf.square(rnn_hor_posi - target_hor)
    #
    pow_ver_neg = tf.square(rnn_ver_neg)
    pow_hor_neg = tf.square(rnn_hor_neg)
    #
    loss_ver_posi = tf.reduce_sum(pow_ver_posi * mod_con) / num_posi
    loss_hor_posi = tf.reduce_sum(pow_hor_posi * mod_con) / num_posi
    #
    loss_ver_neg = tf.reduce_sum(pow_ver_neg * mod_neg) / num_neg
    loss_hor_neg = tf.reduce_sum(pow_hor_neg * mod_neg) / num_neg
    #
    loss_ver = loss_ver_posi + loss_ver_neg
    loss_hor = loss_hor_posi + loss_hor_neg
    #
    
    #
    loss = tf.add(loss_cls, loss_ver + loss_hor, name = 'loss')
    #
    
    #
    return loss
    #
