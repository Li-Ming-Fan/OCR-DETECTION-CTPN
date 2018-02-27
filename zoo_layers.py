# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


'''
tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1),
                 padding='valid', data_format='channels_last',
                 dilation_rate=(1,1), use_bias=True,
                 kernel_initializer=None, bias_initializer=init_ops.zeros_initializer(), 
                　kernel_regularizer=None, bias_regularizer=None, 
                 activation=None, activity_regularizer=None,
                 trainable=True,　name=None, reuse=None)
'''

def conv_layer(inputs, params, training):
    '''define a convolutional layer with params'''
    #
    # 输入数据维度为 4-D tensor: [batch_size, width, height, channels]
    #                         or [batch_size, height, width, channels]
    #
    # params = [filters, kernel_size, strides, padding, batch_norm, relu, name]
    #
    # batch_norm = True or False
    # relu = True or False
    #
    #
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    # kernel_initializer = tf.contrib.layers.xavier_initializer()    
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    outputs = tf.layers.conv2d(inputs, params[0], params[1], strides = params[2],
                               padding = params[3],
                               kernel_initializer = kernel_initializer,
                               bias_initializer = bias_initializer,
                               name = params[6])
    #
    if params[4]: # batch_norm
        outputs = norm_layer(outputs, training, name = params[6]+'/batch_norm')
    #
    if params[5]: # relu
        outputs = tf.nn.relu(outputs, name = params[6]+'/relu')
    #
    return outputs
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
        
'''
# tf.pad(tensor, paddings, mode='CONSTANT', name=None)
#
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1,], [2, 2]].
# rank of 't' is 2.
#
# padd1 = padd_layer(conv1, [[0,0],[0,0],[0,1],[0,0]], name='padd1')
'''

def padd_layer(tensor, paddings, mode='CONSTANT', name=None):
    ''' define padding layer '''
    return tf.pad(tensor, paddings, mode, name)
    #

'''
tf.layers.max_pooling2d(inputs, pool_size, strides,
                        padding='valid', data_format='channels_last', name=None)
'''

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
    
'''
fc = tf.layers.dense(rnn2, fc_size,
                     activation = tf.nn.relu,
                     kernel_initializer = weight_initializer,
                     bias_initializer = bias_initializer,
                     name = 'fc')
#
# dense operates on the last dim
#
# activation = tf.nn.sigmoid,
# activation = tf.nn.tanh,
# activation = tf.nn.relu,
#

'''

'''
blocks

'''

def block_resnet_others(inputs, layer_params, relu, training, name):
    '''define resnet block'''
    #
    # 1，图像大小不缩小，或者，图像大小只能降，1/2, 1/3, 1/4, ...
    # 2，深度，卷积修改
    #
    with tf.variable_scope(name):
        #
        #short_cut = tf.add(inputs, 0)
        short_cut = tf.identity(inputs)
        #
        shape_in = inputs.get_shape().as_list()
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
            item = [shape_out[3], 1, (1,1), 'same', True, False, 'shortcut_conv']
            #
            short_cut = conv_layer(short_cut, item, training)
            #
        #
        outputs = tf.add(inputs, short_cut, name = 'add')    
        #
        if relu: outputs = tf.nn.relu(outputs, 'last_relu')
        #
    #   
    return outputs
    #

def block_resnet(inputs, filters, flag_size, relu, training, name):
    '''define resnet block'''
    #
    with tf.variable_scope(name):
        #
        if flag_size == 1:  # same_size
            #
            item1 = [ filters, (3,3), (1,1), 'same', True,  True, 'conv1']
            item2 = [ filters, (3,3), (1,1), 'same', True, False, 'conv2']
            outputs = conv_layer(inputs, item1, training)
            outputs = conv_layer(outputs, item2, training)
            #
            outputs = tf.add(outputs, inputs, name = 'add')  
            if relu: outputs = tf.nn.relu(outputs, 'last_relu')
            #
            return outputs
            #
        elif flag_size == 2:  # half_size
            #
            outputs = padd_layer(inputs, [[0,0],[0,1],[0,1],[0,0]], name='padd')
            #
            item1 = [ filters, (3,3), (2,2), 'valid', True,  True, 'conv1']
            item2 = [ filters, (3,3), (1,1),  'same', True, False, 'conv2']
            outputs = conv_layer(outputs, item1, training)
            outputs = conv_layer(outputs, item2, training)
            #
            short_cut = maxpool_layer(inputs, (2,2), (2,2), 'valid', 'skip_pool')
            #
            item = [filters, 1, (1,1), 'same', True, False, 'skip_conv']
            short_cut = conv_layer(short_cut, item, training)  
            #
            outputs = tf.add(outputs, short_cut, name = 'add')  
            if relu: outputs = tf.nn.relu(outputs, 'last_relu')
            #
            return outputs
            #
        else:            
            print('flag_size not 1 or 2, in block_resnet_paper()')
            #
            return inputs
            #

def block_bottleneck(inputs, depth_arr, relu, training, name):
    '''define bottleneck block'''
    #
    #shape_in = inputs.get_shape().as_list()
    #
    #short_cut = inputs
    #
    with tf.variable_scope(name):
        #
        item1 = [depth_arr[0], (1,1), (1,1), 'same', True,  True, 'conv1']
        item2 = [depth_arr[1], (3,3), (1,1), 'same', True,  True, 'conv2']
        item3 = [depth_arr[2], (1,1), (1,1), 'same', True, False, 'conv3']
        #
        outputs = conv_layer(inputs, item1, training)
        outputs = conv_layer(outputs, item2, training) 
        outputs = conv_layer(outputs, item3, training) 
        #
        outputs = tf.add(outputs, inputs, name = 'add')
        if relu: outputs = tf.nn.relu(outputs, 'last_relu')
        #
    #    
    return outputs
    #

def block_inception(inputs, K, depth_arr, relu, training, name):
    ''' define inception-like block '''
    #
    with tf.variable_scope(name):
        #
        params_1 = [depth_arr[0], [1, K], (1,1), 'same', True, False, 'branch1']
        params_2 = [depth_arr[1], [K, 1], (1,1), 'same', True, False, 'branch2']
        params_3_1 = [depth_arr[2], [1, K], (1,1), 'same', True, False, 'branch3_1']
        params_3_2 = [depth_arr[3], [K, 1], (1,1), 'same', True, False, 'branch3_2']
        params_4 = [depth_arr[4], [K, K], (1,1), 'same', True, False, 'branch4']
        #
        branch_1 = conv_layer(inputs, params_1, training)
        branch_2 = conv_layer(inputs, params_2, training)
        branch_3 = conv_layer(inputs, params_3_1, training)
        branch_3 = conv_layer(branch_3, params_3_2, training)
        branch_4 = conv_layer(inputs, params_4, training)
        #
        outputs = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)
        #
        if relu: outputs = tf.nn.relu(outputs, 'last_relu')
        #    
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
    # time_major = True
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
    #
    rnn_output_stack = tf.concat(rnn_output, 2, name = 'output_stack')
    #rnn_output_stack = rnn_output[0] + rnn_output[1]
    
    return rnn_output_stack
    #
    
def gru_layer(input_sequence, sequence_length, rnn_size, scope):
    '''build bidirectional (concatenated output) lstm layer'''
    #
    # time_major = True
    #
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
    #
    rnn_output_stack = tf.concat(rnn_output, 2, name = 'output_stack')
    #rnn_output_stack = rnn_output[0] + rnn_output[1]
    
    return rnn_output_stack
    #

'''
output_tensor = graph.get_tensor_by_name('output_op:0')

# stop the gradient for fine-tuning
# this tensor could be an input for extention layers
#
output_sg = tf.stop_gradient(output_tensor) # identity operation

# shape
output_shape = output_sg.get_shape().as_list()

# further layers


参照：http://blog.csdn.net/u010911921/article/details/71079367

loss的最大值由FLT_MIN计算得到，
FLT_MIN是1.17549435e−38F​其对应的自然对数正好是-87.3356，这也就对应上了loss保持87.3356了

当softmax之前的feature值过大时，由于softmax先求指数，会超出float的数据范围，成为inf。
inf与其他任何数值的和都是inf，softmax在做除法时任何正常范围的数值除以inf都会变成0.
然后求loss就出现了87.3356的情况。


'''

