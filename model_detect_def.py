# -*- coding: utf-8 -*-


import tensorflow as tf

import zoo_layers as layers


#
# model
#
def conv_feat_layers(inputs, width, training):
    #
    # convolutional features maps for detection
    #

    #
    # detection
    #
    # [3,1; 1,1],
    # [9,2; 3,2], [9,2; 3,2], [9,2; 3,2]
    # [18,4; 6,4], [18,4; 6,4], [18,4; 6,4]
    # [36,8; 12,8], [36,8; 12,8], [36,8; 12,8], 
    #
    # anchor width:  8,
    # anchor height: 6, 12, 24, 36,
    #
    # feature_layer --> receptive_field
    # [0,0] --> [0:36, 0:8]
    # [0,1] --> [0:36, 8:8+8]
    # [i,j] --> [12*i:36+12*i, 8*j:8+8*j]
    #
    # feature_layer --> anchor_center
    # [0,0] --> [18, 4]
    # [0,1] --> [18, 4+8]
    # [i,j] --> [18+12*i, 4+8*j]
    #

    #
    layer_params = [ [  64, (3,3), (1,1),  'same', True, True, 'conv1'], 
                     [ 128, (3,3), (1,1),  'same', True, True, 'conv2'],
                     [ 128, (2,2), (2,2), 'valid', True, True, 'pool1'], # for pool
                     [ 128, (3,3), (1,1),  'same', True, True, 'conv3'], 
                     [ 256, (3,3), (1,1),  'same', True, True, 'conv4'],
                     [ 256, (2,2), (2,2), 'valid', True, True, 'pool2'], # for pool
                     [ 256, (3,3), (1,1),  'same', True, True, 'conv5'],
                     [ 512, (3,3), (1,1),  'same', True, True, 'conv6'],
                     [ 512, (3,2), (3,2), 'valid', True, True, 'pool3'], # for pool
                     [ 512, (3,1), (1,1), 'valid', True, True, 'conv_feat'] ] # for feat
    
    #
    with tf.variable_scope("conv_comm"):
        #        
        inputs = layers.conv_layer(inputs, layer_params[0], training)
        inputs = layers.conv_layer(inputs, layer_params[1], training)
        inputs = layers.padd_layer(inputs, [[0,0],[0,1],[0,1],[0,0]], name='padd1')
        #inputs = layers.conv_layer(inputs, layer_params[2], training)
        inputs = layers.maxpool_layer(inputs, (2,2), (2,2), 'valid', 'pool1')
        #        
        params = [[ 128, 3, (1,1), 'same', True,  True, 'conv1'], 
                  [ 128, 3, (1,1), 'same', True, False, 'conv2']] 
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res1')
        #
        inputs = layers.conv_layer(inputs, layer_params[3], training)
        inputs = layers.conv_layer(inputs, layer_params[4], training)
        inputs = layers.padd_layer(inputs, [[0,0],[0,1],[0,1],[0,0]], name='padd2')
        #inputs = layers.conv_layer(inputs, layer_params[5], training)
        inputs = layers.maxpool_layer(inputs, (2,2), (2,2), 'valid', 'pool2')
        #
        params = [[ 256, 3, (1,1), 'same', True,  True, 'conv1'], 
                  [ 256, 3, (1,1), 'same', True, False, 'conv2']] 
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res2')
        #
        inputs = layers.conv_layer(inputs, layer_params[6], training)
        inputs = layers.conv_layer(inputs, layer_params[7], training)
        inputs = layers.padd_layer(inputs, [[0,0],[0,0],[0,1],[0,0]], name='padd3')
        inputs = layers.conv_layer(inputs, layer_params[8], training)
        #inputs = layers.maxpool_layer(inputs, (3,2), (3,2), 'valid', 'pool3')
        #
        params = [[ 512, 3, (1,1), 'same', True,  True, 'conv1'], 
                  [ 512, 3, (1,1), 'same', True, False, 'conv2']] 
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res3')
        #
        conv_feat = layers.conv_layer(inputs, layer_params[9], training)
        # 
        feat_size = tf.shape(conv_feat)
        #
    #
    # Calculate resulting sequence length from original image widths
    #
    two = tf.constant(2, dtype=tf.float32, name='two')
    #
    w = tf.cast(width, tf.float32)
    #
    w = tf.div(w, two)
    w = tf.ceil(w)
    #
    w = tf.div(w, two)
    w = tf.ceil(w)
    #
    w = tf.div(w, two)
    w = tf.ceil(w)
    #
    w = tf.cast(w, tf.int32)
    #
    w = tf.tile(w, [feat_size[1] ])
    #
    # Vectorize
    sequence_length = tf.reshape(w, [-1], name='seq_len') 
    #
    
    #
    return conv_feat, sequence_length
    #

def rnn_detect_layers(conv_feat, sequence_length, num_anchors):
    #
    # one-picture features
    conv_feat = tf.squeeze(conv_feat, axis = 0) # squeeze
    #
    #
    # Transpose to time-major order for efficiency
    #  --> [paddedSeqLen batchSize numFeatures]
    #
    rnn_sequence = tf.transpose(conv_feat, perm = [1, 0, 2], name = 'time_major')
    #
    
    #
    rnn_size = 256  # 256, 512
    fc_size = 512  # 256, 384, 512
    #    
    #
    rnn1 = layers.rnn_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
    rnn2 = layers.rnn_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
    #rnn3 = rnn_layer(rnn2, sequence_length, rnn_size, 'bdrnn3')
    #
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    rnn_feat = tf.layers.dense(rnn2, fc_size,
                               activation = tf.nn.relu,
                               kernel_initializer = weight_initializer,
                               bias_initializer = bias_initializer,
                               name = 'rnn_feat')
    #
    # out
    #
    rnn_cls = tf.layers.dense(rnn_feat, num_anchors * 2,
                              activation = tf.nn.sigmoid,
                              kernel_initializer = weight_initializer,
                              bias_initializer = bias_initializer,
                              name = 'text_cls')
    #
    rnn_ver = tf.layers.dense(rnn_feat, num_anchors * 2,
                              activation = tf.nn.tanh,
                              kernel_initializer = weight_initializer,
                              bias_initializer = bias_initializer,
                              name = 'text_ver')
    #
    rnn_hor = tf.layers.dense(rnn_feat, num_anchors * 2,
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
    
"""

In the code, anchors are fixed with width 8 pixels, and anchor_heights = [6, 12, 24, 36].

(in function get_image_and_targets(), model_data_detect.py )
target_cls = np.zeros((height_feat, width_feat, 2 * num_anchors))
target_ver = np.zeros((height_feat, width_feat, 2 * num_anchors))
target_hor = np.zeros((height_feat, width_feat, 2 * num_anchors))

For each point in the last feature map, there is a corresponding anchor center in the original picture.
For each anchor center, there are 4 anchors attached (same width, different heights).

Through some rules, each anchor box can be assigned positive, or negative.
Roughly, in width: if more than half of the anchor is text, positive;
in height: if more than half of the anchor is text, positive;
in heigth_IoU: of the 4 anchors, choose the one with max height_IoU.
Please see calculate_targets_at(anchor_center, txt_list, anchor_heights) in model_data_detect.py for details.

If an anchor box is negative, then target_cls = [0, 0], target_ver = [0, 0], and target_hor = [0, 0].
If it is positive, then target_cls = [1, 1].
And target_ver = [0, 0], target_hor = [0, 0] for initialization.
Suppose the anchor is [p_left, p_up, p_right, p_down],
and the nearest text bbox is [t_left, t_t_up, t_right, t_down],
then the targets are calcaluted as the following snippet goes:

    ratio_bbox = [0, 0, 0, 0]
    #
    ratio = (text_bbox[0]-anchor_bbox[0]) /anchor_width
    if abs(ratio) < 1: 
        ratio_bbox[0] = ratio
    #
    # print(ratio)
    #
    ratio = (text_bbox[2]-anchor_bbox[2]) /anchor_width
    if abs(ratio) < 1:
        ratio_bbox[2] = ratio
    #
    # print(ratio)
    #
    ratio_bbox[1] = (text_bbox[1]-anchor_bbox[1]) /ah
    ratio_bbox[3] = (text_bbox[3]-anchor_bbox[3]) /ah
    #
    # print(ratio_bbox)
    #
    ver.extend([ratio_bbox[1], ratio_bbox[3]])
    hor.extend([ratio_bbox[0], ratio_bbox[2]]) 
    #
So side-refinement is incorporated into target_hor. As you can see, 
target_hor is the ratio of side-displacement over anchor-width if the anchor is at one of the two ends. 
If the anchor is in the middle, then target_hor = [0, 0]. 
And target_ver is the ratio of vertical-displacement over anchor-height.

In the loss function, I first treat positive anchors and negative anchors separately, 
loss of positive anchors are averaged over positive anchors, loss of negative ones are averaged on the negative ones. 
This is because there are too many negative anchors in one picture, there is an imbalance problem. 
To easily separate the positive and negative ones, 
I specially set target_cls = [1, 1] for positive and [0, 0] for negative, using a doubled indicator.

Secondly, I modified the weights for different anchors. If the learned bbox is near the target bbox, 
that is to say the loss is small, then the weight goes down; if the loss is large, the weight goes up. 
In spirit, it is same with the focal loss, but in different implementations.

"""

