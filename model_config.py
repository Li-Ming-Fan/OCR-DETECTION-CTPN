# -*- coding: utf-8 -*-
"""
@author: limingfan

"""


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

#
model_detect_dir = './model_detect'
model_detect_name = 'model_detect'
model_detect_pb_file = model_detect_name + '.pb'
#
anchor_heights = [16, 24, 32, 48, 64]
#
threshold = 0.5  #
#

#
model_recog_dir = './model_recog'
model_recog_name = 'model_recog'
model_recog_pb_file = model_recog_name + '.pb'
#
height_norm = 64  # 
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



