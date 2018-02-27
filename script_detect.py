# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import model_comm_meta as meta

import model_detect_data
from model_detect_wrap import ModelDetect


import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
#


#
model = ModelDetect()
#


#
# train
model.train_and_valid()
#

#
# validate
model.validate(40000, False)
#

#
# predict
model.load_pb_for_prediction()
sess = model.create_session_for_prediction()
#
list_images_valid = model_detect_data.getFilesInDirect(meta.dir_images_valid, meta.str_dot_img_ext)
for img_file in list_images_valid:
    #
    # img_file = './data_test/images/bkgd_1_0_generated_0.png'
    #
    print(img_file)
    #
    model.predict(sess, img_file, out_dir = './results_prediction')
    #

