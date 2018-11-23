# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import model_detect_meta as meta
import model_detect_data as model_data

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
# predict
model.prepare_for_prediction()
#
list_images_valid = model_data.get_files_with_ext(meta.dir_images_valid, 'png')
for img_file in list_images_valid:
    #
    # img_file = './data_test/images/bkgd_1_0_generated_0.png'
    #
    print(img_file)
    #
    conn_bbox, text_bbox, conf_bbox = model.predict(img_file, out_dir = './results_prediction')
    #

