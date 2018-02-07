# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import model_config as meta
import model_data_detect

from model_wrap_detect import ModelDetect


#
model = ModelDetect()
#


#
# train
model.train_and_valid()
#

#
# validate
model.validate(40, False)
#

#
# predict
model.load_pb_for_prediction()
sess = model.create_session_for_prediction()
#
list_images_valid = model_data_detect.getFilesInDirect(meta.dir_images_valid, meta.str_dot_img_ext)
for img_file in list_images_valid:
    #
    # img_file = './data_test/images/bkgd_1_0_generated_0.png'
    #
    print(img_file)
    #
    model.predict(sess, img_file)
    #

