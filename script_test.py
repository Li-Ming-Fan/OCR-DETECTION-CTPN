# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

from model_detect_class import ModelDetect
import model_detect_data
import model_detect_meta as meta


#
model = ModelDetect()
#


#
# train
# model.train_and_valid()
#

#
# validate
model.z_validate(40, False)
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
    model.predict(sess, img_file)
    #

