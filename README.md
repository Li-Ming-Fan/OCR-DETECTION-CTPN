# OCR-DETECTION-CTPN
  
  
CNN+LSTM (CTPN) for image text detection

</br>
  
### description
  
To run this repo:
  
1, python data_base_normalize.py    &nbsp; &nbsp; &nbsp;   # to normalize the pre-normalize background images
  
2, python data_generator.py 0    &nbsp; &nbsp; &nbsp;  # to generate validation data
  
3, python data_generator.py 1     &nbsp; &nbsp; &nbsp;  # to generate training data
  
4, python script_detect.py    &nbsp; &nbsp; &nbsp;  # to train and validate
 
</br>
  
By 1, the pre-normalized images will firstly be rescaled if not of size 800x600, then 800x600 rects will be cropped from the rescaled images. The 800x600 images will be stored in a newly-maked directory, images_base/.
  
By 2 and 3, validation data and training data will be generated. These will be store in the newly-maked directories, data_test/ and data_generated/, respectively.
  
By 4, the model will be trained and validated. The validation results will be stored in data_test/results/. The ckpt files will be stored in a newly-maked directory, model_detect/.



### detection model
  
The model is mainly based on the method described in the article:
  
Detecting Text in Natural Image with Connectionist Text Proposal Network
  
Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao
  
https://arxiv.org/abs/1609.03605




