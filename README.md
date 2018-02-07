# OCR-CTPN-CRNN
  
  
CNN+LSTM (CTPN/CRNN) for image text detection and recognition
  
  
### description
  
The detection model (CTPN) can easily be trained from scratch, while the recognition model (CRNN+CTC) can hardly be trainded from scratch.
  
So the two are designed to share large part of CNN layers and the RNN layers. So the recognition model can be trained with the detection model as the pretrained model.
  

### detection model
  
The model is mainly based on the method described in the article:
  
Detecting Text in Natural Image with Connectionist Text Proposal Network
  
Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao
  
https://arxiv.org/abs/1609.03605


### recognition model

The model is mainly based on the method described in the article:
  
An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition
  
Baoguang Shi, Xiang Bai, Cong Yao
  
https://arxiv.org/abs/1507.05717

###



