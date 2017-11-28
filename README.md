# Caffe modified U-Net
  Automatic segmentation of retinal blood vessels from fundus images plays a key role in the computer aided diagnostic system, which is helpful for the early treatment of many fundus diseases including diabetic retinopathy, glaucoma and hypertension. In this section, a modified U-Net is proposed to achieve semantic segmentation of retinal blood vessels. In addition, we use Condition Random Field to integrate the global information. The comparison between our method and other typical methods is given to evaluate the proposed method. Our network architecture achieves a satisfactory result on publicly available DRIVE database and we have obtained an average accuracy of 86.5% for retinal blood vessels segmentation task.
#### segmentation results
<div align=center>
  <img src="https://github.com/actionLUO/Modified-U-Net-using-Caffeframework/blob/master/picture/01_test.png" width="150" height="150" alt="vessel1"/>
<img src="https://github.com/actionLUO/Modified-U-Net-using-Caffeframework/blob/master/picture/02_test.png" width="150" height="150" alt="vessel2"/>
<img src="https://github.com/actionLUO/Modified-U-Net-using-Caffeframework/blob/master/picture/03_test.png" width="150" height="150" alt="vessel3"/>
<img src="https://github.com/actionLUO/Modified-U-Net-using-Caffeframework/blob/master/picture/04_test.png" width="150" height="150" alt="vessel4"/>
<img src="https://github.com/actionLUO/Modified-U-Net-using-Caffeframework/blob/master/picture/05_test.png" width="150" height="150" alt="vessel5"/</div>
#### roc curve
<div align=center>
    <img src="https://github.com/actionLUO/Modified-U-Net-using-Caffeframework/blob/master/ROC.png" alt="roc_curve"/></div>
