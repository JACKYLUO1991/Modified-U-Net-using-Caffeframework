import numpy as np
from PIL import Image
import caffe
import glob
import scipy.misc
from sklearn import metrics
import matplotlib.pyplot as plt

# Use GPU or not?
use_gpu = 1
gpu_id = 0

if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

fileList = glob.glob('../image_augmentor/DRIVE/val/PNGImages/*.png')
fileList2 = glob.glob('../image_augmentor/DRIVE/val/SegmentationClass/*.png')

# load net
net = caffe.Net('deploy.prototxt', 'caffemodel/1_4/_iter_20000.caffemodel', caffe.TEST)

y_true = []
y_scores = []

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
for idx in range(0, len(fileList)):
    print 'The number of the picture is: ' + str(idx)
    im = Image.open(fileList[idx])
    im = im.resize((500, 500), Image.ANTIALIAS)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((41.4661, 69.1061, 126.993))
    in_ = in_.transpose((2, 0, 1))
    
    # groundtruth
    groundtruth = Image.open(fileList2[idx])
    groundtruth = groundtruth.resize((500, 500), Image.ANTIALIAS)
    groundtruth = np.array(groundtruth, dtype=np.uint8)
    # print np.array(groundtruth).shape
    y_true.append(groundtruth)

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net
    out = net.forward()
    #
    # out = net.blobs['score'].data[0].argmax(axis=0)
    # out_1 = np.array(out).astype('uint8')
    # print np.amax(out_1)  # 1
    # out_2 = np.where(out_1 != 0, 255, 0)
    #
    # # save the picture
    # scipy.misc.imsave('DRIVE/test/result/' + fileList[idx][17:], out_2)

    # the probability of 1
    out_prob = np.array(out['prob'][:, 1]).reshape(500, 500)
    y_scores.append(out_prob)
    
y_true_flatten = np.array(y_true).flatten()
y_score_flatten = np.array(y_scores).flatten()

fpr, tpr, thresholds = metrics.roc_curve(y_true_flatten, y_score_flatten, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
    
    


