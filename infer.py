import numpy as np
from PIL import Image
import scipy.misc
import caffe
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import time

fn_output = 'densecrf.png'

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, axis=0)
    # print output_probs.shape  # (1, 500, 500)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)
    # print output_probs.shape

    d = dcrf.DenseCRF2D(w, h, 2)
    U = unary_from_softmax(output_probs)

    # Return a contiguous array in memory
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=1, compat=1, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=1, srgb=1, rgbim=im, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)


    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

caffe.set_mode_cpu()

# load net
net = caffe.Net('deploy.prototxt', 'caffemodel/1_4/_iter_20000.caffemodel', caffe.TEST)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('../image_augmentor/DRIVE/val/PNGImages/02_test.png')
im = im.resize((500, 500), Image.ANTIALIAS)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((41.4661, 69.1061, 126.993))
in_ = in_.transpose((2, 0, 1))

# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

out = net.forward()

out_prob = np.array(out['prob'][:, 1]).reshape(500, 500)

im = net.blobs['data'].data[0]
im = np.transpose(im, (1, 2, 0))
im = im[:,:,(2, 1, 0)]
im = np.array(im, dtype=np.uint8)
im = im.copy(order='C')

Q = dense_crf(im, out_prob)
imsave(fn_output, Q.reshape(im.shape[1], im.shape[0]))
