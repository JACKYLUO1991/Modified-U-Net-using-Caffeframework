import caffe
import numpy as np
import score
from math import ceil
from numpy import zeros, arange
import matplotlib.pyplot as plt

# solver from None
use_gpu=1

if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

# solver = caffe.SGDSolver('solver.prototxt')
# solver.solve()
solver = caffe.AdamSolver('solver.prototxt')
# from data to loss
# solver.net.forward()

# from loss to data
# solver.net.backward()

val = np.loadtxt('../image_augmentor/DRIVE/val/ImageSets/Segmentation/val.txt', dtype=str)

for _ in range(5):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')



