import caffe
import numpy as np
import score
from math import ceil
from numpy import zeros, arange
import matplotlib.pyplot as plt

# solver from already exit weights
use_gpu=1
weights = 'caffemodel/1_4/_iter_20000.caffemodel'

if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

solver = caffe.AdamSolver('solver.prototxt')
solver.net.copy_from(weights)

val = np.loadtxt('../image_augmentor/DRIVE/val/ImageSets/Segmentation/val.txt', dtype=str)

for _ in range(5):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')