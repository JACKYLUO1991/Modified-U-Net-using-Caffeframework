from PIL import Image
import numpy as np
import scipy.misc

im = Image.open('densecrf.png')
im = np.array(im).astype(np.uint8)
out = np.where(im!=0, 255, 0)
scipy.misc.imsave('crf.png', out)

