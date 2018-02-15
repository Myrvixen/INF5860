import numpy as np
#from skimage.io import imread, imsave
from matplotlib import pyplot as plt
from scipy.misc import imread, imsave, imresize


def test_convolution():
  from konvolusjon import convolution
  img = plt.imread('lena.png')
  out = convolution(img, np.arange(25).reshape((5, 5)))
  out -= out.min()
  out /= out.max()
  correct = plt.imread('convolution_lena.png')[:, :, :3]
  assert np.abs(out[10:-10, 10:-10] - correct[10:-10, 10:-10]).max() < 1e-2

def test_sobel_filter():
  from konvolusjon import sobel_filter
  img = plt.imread('lena.png')
  out = sobel_filter(img)
  out -= out.min()
  out /= out.max()
  correct = plt.imread('sobel_lena.png')[:, :, :3]
  assert np.abs(out[10:-10, 10:-10] - correct[10:-10, 10:-10]).max() < 1e-2

def test_blur_filter():
  from konvolusjon import blur_filter
  img = plt.imread('lena.png').astype(np.float) / 255
  out = blur_filter(img)
  out -= out.min()
  out /= out.max()
  correct = plt.imread('blur_lena.png')[:, :, :3]
  assert np.abs(out[10:-10, 10:-10] - correct[10:-10, 10:-10]).max() < 1e-2
