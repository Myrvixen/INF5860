import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
import time

img = misc.imread('lena.tif').astype(np.float)/255
plt.imshow(img)
plt.show()
start = time.time()
print('Image size:',img.shape)

def convolution(image, kernel):
  """
  Write a general function to convolve an image with an arbitrary kernel.
  """
  out = np.zeros(image.shape)
  print(kernel)
  kernel = kernel[::-1, ::-1] #Flipping kernel to follow convention
  print(kernel)
  N, M, C = image.shape
  Nk, Mk = kernel.shape
  nk_2 = Nk // 2
  mk_2 = Mk // 2
  for i in range(nk_2, N - nk_2):
    for j in range(mk_2, M - mk_2):
      for c in range(C):
        out[i, j, c] = np.sum((image[i-nk_2:i+nk_2+1, j-nk_2:j+nk_2+1, c]*kernel))
  return out

def blur_filter(img):
  """
  Use your convolution function to filter your image with an average filter (box filter)
  with kernal size of 11.
  """
  k_size = 11
  kernel = np.ones((k_size, k_size))/k_size**2
  return convolution(img, kernel)

def sobel_filter(img):
  """
  Use your convolution function to filter your image with a vertical sobel kernel to find vertical edges
  """
  kernel = [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
  kernel = np.array(kernel)
  return convolution(img, kernel)


 # CHeck that your image looks correct
img = plt.imread('lena.tif')


# Convolution, blur
#---------------------
#out = convolution(img, np.arange(25).reshape((5, 5)))
#out = blur_filter(img)
out = sobel_filter(img)
#----------------------

out -= out.min()
out /= out.max()
correct = plt.imread('convolution_lena.tif')[:, :, :3]
plt.figure()
plt.imshow(correct, vmin=correct.min(), vmax=correct.max(), cmap='gray')
plt.figure()
plt.imshow(out, vmin=out.min(), vmax=out.max())
plt.show()
