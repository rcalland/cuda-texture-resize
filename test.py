from __future__ import print_function

import numpy as np
import ctypes
from ctypes import *

from PIL import Image

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_sum():
	dll = ctypes.CDLL('./cuda_sum.so', mode=ctypes.RTLD_GLOBAL)
	func = dll.cuda_sum
	func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_size_t]
	return func

# create __cuda_sum function with get_cuda_sum()
__cuda_sum = get_cuda_sum()

def get_cuda_resize():
	dll = ctypes.CDLL('./cuda_sum.so', mode=ctypes.RTLD_GLOBAL)
	func = dll.cuda_resize
	func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t, c_size_t, c_size_t] # POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]
	return func

__cuda_resize = get_cuda_resize()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_sum(a, b, c, size):
	a_p = a.ctypes.data_as(POINTER(c_float))
	b_p = b.ctypes.data_as(POINTER(c_float))
	c_p = c.ctypes.data_as(POINTER(c_float))

	__cuda_sum(a_p, b_p, c_p, size)

def cuda_resize(a, new_size): # b, c, size1, size2):
	# pad the 4th dim
	a4 = np.zeros((a.shape[0], a.shape[1],1)).astype('float32')
	print(a.shape, a4.shape)

	a = np.concatenate((a, a4), axis=2)
	print(a.shape)

	#num_interps = 100
	out = np.zeros((new_size[0], new_size[1], 4)).astype('float32')

	a_p = a.ctypes.data_as(POINTER(c_float))
	out_p = out.ctypes.data_as(POINTER(c_float))
	#b_p = b.ctypes.data_as(POINTER(c_float))
	#c_p = c.ctypes.data_as(POINTER(c_float))

	__cuda_resize(a_p, out_p, a.shape[0], a.shape[1], new_size[0], new_size[1])
	return out

#def cuda_float4_array(a)

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':
	"""
	size=int(1024*1024)

	a = np.ones(size).astype('float32')
	b = np.ones(size).astype('float32')
	c = np.zeros(size).astype('float32')

	cuda_sum(a, b, c, size)

	print c[:10]
	"""

	# try texture stuff
	img_size = (1024, 1024, 3)
	new_size = (512, 512, 3)
	#image = np.random.random(img_size).astype('float32')
	image = np.ones(img_size).astype("float32")
	#print image.shape

	#grid = np.random.uniform(0, 1, size=grid_size).astype('float32')
	#new_image = np.zeros(grid_size).astype('float32')

	image = cuda_resize(image, new_size)
	print(image.shape)
	print(image)

	pic = Image.fromarray((image[:,:,:3] * 255.0).astype(np.uint8))
	pic.save("test.png")

	#print(interps[:10])

	"""
	print new_image[:10]
	"""