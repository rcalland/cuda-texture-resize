import numpy as np
import ctypes
from ctypes import *

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_sum():
	dll = ctypes.CDLL('./cuda_sum.so', mode=ctypes.RTLD_GLOBAL)
	func = dll.cuda_sum
	func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_size_t]
	return func

# create __cuda_sum function with get_cuda_sum()
__cuda_sum = get_cuda_sum()

def get_cuda_interp1D():
	dll = ctypes.CDLL('./cuda_sum.so', mode=ctypes.RTLD_GLOBAL)
	func = dll.cuda_interp1D
	func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]
	return func

__cuda_interp1D = get_cuda_interp1D()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_sum(a, b, c, size):
	a_p = a.ctypes.data_as(POINTER(c_float))
	b_p = b.ctypes.data_as(POINTER(c_float))
	c_p = c.ctypes.data_as(POINTER(c_float))

	__cuda_sum(a_p, b_p, c_p, size)

def cuda_interp1D(a, b, c, size1, size2):
	a_p = a.ctypes.data_as(POINTER(c_float))
	b_p = b.ctypes.data_as(POINTER(c_float))
	c_p = c.ctypes.data_as(POINTER(c_float))

	__cuda_interp1D(a_p, b_p, c_p, size1, size2)

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':
	size=int(1024*1024)

	a = np.ones(size).astype('float32')
	b = np.ones(size).astype('float32')
	c = np.zeros(size).astype('float32')

	cuda_sum(a, b, c, size)

	print c[:10]

	# try texture stuff
	img_size = int(1024)
	grid_size = int(1024*1024)
	image = np.random.uniform(0, 1, size=img_size).astype('float32')
	grid = np.random.uniform(0, 1, size=grid_size).astype('float32')
	new_image = np.zeros(size).astype('float32')

	cuda_interp1D(image, new_image, grid, img_size, grid_size)

	print new_image[:10]