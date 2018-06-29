from __future__ import print_function

import numpy as np
import ctypes
from ctypes import *

from PIL import Image

#import chainer
import cupy

def get_cuda_resize():
	dll = ctypes.CDLL('./cuda_resize.so', mode=ctypes.RTLD_GLOBAL)
	func = dll.cuda_resize
	func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t, c_size_t, c_size_t] 
	return func

__cuda_resize = get_cuda_resize()

def cuda_resize(a, new_size):

	start = cupy.cuda.Event()
	end = cupy.cuda.Event()
	start.record()

	# pad the 4th dim
	a4 = np.zeros((a.shape[0], a.shape[1],1)).astype('float32')

	a = np.concatenate((a, a4), axis=2)

	out = np.zeros((new_size[0], new_size[1], 4)).astype('float32')

	a_p = a.ctypes.data_as(POINTER(c_float))
	out_p = out.ctypes.data_as(POINTER(c_float))


	__cuda_resize(a_p, out_p, a.shape[0], a.shape[1], new_size[0], new_size[1])

	out = out[:,:,:3]

	end.record()
	end.synchronize()
	time = cupy.cuda.get_elapsed_time(start, end)  # milliseconds
	print(time)

	return out

if __name__ == '__main__':

	input_image = np.array(Image.open("len_std.jpg")).astype("float32") / 255.0
	
	img_size = input_image.shape #(1024, 1024, 3)
	new_size = (512, 512, 3)

	image = cuda_resize(input_image, new_size)

	image = cuda_resize(input_image, new_size)

	pic = Image.fromarray((image * 255.0).astype(np.uint8))
	pic.save("test.png")
