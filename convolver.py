from PIL import Image, ImageDraw
import numpy as np
import math as mh
import cmath as cm
from collections import namedtuple
from numpy.fft import fftn, fftfreq, ifftn

EyeCoord = namedtuple('EyeCoord', ['x1', 'x2', 'r1', 'r2'])


def bilinear(odj, x):
	#[x11, x12, x21, x22]
	x_prev = int(x[0])
	y_prev = int(x[1])
	
	x_t = x[0] - x_prev
	y_t = x[1] - y_prev

	f = [odj[x_prev, y_prev], 
		 odj[x_prev, y_prev + 1], 
		 odj[x_prev + 1, y_prev], 
		 odj[x_prev + 1, y_prev + 1]]
	return f[0]*(1- x_t)*(1-y_t) + f[2]*(x_t)*(1-y_t) + f[1]*(1-x_t)*(y_t) + f[3]*(x_t)*(y_t)



def CiclePoint(x, r, phi):
	return x + r * np.array([np.cos(phi), np.sin(phi)])

def line(x1, r1, x2, r2, phi):
	return CiclePoint(x2, r2, phi) - CiclePoint(x1, r1, phi)

def point(x1, r1, l, norm, phi, rho):
	return (l / norm * rho) + CiclePoint(x1, r1, phi)


def GaborFunc(x, S, W):
	return np.exp(-(x - W)**2 / (2*S)**2)


def LogGaborFunc(x, S, W):
	return np.exp(-(np.log(x) - np.log(W))**2 / (2*np.log(S/W)**2))


def FitToRange255(matrix):
	width, height = len(matrix), len(matrix[0])
	new_matrix = matrix.copy()

	max_t = max([max(line) for line in matrix])
	min_t = min([min(line) for line in matrix])
	init_range = max_t - min_t

	for i in range(width):
		for j in range(height):
			init_val = matrix[i][j]
			fin_val = round(((init_val - min_t) / init_range) * 255)
			new_matrix[i][j] = fin_val
	return new_matrix.astype(np.uint8)


def BinFunc(convol_arr):
	tmp = convol_arr.copy()
	avg = np.median(convol_arr)
	
	for i in range(len(tmp)):
		for j in range(len(tmp[i])):
			if tmp[i][j] > avg:
				tmp[i][j] = 1
			else:
				tmp[i][j] = 0
	return tmp


def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def HammingDist(bin_arr1, bin_arr2):
	assert bin_arr1.shape == bin_arr2.shape
	return np.count_nonzero(bin_arr1 != bin_arr2)


def ImgToPolar(img, eye_coord, new_size):
	image_new = np.zeros(new_size, dtype=np.uint8)

	for i in range(new_size[0]):
		phi = 2 * np.pi / new_size[0] * (i)
		lin = line(eye_coord.x1, eye_coord.r1, eye_coord.x2, eye_coord.r2, phi)
		l = np.linalg.norm(lin)
		#print('polar', lin)
		for j in range(new_size[1]):
			rho = (j) * l / new_size[1]
			res = bilinear(img, point(eye_coord.x1, eye_coord.r1, lin, l, phi, rho))
			r = int(round(res))
			image_new[i, j] = r
	return image_new


def GetPixels(file_path):
	with Image.open(file_path) as image:
		pixels = image.load()
		return pixels


def Convolve(img, spectr_size, conv_func, **kwargs):
	tmp = img.copy()
	spectrum = fftn(tmp)#.real

	for i in range(spectr_size[0]):
		for j in range(spectr_size[1]):
			spectrum[i][j] *= conv_func((i + 1)/256, **kwargs)
		
	spectrum = ifftn(spectrum).real
	return spectrum

polar_cache = dict()

def ImgToBinSpectrum(img_path, eye_coord, spectr_size, conv_func, **kwargs):
	eye = GetPixels(img_path)

	if (img_path, spectr_size) in polar_cache:
		pixels = polar_cache[(img_path, spectr_size)]
	else:
		pixels = ImgToPolar(eye, eye_coord, spectr_size)

		polar_cache[(img_path, spectr_size)] = pixels

	#pixels = ImgToPolar(eye, eye_coord, spectr_size)
	#Image.fromarray((pixels).T).save('./tmp/' + str(eye_coord) + "polar.bmp")
	
	spectrum = Convolve(pixels , spectr_size, conv_func, **kwargs)
	tmp = spectrum.copy()
	tmp = FitToRange255(tmp)
	#Image.fromarray(tmp.T, 'L').save('./tmp/' + str(eye_coord) + "polar.bmp")	
	tmp = BinFunc(tmp)
	#Image.fromarray(FitToRange255(tmp).T, 'L').save('./tmp/' + str(eye_coord) + "polar.bmp")
	#print(spectrum)
	#Image.fromarray((spectrum).T).save('./tmp/' + str(eye_coord) + "polar.bmp")
	return tmp
'''
sigma = 17
T = 10
spectr_size = (256, 64)
eye_coord = EyeCoord(x1=np.array([337, 241]), x2=np.array([338, 245]), r1=50, r2=95)

spectrum = ImgToBinSpectrum("./2001L01.bmp", eye_coord, spectr_size, GaborFunc, S=1/sigma, W=1/T)

Image.fromarray(FitToRange255(spectrum).T, 'L').save("gabor.bmp")
'''
#====================================================================
#sigma = 17
#T = 10

#spectrum = Convolve(obj, GaborFunc, S=1/sigma, W=1/T)
#Image.fromarray(FitToRange255(spectrum).T, 'L').save("gabor.bmp")


#log_spectrum = Convolve(obj, LogGaborFunc, S=1/sigma, W=1/T)
#Image.fromarray(FitToRange255(log_spectrum).T, 'L').save("log-gabor.bmp")

#BinFunc(spectrum)

#Image.fromarray(FitToRange255(spectrum).T, 'L').save("bin-gabor.bmp")
#print(ImgHammDist(spectrum, spectrum3))

#====================================================================
