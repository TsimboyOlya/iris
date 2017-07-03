#!usr/bin/python
import sys
from os import path
import numpy as np

from convolver import EyeCoord, GaborFunc, ImgToBinSpectrum, HammingDist, FitToRange255, LogGaborFunc
from PIL import Image
from multiprocessing import Pool

import cProfile

def get_data_dict(data_path, imgs_path):
	res_dict = dict()
	with open(data_path, 'r') as data_file:
		i = 0
		k = 0
		for line in data_file:
			line_items = line.split()
			img_name = line_items[0]
			k += 1
			if path.exists(imgs_path + '/' + img_name):
				i += 1
				res_dict[img_name] = EyeCoord(	x1=np.array([int(i) for i in line_items[6:8]]),
												r1=int(line_items[8]),
												x2=np.array([int(i) for i in line_items[11:13]]),
												r2=int(line_items[13]))
	print(i, k)
	return res_dict


def is_test_img(name):
#	return 1
	return int(name[:4]) in range(2001, 2151)
	#return 'R' in name and int(name[:4]) in range(2001, 2151)
	#return 'L' in name and int(name[:4]) in range(2001, 2151)


def ImageSpectrum(arg_tuple):
	eye_name, eye_coord, imgs_path = arg_tuple
	spectr_size = (256, 64)
	sigma = 17
	T = 8
	img_path = imgs_path + '/' + eye_name
	return ImgToBinSpectrum(img_path, eye_coord, spectr_size, LogGaborFunc, S=1/sigma, W=1/T)
	#return ImgToBinSpectrum(img_path, eye_coord, spectr_size, GaborFunc, S=1/sigma, W=1/T)

def HammDistMatrix(imgs_path, imgs_dict):
	dist_same = []
	dist_diff = []
	
	img_to_idx = dict()
	idx_to_img = dict()
	pool = Pool(4)

	for idx, (eye_name, _) in enumerate(imgs_dict.items()):
		img_to_idx[eye_name] = idx
		idx_to_img[idx] = eye_name

	arg_tuples = list(zip(imgs_dict.keys(), imgs_dict.values(), [imgs_path] * len(imgs_dict)))

	spectrums = pool.map(ImageSpectrum, arg_tuples)

	assert len(imgs_dict) == len(spectrums)
	img_to_spectrum = dict(zip(imgs_dict.keys(), spectrums))

	for x in range(len(imgs_dict)):
		spectr_x = img_to_spectrum[idx_to_img[x]]
		for y in range(x + 1, len(imgs_dict)):
			spectr_y = img_to_spectrum[idx_to_img[y]]

			dist = HammingDist(spectr_x, spectr_y)
			print(dist)
			#Image.fromarray(FitToRange255(spectr_x).T, 'L').save('./tmp/spectr_' + idx_to_img[x])
			if idx_to_img[y][:5] == idx_to_img[x][:5]:
				dist_same.append(dist)
				#print('same', idx_to_img[y], idx_to_img[x], '->', dist)
			else:
				dist_diff.append(dist)
				#print('diff', idx_to_img[y], idx_to_img[x], '->', dist)

	return dist_same, dist_diff


if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Not enough args')
		sys.exit(1)


#	try:

	data_path = sys.argv[1]
	imgs_path = sys.argv[2]

	data_dict = get_data_dict(data_path, imgs_path)
	print(len(data_dict))
	test_set = dict((key,value) for key, value in data_dict.items() if is_test_img(key))
	print(len(test_set))

	#cProfile.run('HammDistMatrix(imgs_path, dict(list(test_set.items())[:50]))')
	same, diff = HammDistMatrix(imgs_path, test_set)
	f = open('same.txt', 'w')
	f.write(' '.join([str(int(j)) for j in same]))
	f.close()
	f = open('diff.txt', 'w')
	f.write(' '.join([str(i) for i in list(diff)]))
	f.close()

#	except Exception as e:
#		print(e)
#		sys.exit(1)