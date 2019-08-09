from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import train_synth.config as config
import json
from train_synth.dataloader import resize, generate_affinity, generate_target


DEBUG = False


class DataLoaderMIX(data.Dataset):

	def __init__(self, type_, path_gen):

		self.type_ = type_
		self.base_path_synth = config.DataLoaderSYNTH_base_path
		self.base_path_other_images = config.ICDAR2013_path+'/Images'
		self.base_path_other_gt = path_gen

		if DEBUG:
			import os
			if not os.path.exists('cache.pkl'):
				with open('cache.pkl', 'wb') as f:
					import pickle
					from scipy.io import loadmat
					mat = loadmat(config.DataLoaderSYNTH_mat)
					pickle.dump([mat['imnames'][0][0:1000], mat['charBB'][0][0:1000], mat['txt'][0][0:1000]], f)
					print('Created the pickle file, rerun the program')
					exit(0)
			else:
				with open('cache.pkl', 'rb') as f:
					import pickle
					self.imnames, self.charBB, self.txt = pickle.load(f)
					print('Loaded DEBUG')

		else:

			from scipy.io import loadmat
			mat = loadmat(config.DataLoaderSYNTH_mat)

			total_number = mat['imnames'][0].shape[0]
			train_images = int(total_number * 0.9)

			if self.type_ == 'train':

				self.imnames = mat['imnames'][0][0:train_images]
				self.charBB = mat['charBB'][0][0:train_images]  # number of images, 2, 4, num_character
				self.txt = mat['txt'][0][0:train_images]

			else:

				self.imnames = mat['imnames'][0][train_images:]
				self.charBB = mat['charBB'][0][train_images:]  # number of images, 2, 4, num_character
				self.txt = mat['txt'][0][train_images:]

		for no, i in enumerate(self.txt):
			all_words = []
			for j in i:
				all_words += [k for k in ' '.join(j.split('\n')).split() if k!='']
			self.txt[no] = all_words

		with open(self.base_path_other_gt, 'r') as f:
			self.gt = json.load(f)

		# ToDo - get the character_bbox, affinity_bbox, weights corresponding to each image from self.gt

	def __getitem__(self, item):

		# ToDo - random choice between Synth and other dataset

		image = plt.imread(self.base_path_synth+'/'+self.imnames[item][0])
		image, character = resize(image, self.charBB[item].copy())
		image = image.transpose(2, 0, 1)/255
		weight = generate_target(image.shape, character.copy())
		weight_affinity = generate_affinity(image.shape, character.copy(), self.txt[item].copy())

		# ToDo - write function to generate ground truth for other dataset

		# ToDo - pass weights for each word bbox also

		return image.astype(np.float32), weight.astype(np.float32), weight_affinity.astype(np.float32)

	def __len__(self):

		return len(self.imnames)
