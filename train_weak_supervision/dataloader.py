from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import train_weak_supervision.config as config
import json
from train_synth.dataloader import resize, resize_generated, generate_affinity, generate_target, generate_target_others, generate_affinity_others


DEBUG = True


class DataLoaderMIX(data.Dataset):

	def __init__(self, type_, iteration):

		self.type_ = type_
		self.base_path_synth = config.DataLoaderSYNTH_base_path
		self.base_path_other_images = config.ICDAR2013_path+'/Images'
		self.base_path_other_gt = config.ICDAR2013_path+'/Generated/'+str(iteration)

		if DEBUG:
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

		self.gt = []

		for i in sorted(os.listdir(self.base_path_other_gt)):
			with open(self.base_path_other_gt+'/'+i, 'r') as f:
				self.gt.append(['.'.join(i.split('.')[:-1]) + '.jpg', json.load(f)])

	def __getitem__(self, item):

		if np.random.uniform() < config.prob_synth:

			image = plt.imread(self.base_path_synth+'/'+self.imnames[item][0])
			image, character = resize(image, self.charBB[item].copy())
			image = image.transpose(2, 0, 1)/255
			weight_character, weak_supervision_char = generate_target(image.shape, character.copy(), weight=1)
			weight_affinity, weak_supervision_affinity = generate_affinity(image.shape, character.copy(), self.txt[item].copy(), weight=1)

		else:

			image = plt.imread(self.base_path_other_images+'/'+self.gt[item % len(self.gt)][0])
			character = self.gt[item % len(self.gt)][1]['characters']
			image, character = resize_generated(image, character.copy())
			image = image.transpose(2, 0, 1) / 255
			weights = [i for i in self.gt[item % len(self.gt)][1]['weights']]

			weight_character, weak_supervision_char = generate_target_others(image.shape, character.copy(), weights)
			weight_affinity, weak_supervision_affinity = generate_affinity_others(image.shape, character.copy(), self.gt[item % len(self.gt)][1]['text'], weights)

		return image.astype(np.float32), weight_character.astype(np.float32), weight_affinity.astype(np.float32), \
				weak_supervision_char.astype(np.float32), weak_supervision_affinity.astype(np.float32)

	def __len__(self):

		return config.iterations
