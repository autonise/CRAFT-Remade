from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import train_synth.config as config
from src.utils.data_manipulation import resize, normalize_mean_variance, generate_affinity, generate_target

"""
	globally generating gaussian heatmap which will be warped for every character bbox
"""


class DataLoaderSYNTH(data.Dataset):

	"""
		DataLoader for strong supervised training on Synth-Text
	"""

	DEBUG = False  # Make this True if you want to do a run on small set of Synth-Text

	def __init__(self, type_):

		self.type_ = type_
		self.base_path = config.DataLoaderSYNTH_base_path

		if DataLoaderSYNTH.DEBUG:

			# To check for small data sample of Synth

			if not os.path.exists('cache.pkl'):

				# Create cache of 1000 samples if it does not exist

				with open('cache.pkl', 'wb') as f:
					import pickle
					from scipy.io import loadmat
					mat = loadmat(config.DataLoaderSYNTH_mat)
					pickle.dump([mat['imnames'][0][0:1000], mat['charBB'][0][0:1000], mat['txt'][0][0:1000]], f)
					print('Created the pickle file, rerun the program')
					exit(0)
			else:

				# Read the Cache

				with open('cache.pkl', 'rb') as f:
					import pickle
					self.imnames, self.charBB, self.txt = pickle.load(f)
					print('Loaded DEBUG')

		else:

			from scipy.io import loadmat
			mat = loadmat(config.DataLoaderSYNTH_mat)  # Loads MATLAB .mat extension as a dictionary of numpy arrays

			# Read documentation of how synth-text dataset is stored to understand the processing at
			# http://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt

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
				all_words += [k for k in ' '.join(j.split('\n')).split() if k != '']
				# Getting all words given paragraph like text in SynthText

			self.txt[no] = all_words

	def __getitem__(self, item):

		item = item % len(self.imnames)
		image = plt.imread(self.base_path+'/'+self.imnames[item][0])  # Read the image

		if len(image.shape) == 2:
			image = np.repeat(image[:, :, None], repeats=3, axis=2)
		elif image.shape[2] == 1:
			image = np.repeat(image, repeats=3, axis=2)
		else:
			image = image[:, :, 0: 3]

		image, character = resize(image, self.charBB[item].copy())  # Resize the image to (768, 768)
		normal_image = image.astype(np.uint8).copy()
		image = normalize_mean_variance(image).transpose(2, 0, 1)

		# Generate character heatmap
		weight_character = generate_target(image.shape, character.copy())

		# Generate affinity heatmap
		weight_affinity, affinity_bbox = generate_affinity(image.shape, character.copy(), self.txt[item].copy())

		cv2.drawContours(
			normal_image,
			np.array(affinity_bbox).reshape([len(affinity_bbox), 4, 1, 2]).astype(np.int64), -1, (0, 255, 0), 2)

		enlarged_affinity_bbox = []

		for i in affinity_bbox:
			center = np.mean(i, axis=0)
			i = i - center[None, :]
			i = i*60/25
			i = i + center[None, :]
			enlarged_affinity_bbox.append(i)

		cv2.drawContours(
			normal_image,
			np.array(enlarged_affinity_bbox).reshape([len(affinity_bbox), 4, 1, 2]).astype(np.int64),
			-1, (0, 0, 255), 2
		)

		return \
			image.astype(np.float32), \
			weight_character.astype(np.float32), \
			weight_affinity.astype(np.float32), \
			normal_image

	def __len__(self):

		if self.type_ == 'train':
			return int(len(self.imnames)*config.num_epochs_strong_supervision)
		else:
			return len(self.imnames)


class DataLoaderEval(data.Dataset):

	"""
		DataLoader for evaluation on any custom folder given the path
	"""

	def __init__(self, path):

		self.base_path = path
		self.imnames = sorted(os.listdir(self.base_path))

	def __getitem__(self, item):

		image = plt.imread(self.base_path+'/'+self.imnames[item])  # Read the image

		if len(image.shape) == 2:
			image = np.repeat(image[:, :, None], repeats=3, axis=2)
		elif image.shape[2] == 1:
			image = np.repeat(image, repeats=3, axis=2)
		else:
			image = image[:, :, 0: 3]

		# ------ Resize the image to (768, 768) ---------- #

		height, width, channel = image.shape
		max_side = max(height, width)
		new_resize = (int(width / max_side * 768), int(height / max_side * 768))
		image = cv2.resize(image, new_resize)

		big_image = np.ones([768, 768, 3], dtype=np.float32) * np.mean(image)
		big_image[
			(768 - image.shape[0]) // 2: (768 - image.shape[0]) // 2 + image.shape[0],
			(768 - image.shape[1]) // 2: (768 - image.shape[1]) // 2 + image.shape[1]] = image
		big_image = normalize_mean_variance(big_image)
		big_image = big_image.transpose(2, 0, 1)

		return big_image.astype(np.float32), self.imnames[item], np.array([height, width])

	def __len__(self):

		return len(self.imnames)
