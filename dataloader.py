from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2

DEBUG = False


def four_point_transform(image, pts):

	max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

	dst = np.array([
		[0, 0],
		[image.shape[1] - 1, 0],
		[image.shape[1] - 1, image.shape[0] - 1],
		[0, image.shape[0] - 1]], dtype="float32")

	M = cv2.getPerspectiveTransform(dst, pts)
	warped = cv2.warpPerspective(image, M, (max_x, max_y))

	return warped


class DataLoaderSYNTH(data.Dataset):

	def __init__(self, type_):

		self.type_ = type_
		self.base_path = '/home/SharedData/Mayank/SynthText/Images'
		if DEBUG:
			import os
			if not os.path.exists('cache.pkl'):
				with open('cache.pkl', 'wb') as f:
					import pickle
					from scipy.io import loadmat
					mat = loadmat('/home/SharedData/Mayank/SynthText/gt.mat')
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
			mat = loadmat('/home/SharedData/Mayank/SynthText/gt.mat')

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

		sigma = 10
		spread = 3
		extent = int(spread * sigma)
		mult = 1.3
		center = spread * sigma * mult / 2

		self.gaussian_heatmap = np.zeros([int(extent*mult), int(extent*mult)], dtype=np.float32)

		for i in range(int(extent*mult)):
			for j in range(int(extent*mult)):
				self.gaussian_heatmap[i, j] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
					-1 / 2 * ((i - center - 0.5) ** 2 + (j - center - 0.5) ** 2) / (sigma ** 2))

		self.gaussian_heatmap = (self.gaussian_heatmap / np.max(self.gaussian_heatmap) * 255).astype(np.uint8)

	def add_character(self, image, bbox):

		backup = image.copy()

		try:

			top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
			if top_left[1] > image.shape[0] or top_left[0] > image.shape[1]:
				return image
			bbox -= top_left[None, :]
			transformed = four_point_transform(self.gaussian_heatmap.copy(), bbox.astype(np.float32))

			start_row = max(top_left[1], 0) - top_left[1]
			start_col = max(top_left[0], 0) - top_left[0]
			end_row = min(top_left[1]+transformed.shape[0], image.shape[0])
			end_col = min(top_left[0]+transformed.shape[1], image.shape[1])
			image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1], start_col:end_col - top_left[0]]

			return image

		except:

			return backup

	def generate_target(self, image_size, character_bbox):

		character_bbox = character_bbox.transpose(2, 1, 0)

		channel, height, width = image_size

		target = np.zeros([height, width], dtype=np.uint8)

		for i in range(character_bbox.shape[0]):

			target = self.add_character(target, character_bbox[i].copy())

		return target/255

	def add_affinity(self, image, bbox_1, bbox_2):

		backup = image.copy()

		try:

			center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
			tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
			bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
			tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
			br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

			affinity = np.array([tl, tr, br, bl])

			return self.add_character(image, affinity)

		except:

			return backup

	def generate_affinity(self, image_size, character_bbox, text):

		"""

		:param image_size: shape = [3, image_height, image_width]
		:param character_bbox: [2, 4, num_characters]
		:param text: [num_words]
		:return:
		"""

		character_bbox = character_bbox.transpose(2, 1, 0)

		channel, height, width = image_size

		target = np.zeros([height, width], dtype=np.uint8)

		total_letters = 0

		for word in text:
			for char_num in range(len(word)-1):
				target = self.add_affinity(target, character_bbox[total_letters].copy(), character_bbox[total_letters+1].copy())
				total_letters += 1
			total_letters += 1

		return target / 255

	def resize(self, image, character, side=768):

		height, width, channel = image.shape
		max_side = max(height, width)
		new_reisze = (int(width/max_side*side), int(height/max_side*side))
		image = cv2.resize(image, new_reisze)

		character[0, :, :] = character[0, :, :]/width*new_reisze[0]
		character[1, :, :] = character[1, :, :]/height*new_reisze[1]

		big_image = np.ones([side, side, 3], dtype=np.float32)*np.mean(image)
		big_image[(side-image.shape[0])//2: (side-image.shape[0])//2 + image.shape[0], (side-image.shape[1])//2: (side-image.shape[1])//2 + image.shape[1]] = image
		big_image = big_image.astype(np.uint8)

		character[0, :, :] += (side-image.shape[1])//2
		character[1, :, :] += (side-image.shape[0])//2

		return big_image, character

	def __getitem__(self, item):

		image = plt.imread(self.base_path+'/'+self.imnames[item][0])
		image, character = self.resize(image, self.charBB[item].copy())
		image = image.transpose(2, 0, 1)/255
		weight = self.generate_target(image.shape, character.copy())
		weight_affinity = self.generate_affinity(image.shape, character.copy(), self.txt[item].copy())

		return image.astype(np.float32), weight.astype(np.float32), weight_affinity.astype(np.float32)

	def __len__(self):

		return len(self.imnames)
