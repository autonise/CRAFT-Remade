from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import train_synth.config as config

"""
	globally generating gaussian heatmap which will be warped for every character bbox
"""


sigma = 10
spread = 3
extent = int(spread * sigma)
mult = 1.3
center = spread * sigma * mult / 2
gaussian_heatmap = np.zeros([int(extent*mult), int(extent*mult)], dtype=np.float32)

for i_ in range(int(extent*mult)):
	for j_ in range(int(extent*mult)):
		gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
			-1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))

gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
	# should be RGB order
	img = in_img.copy().astype(np.float32)

	img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
	img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
	return img


def denormalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
	# should be RGB order
	img = in_img.copy()
	img *= variance
	img += mean
	img *= 255.0
	img = np.clip(img, 0, 255).astype(np.uint8)
	return img


def order_points(pts):

	"""
	Orders the 4 co-ordinates of a bounding box, top-left, top-right, bottom-right, bottom-left
	:param pts: numpy array with shape [4, 2]
	:return: numpy array, shape = [4, 2], ordered bbox
	"""

	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def four_point_transform(image, pts):

	"""
	Using the pts and the image a perspective transform is performed which returns the transformed 2d Gaussian image
	:param image: np.array, dtype=np.uint8, shape = [height, width]
	:param pts: np.array, dtype=np.float32 or np.int32, shape = [4, 2]
	:return:
	"""

	max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

	dst = np.array([
		[0, 0],
		[image.shape[1] - 1, 0],
		[image.shape[1] - 1, image.shape[0] - 1],
		[0, image.shape[0] - 1]], dtype="float32")

	warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(dst, pts), (max_x, max_y))

	return warped


def resize(image, character, side=768):

	"""
		Resizing the image while maintaining the aspect ratio and padding with average of the entire image to make the
		reshaped size = (side, side)
		:param image: np.array, dtype=np.uint8, shape=[height, width, 3]
		:param character: np.array, dtype=np.int32 or np.float32, shape = [2, 4, num_characters]
		:param side: new size to be reshaped to
		:return: resized_image, corresponding reshaped character bbox
	"""

	height, width, channel = image.shape
	max_side = max(height, width)
	new_resize = (int(width/max_side*side), int(height/max_side*side))
	image = cv2.resize(image, new_resize)

	character[0, :, :] = character[0, :, :]/width*new_resize[0]
	character[1, :, :] = character[1, :, :]/height*new_resize[1]

	big_image = np.ones([side, side, 3], dtype=np.float32)*np.mean(image)
	big_image[
		(side-image.shape[0])//2: (side-image.shape[0])//2 + image.shape[0],
		(side-image.shape[1])//2: (side-image.shape[1])//2 + image.shape[1]] = image
	big_image = big_image.astype(np.uint8)

	character[0, :, :] += (side-image.shape[1])//2
	character[1, :, :] += (side-image.shape[0])//2

	return big_image, character


def resize_generated(image, character, affinity, side=768):
	"""
		Resizing the image while maintaining the aspect ratio and padding with average of the entire image to make the
		reshaped size = (side, side)
		:param image: np.array, dtype=np.uint8, shape=[height, width, 3]
		:param character: list of np.array, dtype=np.int64, shape = [num_words, num_characters, 4, 1, 2]
		:param affinity: list of np.array, dtype=np.int64, shape = [num_words, num_affinity, 4, 1, 2]
		:param side: new size to be reshaped to
		:return: resized_image, corresponding reshaped character bbox
	"""

	height, width, channel = image.shape
	max_side = max(height, width)
	new_resize = (int(width/max_side*side), int(height/max_side*side))
	image = cv2.resize(image, new_resize)

	for word_no in range(len(character)):

		character[word_no][:, :, :, 0] = character[word_no][:, :, :, 0] / width * new_resize[0]
		character[word_no][:, :, :, 1] = character[word_no][:, :, :, 1] / height * new_resize[1]
		affinity[word_no][:, :, :, 0] = affinity[word_no][:, :, :, 0] / width * new_resize[0]
		affinity[word_no][:, :, :, 1] = affinity[word_no][:, :, :, 1] / height * new_resize[1]

	big_image = np.ones([side, side, 3], dtype=np.float32)*np.mean(image)
	big_image[
		(side-image.shape[0])//2: (side-image.shape[0])//2 + image.shape[0],
		(side-image.shape[1])//2: (side-image.shape[1])//2 + image.shape[1]] = image
	big_image = big_image.astype(np.uint8)

	for word_no in range(len(character)):

		character[word_no][:, :, :, 0] += (side - image.shape[1]) // 2
		character[word_no][:, :, :, 1] += (side - image.shape[0]) // 2
		affinity[word_no][:, :, :, 0] += (side - image.shape[1]) // 2
		affinity[word_no][:, :, :, 1] += (side - image.shape[0]) // 2

	return big_image, character, affinity


def add_character(image, bbox):

	"""
		Add gaussian heatmap for character bbox to the image
		:param image: 2-d array containing character heatmap
		:param bbox: np.array, dtype=np.int32, shape = [4, 2]
		:return: image in which the gaussian character bbox has been added
	"""

	top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
	if top_left[1] > image.shape[0] or top_left[0] > image.shape[1]:
		return image
	bbox -= top_left[None, :]
	transformed = four_point_transform(gaussian_heatmap.copy(), bbox.astype(np.float32))

	start_row = max(top_left[1], 0) - top_left[1]
	start_col = max(top_left[0], 0) - top_left[0]
	end_row = min(top_left[1] + transformed.shape[0], image.shape[0])
	end_col = min(top_left[0] + transformed.shape[1], image.shape[1])
	image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += \
		transformed[
		start_row:end_row - top_left[1],
		start_col:end_col - top_left[0]]

	return image


def add_character_others(image, weight_map, weight_val, bbox):
	"""
		Add gaussian heatmap for character bbox to the image and also generate weighted map for weak-supervision
		:param image: 2-d array containing character heatmap
		:param weight_map: 2-d array containing weight heatmap
		:param weight_val: weight to be given to the current bbox
		:param bbox: np.array, dtype=np.int32, shape = [4, 2]
		:return:    image in which the gaussian character bbox has been added,
					weight_map in which the weight as per weak-supervision has been calculated
	"""

	top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
	if top_left[1] > image.shape[0] or top_left[0] > image.shape[1]:
		return image, weight_map
	bbox -= top_left[None, :]
	transformed = four_point_transform(gaussian_heatmap.copy(), bbox.astype(np.float32))

	start_row = max(top_left[1], 0) - top_left[1]
	start_col = max(top_left[0], 0) - top_left[0]
	end_row = min(top_left[1] + transformed.shape[0], image.shape[0])
	end_col = min(top_left[0] + transformed.shape[1], image.shape[1])
	image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += \
		transformed[
		start_row:end_row - top_left[1],
		start_col:end_col - top_left[0]]

	weight_map[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += \
		np.float32(transformed[
			start_row:end_row - top_left[1],
			start_col:end_col - top_left[0]] != 0)*weight_val

	return image, weight_map


def add_affinity(image, bbox_1, bbox_2):

	"""
		Add gaussian heatmap for affinity bbox to the image between bbox_1, bbox_2
		:param image: 2-d array containing affinity heatmap
		:param bbox_1: np.array, dtype=np.int32, shape = [4, 2]
		:param bbox_2: np.array, dtype=np.int32, shape = [4, 2]
		:return: image in which the gaussian affinity bbox has been added
	"""

	bbox_1 = order_points(bbox_1)
	bbox_2 = order_points(bbox_2)

	center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
	tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
	bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
	tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
	br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

	affinity = np.array([tl, tr, br, bl])

	return add_character(image, affinity)


def two_char_bbox_to_affinity(bbox_1, bbox_2):

	"""
	Given two character bbox generates the co-ordinates of the affinity bbox between them
	:param bbox_1: type=np.array, dtype=np.int64, shape = [4, 1, 2]
	:param bbox_2: type=np.array, dtype=np.int64, shape = [4, 1, 2]
	:return: affinity bbox, type=np.array, dtype=np.int64, shape = [4, 1, 2]
	"""

	bbox_1 = bbox_1[:, 0, :].copy()
	bbox_2 = bbox_2[:, 0, :].copy()

	center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
	tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
	bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
	tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
	br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

	affinity = np.array([tl, tr, br, bl]).reshape([4, 1, 2])

	return affinity


def add_affinity_others(image, weight, weight_val, bbox_1, bbox_2):

	"""
		Add gaussian heatmap for affinity bbox to the image and also generate weighted map for weak-supervision
		:param image: 2-d array containing affinity heatmap
		:param weight: 2-d array containing weight heatmap
		:param weight_val: weight to be given to the current affinity bbox
		:param bbox_1: np.array, dtype=np.int32, shape = [4, 2]
		:param bbox_2: np.array, dtype=np.int32, shape = [4, 2]
		:return:    image in which the gaussian affinity bbox has been added between bbox_1 and bbox_2,
					weight_map in which the weight as per weak-supervision has been calculated
	"""

	bbox_1 = order_points(bbox_1)
	bbox_2 = order_points(bbox_2)
	backup = image.copy(), weight.copy()

	try:

		affinity = two_char_bbox_to_affinity(bbox_1, bbox_2)

		return add_character_others(image, weight, weight_val, affinity)

	except:

		return backup


def generate_target(image_size, character_bbox, weight=None):

	"""

	:param image_size: size of the image on which the target needs to be generated
	:param character_bbox: np.array, shape = [2, 4, num_characters]
	:param weight: this function is currently only used for synth-text in which we have 100 % confidence so weight = 1
					where the character bbox are present
	:return: if weight is not None then target_character_heatmap otherwise target_character_heatmap,
																			weight for weak-supervision
	"""

	character_bbox = character_bbox.transpose(2, 1, 0)

	channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)

	for i in range(character_bbox.shape[0]):

		target = add_character(target, character_bbox[i].copy())

	if weight is not None:
		return target/255, np.float32(target != 0)
	else:
		return target/255


def generate_target_others(image_size, character_bbox, weight):
	"""

		:param image_size: size of the image on which the target needs to be generated
		:param character_bbox: np.array, shape = [word_length, num_characters, 4, 1, 2]
		:param weight: this function is currently only used for icdar2013, so weight is the value of weight
																							for each character bbox
		:return: if weight is not None then target_character_heatmap otherwise target_character_heatmap,
																				weight for weak-supervision
		"""

	if len(image_size) == 2:
		height, width = image_size
	else:
		channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)
	weight_map = np.zeros([height, width], dtype=np.float32)

	for word_no in range(len(character_bbox)):

		for i in range(character_bbox[word_no].shape[0]):

			target, weight_map = add_character_others(
				target, weight_map, weight[word_no], character_bbox[word_no][i].copy()[:, 0, :])

	return target/255, weight_map


def generate_affinity(image_size, character_bbox, text, weight=None):

	"""

	:param image_size: shape = [3, image_height, image_width]
	:param character_bbox: [2, 4, num_characters]
	:param text: [num_words]
	:param weight: This is currently used only for synth-text so specifying weight as not None will generate a heatmap
					having value one where there is affinity
	:return: if weight is not None then target_affinity_heatmap otherwise target_affinity_heatmap,
																				weight for weak-supervision

	"""

	character_bbox = character_bbox.transpose(2, 1, 0)

	if len(image_size) == 2:
		height, width = image_size
	else:
		channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)

	total_letters = 0

	for word in text:
		for char_num in range(len(word)-1):
			target = add_affinity(target, character_bbox[total_letters].copy(), character_bbox[total_letters+1].copy())
			total_letters += 1
		total_letters += 1

	if weight is not None:

		return target / 255, np.float32(target != 0)

	else:

		return target / 255


def generate_affinity_others(image_size, character_bbox, weight):

	"""

	:param image_size: shape = [3, image_height, image_width]
	:param character_bbox: [2, 4, num_characters]
	:param weight: This is currently used only for icdar 2013. it is a list containing weight for each bbox
	:return: target_affinity_heatmap, weight for weak-supervision

	"""

	if len(image_size) == 2:
		height, width = image_size
	else:
		channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)
	weight_map = np.zeros([height, width], dtype=np.float32)

	for i, word in enumerate(character_bbox):
		for char_num in range(len(word)-1):
			target, weight_map = add_affinity_others(
				target,
				weight_map,
				weight[i],
				word[char_num][:, 0, :].copy(),
				word[char_num+1][:, 0, :].copy())

	return target/255, weight_map


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

		image = plt.imread(self.base_path+'/'+self.imnames[item][0])  # Read the image
		image, character = resize(image, self.charBB[item].copy())  # Resize the image to (768, 768)
		image = normalize_mean_variance(image).transpose(2, 0, 1)
		weight = generate_target(image.shape, character.copy())  # Generate character heatmap
		weight_affinity = generate_affinity(image.shape, character.copy(), self.txt[item].copy())  # Generate affinity heatmap

		return image.astype(np.float32), weight.astype(np.float32), weight_affinity.astype(np.float32)

	def __len__(self):

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
