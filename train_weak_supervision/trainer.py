from train_weak_supervision.dataloader import DataLoaderMIX
import train_weak_supervision.config as config
from src.model import Criterian
from src.utils.parallel import DataParallelCriterion
from src.utils.utils import calculate_batch_fscore, generate_word_bbox

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(data, output, target, target_affinity, epoch, no):

	"""
	Saving the synthesised outputs in between the training
	:param data: image as tensor
	:param output: predicted output from the model as tensor
	:param target: character heatmap target as tensor
	:param target_affinity: affinity heatmap target as tensor
	:param no: current iteration number
	:param epoch: current epoch
	:return: None
	"""

	output = output.data.cpu().numpy()
	data = data.data.cpu().numpy()
	target = target.data.cpu().numpy()
	target_affinity = target_affinity.data.cpu().numpy()

	batch_size = output.shape[0]

	base = config.DataLoaderICDAR2013_Synthesis + str(epoch) + '_' + str(no) + '/'

	os.makedirs(base, exist_ok=True)

	for i in range(batch_size):

		os.makedirs(base+str(i), exist_ok=True)
		character_bbox = output[i, 0, :, :]
		affinity_bbox = output[i, 1, :, :]

		plt.imsave(base+str(i) + '/image.png', data[i].transpose(1, 2, 0))

		plt.imsave(base+str(i) + '/target_characters.png', target[i, :, :], cmap='gray')
		plt.imsave(base+str(i) + '/target_affinity.png', target_affinity[i, :, :], cmap='gray')

		plt.imsave(base + str(i) + '/pred_characters.png', character_bbox, cmap='gray')
		plt.imsave(base + str(i) + '/pred_affinity.png', affinity_bbox, cmap='gray')

		# Thresholding the character and affinity heatmap

		plt.imsave(
			base + str(i) + '/pred_characters_thresh.png',
			np.float32(character_bbox > config.threshold_character),
			cmap='gray')
		plt.imsave(
			base + str(i) + '/pred_affinity_thresh.png',
			np.float32(affinity_bbox > config.threshold_affinity),
			cmap='gray')


def train(model, optimizer, iteration):

	"""
	Train the weak-supervised model iteratively
	:param model: Pre-trained model on SynthText
	:param optimizer: Pre-trained model's optimizer
	:param iteration: current iteration of weak-supervision
	:return: model, optimizer
	"""

	dataloader = DataLoader(DataLoaderMIX('train', iteration), batch_size=config.batch_size['train'], num_workers=0)
	loss_criterian = DataParallelCriterion(Criterian())

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	"""
	Currently not changing the learning rate while weak supervision
	
	def change_lr(no):

		# Change learning rate while training

		for i in config.lr:
			if i == no:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer.param_groups:
					param_group['lr'] = config.lr[i]

	change_lr(1)
	"""

	all_loss = []
	all_accuracy = []
	all_count = []

	ground_truth = iterator.iterable.dataset.gt

	for no, (image, character_map, affinity_map, character_weight, affinity_weight, word_bbox, original_dim) in \
		enumerate(iterator):

		if config.use_cuda:
			image, character_map, affinity_map = image.cuda(), character_map.cuda(), affinity_map.cuda()
			character_weight, affinity_weight = character_weight.cuda(), affinity_weight.cuda()

		output = model(image)
		loss = loss_criterian(output, character_map, affinity_map, character_weight, affinity_weight).mean()

		all_loss.append(loss.item())

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		# ---------- Calculating the F-score ------------ #

		if type(output) == list:
			output = torch.cat(output, dim=0)

		output = output.data.cpu().numpy()
		# image = image.data.cpu().numpy()
		original_dim = original_dim.cpu().numpy()

		target_bbox = []
		predicted_ic13 = []
		current_count = 0

		word_bbox = word_bbox.numpy()

		for __, _ in enumerate(word_bbox):

			if _[1] == 1:

				# ToDo - Understand why model.train() gives poor results but model.eval() with torch.no_grad() gives better results

				max_dim = original_dim[__].max()
				resizing_factor = 768 / max_dim
				before_pad_dim = [int(original_dim[__][0] * resizing_factor), int(original_dim[__][1] * resizing_factor)]

				output[__, :, :, :] = np.uint8(output[__, :, :, :] * 255)

				height_pad = (768 - before_pad_dim[0]) // 2
				width_pad = (768 - before_pad_dim[1]) // 2

				character_bbox = cv2.resize(
					output[__, 0, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[__][1], original_dim[__][0])) / 255

				affinity_bbox = cv2.resize(
					output[__, 1, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[__][1], original_dim[__][0])) / 255

				# plt.imsave('char_heatmap_after.png', character_bbox, cmap='gray')
				#
				# image_i = cv2.resize(
				# 	np.uint8(image[__, :, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad +
				# 	before_pad_dim[1]]*255).transpose(1, 2, 0),
				# 	(original_dim[__][1], original_dim[__][0]))

				predicted_bbox = generate_word_bbox(
					character_bbox,
					affinity_bbox,
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity)['word_bbox']

				predicted_ic13.append(predicted_bbox)
				target_bbox.append(np.array(ground_truth[_[0] % len(ground_truth)][1]['word_bbox'], dtype=np.int64))

				# cv2.drawContours(image_i, target_bbox[-1], -1, (0, 255, 0), 2)
				# cv2.drawContours(image_i, predicted_bbox, -1, (255, 0, 0), 2)
				#
				# plt.imsave(str(__) + '.png', image_i)

				current_count += 1

		all_accuracy.append(
			calculate_batch_fscore(
				predicted_ic13,
				target_bbox,
				threshold=config.threshold_fscore)*current_count)

		all_count.append(current_count)

		# ------------- Setting Description ---------------- #

		if np.array(all_count)[-min(1000, len(all_count)):].sum() != 0:
			f_score = int(
						np.array(all_accuracy)[-min(1000, len(all_accuracy)):].sum() * 100000000 /
						np.array(all_count)[-min(1000, len(all_count)):].sum()) / 100000000
		else:
			f_score = 0

		iterator.set_description(
			'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
				len(iterator)) +
			'] Average Loss:' + str(
				int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000000) / 100000000) +
			'| Average F-Score: ' + str(f_score)

		)

	torch.cuda.empty_cache()

	return model, optimizer
