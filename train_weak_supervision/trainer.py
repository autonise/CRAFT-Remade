from train_weak_supervision.dataloader import DataLoaderMIX, DataLoaderEvalICDAR2013
import train_weak_supervision.config as config
from src.model import Criterian
from src.utils.parallel import DataParallelCriterion
from src.utils.utils import calculate_batch_fscore, generate_word_bbox, calculate_fscore

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

	dataloader = DataLoader(
		DataLoaderMIX('train', iteration), batch_size=config.batch_size['train'], num_workers=8, shuffle=True)
	loss_criterian = DataParallelCriterion(Criterian())

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	def change_lr(no_i):

		# Change learning rate while training

		for i in config.lr:
			if i == no_i:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer.param_groups:
					param_group['lr'] = config.lr[i]

	all_loss = []
	all_accuracy = []
	all_count = []

	ground_truth = iterator.iterable.dataset.gt

	for no, (image, character_map, affinity_map, character_weight, affinity_weight, word_bbox, original_dim) in \
		enumerate(iterator):

		change_lr(no)

		if config.use_cuda:
			image, character_map, affinity_map = image.cuda(), character_map.cuda(), affinity_map.cuda()
			character_weight, affinity_weight = character_weight.cuda(), affinity_weight.cuda()

		output = model(image)
		loss = loss_criterian(output, character_map, affinity_map, character_weight, affinity_weight).mean()/4

		all_loss.append(loss.item()*4)

		loss.backward()

		if (no + 1) % 4 == 0:
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

				# os.makedirs('Check/'+str(no), exist_ok=True)
				#
				# plt.imsave('Check/'+str(no)+'/image_'+str(__)+'.png', image[__].data.cpu().numpy().transpose(1, 2, 0))
				# plt.imsave('Check/'+str(no)+'/character_map_'+str(__)+'.png', image[__].data.cpu().numpy().transpose(1, 2, 0)*character_map[__].data.cpu().numpy()[:, :, None])
				# plt.imsave('Check/'+str(no)+'/affinity_map_'+str(__)+'.png', image[__].data.cpu().numpy().transpose(1, 2, 0)*affinity_map[__].data.cpu().numpy()[:, :, None])
				# plt.imsave('Check/'+str(no)+'/character_weight_'+str(__)+'.png', character_map[__].data.cpu().numpy()*character_weight[__].data.cpu().numpy(), cmap='gray')
				# plt.imsave('Check/'+str(no)+'/affinity_weight_'+str(__)+'.png', affinity_map[__].data.cpu().numpy()*affinity_weight[__].data.cpu().numpy(), cmap='gray')

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
			'Loss:' + str(int(loss.item() * 4 * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
				len(iterator)) +
			'] Average Loss:' + str(
				int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000000) / 100000000) +
			'| Average F-Score: ' + str(f_score)
		)

	if len(iterator) % 4 != 0:

		optimizer.step()
		optimizer.zero_grad()

	torch.cuda.empty_cache()

	return model, optimizer


def test(model):

	"""
	Test the weak-supervised model
	:param model: Pre-trained model on SynthText
	:return: F-score, loss
	"""

	dataloader = DataLoader(
		DataLoaderEvalICDAR2013('test'), batch_size=config.batch_size['train'], num_workers=8, shuffle=False)

	with torch.no_grad():
		model.eval()
		iterator = tqdm(dataloader)
		all_accuracy = []

		ground_truth = dataloader.dataset.gt

		for no, (image, image_name, original_dim, item) in enumerate(iterator):

			annots = []

			for i in item:
				annot = ground_truth['annots'][dataloader.dataset.imnames[i]]
				annots.append(annot)

			if config.use_cuda:
				image = image.cuda()

			output = model(image)

			if type(output) == list:
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()
			original_dim = original_dim.cpu().numpy()

			f_score = []

			for i in range(output.shape[0]):
				# --------- Resizing it back to the original image size and saving it ----------- #

				max_dim = original_dim[i].max()
				resizing_factor = 768 / max_dim
				before_pad_dim = [int(original_dim[i][0] * resizing_factor), int(original_dim[i][1] * resizing_factor)]

				output[i, :, :, :] = np.uint8(output[i, :, :, :] * 255)

				height_pad = (768 - before_pad_dim[0]) // 2
				width_pad = (768 - before_pad_dim[1]) // 2

				character_bbox = cv2.resize(
					output[i, 0, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])) / 255

				affinity_bbox = cv2.resize(
					output[i, 1, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])) / 255

				generated_targets = generate_word_bbox(
					character_bbox, affinity_bbox,
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity)

				predicted_word_bbox = generated_targets['word_bbox'].copy()

				f_score.append(calculate_fscore(predicted_word_bbox[:, :, 0, :], np.array(annots[i]['bbox'])))
				# --------------- PostProcessing for creating the targets for the next iteration ---------------- #

			all_accuracy.append(np.mean(f_score))

			iterator.set_description('F-score: ' + str(np.mean(all_accuracy)))

		torch.cuda.empty_cache()

	return np.mean(all_accuracy)
