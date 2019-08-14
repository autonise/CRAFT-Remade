from train_weak_supervision.dataloader import DataLoaderMIX
import train_weak_supervision.config as config
from src.model import Criterian
from src.utils.parallel import DataParallelCriterion
from src.utils.utils import calculate_batch_fscore, generate_word_bbox_batch

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

DATA_DEBUG = False
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(data, output, target, target_affinity, epoch, no):

	output = output.data.cpu().numpy()
	data = data.data.cpu().numpy()
	target = target.data.cpu().numpy()
	target_affinity = target_affinity.data.cpu().numpy()

	batchsize = output.shape[0]

	base = config.DataLoaderICDAR2013_Synthesis + str(epoch) + '_' + str(no) + '/'

	os.makedirs(base, exist_ok=True)
	for i in range(batchsize):
		os.makedirs(base+str(i), exist_ok=True)
		character_bbox = output[i, 0, :, :]
		affinity_bbox = output[i, 1, :, :]

		plt.imsave(base+str(i) + '/image.png', data[i].transpose(1, 2, 0))

		plt.imsave(base+str(i) + '/target_characters.png', target[i, :, :], cmap='gray')
		plt.imsave(base+str(i) + '/target_affinity.png', target_affinity[i, :, :], cmap='gray')

		plt.imsave(base + str(i) + '/pred_characters.png', character_bbox, cmap='gray')
		plt.imsave(base + str(i) + '/pred_affinity.png', affinity_bbox, cmap='gray')

		plt.imsave(base + str(i) + '/pred_characters_thresh.png', np.float32(character_bbox>config.threshold_character), cmap='gray')
		plt.imsave(base + str(i) + '/pred_affinity_thresh.png', np.float32(affinity_bbox>config.threshold_affinity), cmap='gray')


def train(model, optimizer, iteration):

	dataloader = DataLoader(DataLoaderMIX('train', iteration), batch_size=config.batch_size['train'], num_workers=8)
	lossCriterian = DataParallelCriterion(Criterian())

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	"""
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

	for no, (image, character_map, affinity_map, character_weight, affinity_weight) in enumerate(iterator):

		if DATA_DEBUG:
			continue

		if config.use_cuda:
			image, character_map, affinity_map = image.cuda(), character_map.cuda(), affinity_map.cuda()
			character_weight, affinity_weight = character_weight.cuda(), affinity_weight.cuda()

		output = model(image)
		loss = lossCriterian(output, character_map, affinity_map, character_weight, affinity_weight).mean()

		all_loss.append(loss.item())

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if len(all_accuracy) == 0:
			iterator.set_description(
				'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
					len(iterator)) +
				'] Average Loss:' + str(
					int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000000) / 100000000))

		else:

			iterator.set_description(
				'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
					len(iterator)) +
				'] Average Loss:' + str(
					int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000000) / 100000000) +
				'| Average F-Score: ' + str(
					int(np.array(all_accuracy)[-min(1000, len(all_accuracy)):].mean() * 100000000) / 100000000)
			)

		if type(output) == list:
			output = torch.cat(output, dim=0)
		predicted_bbox = generate_word_bbox_batch(
			output[:, 0, :, :].data.cpu().numpy(),
			output[:, 1, :, :].data.cpu().numpy(),
			character_threshold=config.threshold_character,
			affinity_threshold=config.threshold_affinity)
		target_bbox = generate_word_bbox_batch(
			character_map.data.cpu().numpy(),
			affinity_map.data.cpu().numpy(),
			character_threshold=config.threshold_character,
			affinity_threshold=config.threshold_affinity)
		all_accuracy.append(
			calculate_batch_fscore(predicted_bbox, target_bbox, threshold=config.threshold_fscore))

	torch.cuda.empty_cache()

	return model, optimizer
