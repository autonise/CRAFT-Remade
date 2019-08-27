from train_weak_supervision.dataloader import DataLoaderMIX, DataLoaderEvalICDAR2013
import train_weak_supervision.config as config
from src.generic_model import Criterian
from src.utils.parallel import DataParallelCriterion
from src.utils.utils import calculate_batch_fscore, calculate_fscore, resize_bbox
from src.utils.data_manipulation import denormalize_mean_variance

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(no, dataset_name, output, image, character_map, affinity_map, character_weight, affinity_weight):

	os.makedirs('Temporary/' + str(no), exist_ok=True)

	for __, _ in enumerate(dataset_name):

		os.makedirs('Temporary/'+str(no)+'/'+str(__), exist_ok=True)

		plt.imsave('Temporary/'+str(no)+'/'+str(__)+'/image_.png', denormalize_mean_variance(
			image[__].data.cpu().numpy().transpose(1, 2, 0)))

		plt.imsave(
			'Temporary/'+str(no)+'/'+str(__)+'/char_map.png', output[__, 0].data.cpu().numpy(),
			cmap='gray')

		plt.imsave(
			'Temporary/'+str(no)+'/'+str(__)+'/aff_map.png', output[__, 1].data.cpu().numpy(),
			cmap='gray')

		plt.imsave(
			'Temporary/'+str(no)+'/'+str(__)+'/target_char_map.png', character_map[__].data.cpu().numpy(),
			cmap='gray')

		plt.imsave(
			'Temporary/'+str(no)+'/'+str(__)+'/target_affinity_map.png', affinity_map[__].data.cpu().numpy(),
			cmap='gray')

		plt.imsave(
			'Temporary/'+str(no)+'/'+str(__)+'/weight_char_map.png', character_weight[__].data.cpu().numpy(),
			cmap='gray')

		plt.imsave(
			'Temporary/'+str(no)+'/'+str(__)+'/weight_affinity_map.png', affinity_weight[__].data.cpu().numpy(),
			cmap='gray')


def train(model, optimizer, iteration):

	"""
	Train the weak-supervised model iteratively
	:param model: Pre-trained model on SynthText
	:param optimizer: Pre-trained model's optimizer
	:param iteration: current iteration of weak-supervision
	:return: model, optimizer
	"""

	def change_lr():

		# Change learning rate while training
		for param_group in optimizer.param_groups:
			param_group['lr'] = config.lr[iteration]

		print('Learning Rate Changed to ', config.lr[iteration])

	change_lr()

	dataloader = DataLoader(
		DataLoaderMIX('train', iteration), batch_size=config.batch_size['train'], num_workers=0, shuffle=True)
	loss_criterian = DataParallelCriterion(Criterian())

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	all_loss = []
	all_accuracy = []
	all_count = []

	for no, (
			image,
			character_map,
			affinity_map,
			character_weight,
			affinity_weight,
			dataset_name,
			text_target,
			item,
			original_dim) in enumerate(iterator):

		if config.use_cuda:
			image, character_map, affinity_map = image.cuda(), character_map.cuda(), affinity_map.cuda()
			character_weight, affinity_weight = character_weight.cuda(), affinity_weight.cuda()

		output = model(image)
		loss = loss_criterian(
			output,
			character_map,
			affinity_map,
			character_weight,
			affinity_weight
		).mean()/config.optimizer_iterations

		all_loss.append(loss.item()*config.optimizer_iterations)

		loss.backward()

		if (no + 1) % config.optimizer_iterations == 0:
			optimizer.step()
			optimizer.zero_grad()

		# ---------- Calculating the F-score ------------ #

		if type(output) == list:
			output = torch.cat(output, dim=0)

		output[output < 0] = 0
		output[output > 1] = 1

		target_ic13 = []
		predicted_ic13 = []
		target_text = []
		current_count = 0

		if no % config.check_iterations == 0:

			save(no, dataset_name, output, image, character_map, affinity_map, character_weight, affinity_weight)

		output = output.data.cpu().numpy()
		original_dim = original_dim.numpy()

		for __, _ in enumerate(dataset_name):

			if _ != 'SYNTH':

				predicted_ic13.append(resize_bbox(original_dim[__], output[__], config)['word_bbox'])
				target_ic13.append(np.array(dataloader.dataset.gt[item[__]][1]['word_bbox'].copy(), dtype=np.int32))
				target_text.append(text_target[__].split('~'))

				current_count += 1

		if len(predicted_ic13) != 0:

			all_accuracy.append(
				calculate_batch_fscore(
					predicted_ic13,
					target_ic13,
					text_target=target_text,
					threshold=config.threshold_fscore)*current_count
			)

			all_count.append(current_count)

		# ------------- Setting Description ---------------- #

		if np.array(all_count)[-min(1000, len(all_count)):].sum() != 0:

			f_score = int(
						np.array(all_accuracy)[-min(1000, len(all_accuracy)):].sum() * 100000000 /
						np.array(all_count)[-min(1000, len(all_count)):].sum()) / 100000000
		else:
			f_score = 0

		iterator.set_description(
			'Loss:' + str(int(loss.item() * config.optimizer_iterations * 100000) / 100000) + ' Iterations:[' + str(no)
			+ '/' + str(len(iterator)) +
			'] Average Loss:' + str(
				int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000) / 100000) +
			'| Average F-Score: ' + str(f_score)
		)

	if len(iterator) % config.optimizer_iterations != 0:

		optimizer.step()
		optimizer.zero_grad()

	torch.cuda.empty_cache()

	return model, optimizer, all_loss, all_accuracy


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

				f_score.append(
					calculate_fscore(
						resize_bbox(original_dim[i], output[i], config)['word_bbox'][:, :, 0, :],
						np.array(annots[i]['bbox']),
						text_target=annots[i]['text'],
					)
				)

				# --------------- PostProcessing for creating the targets for the next iteration ---------------- #

			all_accuracy.append(np.mean(f_score))

			iterator.set_description('F-score: ' + str(np.mean(all_accuracy)))

		torch.cuda.empty_cache()

	return np.mean(all_accuracy)
