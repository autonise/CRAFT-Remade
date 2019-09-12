from train_weak_supervision.dataloader import DataLoaderMIX, DataLoaderEvalOther
import train_weak_supervision.config as config
from src.generic_model import Criterian
from src.utils.parallel import DataParallelCriterion
from src.utils.utils import calculate_batch_fscore, calculate_fscore, resize_bbox, _init_fn, generate_word_bbox
from src.utils.data_manipulation import denormalize_mean_variance

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(no, dataset_name, output, image, character_map, affinity_map, character_weight, affinity_weight):

	os.makedirs('Temporary/' + str(no), exist_ok=True)

	for __, _ in enumerate(dataset_name):

		os.makedirs('Temporary/'+str(no)+'/'+str(__), exist_ok=True)

		generated = generate_word_bbox(
			output[__, 0].data.cpu().numpy(), output[__, 1].data.cpu().numpy(),
			config.threshold_character, config.threshold_affinity, config.threshold_word,
			config.threshold_character_upper, config.threshold_affinity_upper, config.scale_character, config.scale_affinity
		)

		output_image = denormalize_mean_variance(
			image[__].data.cpu().numpy().transpose(1, 2, 0))

		cv2.drawContours(output_image, generated['word_bbox'], -1, (0, 255, 0), 2)

		plt.imsave('Temporary/'+str(no)+'/'+str(__)+'/image_.png', output_image)

		cv2.imwrite('Temporary/'+str(no)+'/'+str(__)+'/char_map.png', np.uint8(output[__, 0].data.cpu().numpy()*255))

		cv2.imwrite('Temporary/'+str(no)+'/'+str(__)+'/aff_map.png', np.uint8(output[__, 1].data.cpu().numpy()*255))

		cv2.imwrite(
			'Temporary/' + str(no) + '/' + str(__) + '/char_map_threshold_upper.png',
			np.uint8(np.float32(output[__, 0].data.cpu().numpy() > config.threshold_character_upper) * 255))

		cv2.imwrite(
			'Temporary/' + str(no) + '/' + str(__) + '/aff_map_threshold_upper.png',
			np.uint8(np.float32(output[__, 1].data.cpu().numpy() > config.threshold_affinity_upper) * 255))

		cv2.imwrite(
			'Temporary/' + str(no) + '/' + str(__) + '/char_map_threshold_lower.png',
			np.uint8(np.float32(output[__, 0].data.cpu().numpy() > config.threshold_character) * 255))

		cv2.imwrite(
			'Temporary/' + str(no) + '/' + str(__) + '/aff_map_threshold_lower.png',
			np.uint8(np.float32(output[__, 1].data.cpu().numpy() > config.threshold_affinity) * 255))

		cv2.imwrite(
			'Temporary/'+str(no)+'/'+str(__)+'/target_char_map.png', np.uint8(character_map[__].data.cpu().numpy()*255))

		cv2.imwrite(
			'Temporary/'+str(no)+'/'+str(__)+'/target_affinity_map.png', np.uint8(affinity_map[__].data.cpu().numpy()*255))

		cv2.imwrite(
			'Temporary/'+str(no)+'/'+str(__)+'/weight_char_map.png', np.uint8(character_weight[__].data.cpu().numpy()*255))

		cv2.imwrite(
			'Temporary/'+str(no)+'/'+str(__)+'/weight_affinity_map.png', np.uint8(affinity_weight[__].data.cpu().numpy()*255))


def change_lr(optimizer, lr):

	# Change learning rate while training
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	print('Learning Rate Changed to ', lr)

	return optimizer


def train(model, optimizer, iteration):

	"""
	Train the weak-supervised model iteratively
	:param model: Pre-trained model on SynthText
	:param optimizer: Pre-trained model's optimizer
	:param iteration: current iteration of weak-supervision
	:return: model, optimizer
	"""

	optimizer = change_lr(optimizer, config.lr[iteration])

	dataloader = DataLoader(
		DataLoaderMIX('train', iteration),
		batch_size=config.batch_size['train'],
		num_workers=config.num_workers['train'],
		shuffle=True, worker_init_fn=_init_fn,
	)

	loss_criterian = DataParallelCriterion(Criterian())

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	all_loss = []
	all_precision = []
	all_f_score = []
	all_recall = []
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

		if (no + 1) % config.change_lr == 0:
			optimizer = change_lr(optimizer, config.lr[iteration]*(0.8**((no + 1)//config.change_lr - 1)))

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

		if (no + 1) % config.check_iterations == 0:

			if type(output) == list:
				output = torch.cat(output, dim=0)

			output[output < 0] = 0
			output[output > 1] = 1

			save(no, dataset_name, output, image, character_map, affinity_map, character_weight, affinity_weight)

		if (no + 1) % config.calc_f_score == 0:

			if type(output) == list:
				output = torch.cat(output, dim=0)

			target_ic13 = []
			predicted_ic13 = []
			target_text = []
			current_count = 0

			output = output.data.cpu().numpy()

			output[output > 1] = 1
			output[output < 0] = 0

			original_dim = original_dim.numpy()

			for __, _ in enumerate(dataset_name):

				if _ != 'SYNTH':

					predicted_ic13.append(resize_bbox(original_dim[__], output[__], config)['word_bbox'])
					target_ic13.append(np.array(dataloader.dataset.gt[item[__]][1]['word_bbox'].copy(), dtype=np.int32))
					target_text.append(text_target[__].split('#@#@#@'))

					current_count += 1

			if len(predicted_ic13) != 0:

				f_score, precision, recall = calculate_batch_fscore(
						predicted_ic13,
						target_ic13,
						text_target=target_text,
						threshold=config.threshold_fscore)

				all_f_score.append(f_score*current_count)
				all_precision.append(precision*current_count)
				all_recall.append(recall*current_count)

				all_count.append(current_count)

		# ------------- Setting Description ---------------- #

		if np.array(all_count)[-min(100, len(all_count)):].sum() != 0:

			count = np.array(all_count)[-min(100, len(all_count)):].sum()

			f_score = int(np.array(all_f_score)[-min(100, len(all_f_score)):].sum() * 10000 / count) / 10000
			precision = int(np.array(all_precision)[-min(100, len(all_precision)):].sum() * 10000 / count) / 10000
			recall = int(np.array(all_recall)[-min(100, len(all_recall)):].sum() * 10000 / count) / 10000
		else:

			f_score = 0
			precision = 0
			recall = 0
#

		iterator.set_description(
			'Loss:' + str(int(loss.item() * config.optimizer_iterations * 100000) / 100000) + ' Iterations:[' + str(no)
			+ '/' + str(len(iterator)) +
			'] Average Loss:' + str(
				int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000) / 100000) +
			'| Average F-Score: ' + str(f_score) +
			'| Average Recall: ' + str(recall) +
			'| Average Precision: ' + str(precision)
		)

		if (no + 1) % config.test_now == 0:

			del image, loss, affinity_weight, character_weight, affinity_map, character_map, output
			print('\nF-score of testing: ', test(model, iteration), '\n')
			model.train()

	if len(iterator) % config.optimizer_iterations != 0:

		optimizer.step()
		optimizer.zero_grad()

	torch.cuda.empty_cache()

	return model, optimizer, all_loss, all_f_score


def test(model, iteration):

	"""
	Test the weak-supervised model
	:param model: Pre-trained model on SynthText
	:param iteration: Iteration Number
	:return: F-score, loss
	"""

	os.makedirs(config.save_path + '/Test_'+str(iteration), exist_ok=True)

	dataloader = DataLoader(
		DataLoaderEvalOther('test'),
		batch_size=config.batch_size['test'],
		num_workers=config.num_workers['test'],
		shuffle=False, worker_init_fn=_init_fn
	)

	true_positive = 0
	false_positive = 0
	num_positive = 0

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

			output[output > 1] = 1
			output[output < 0] = 0

			original_dim = original_dim.cpu().numpy()

			f_score = []

			for i in range(output.shape[0]):

				# --------- Resizing it back to the original image size and saving it ----------- #

				cur_image = denormalize_mean_variance(image[i].data.cpu().numpy().transpose(1, 2, 0))

				max_dim = original_dim[i].max()
				resizing_factor = 768 / max_dim
				before_pad_dim = [int(original_dim[i][0] * resizing_factor), int(original_dim[i][1] * resizing_factor)]

				height_pad = (768 - before_pad_dim[0]) // 2
				width_pad = (768 - before_pad_dim[1]) // 2

				cur_image = cv2.resize(
					cur_image[height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0]))

				cv2.drawContours(cur_image, resize_bbox(original_dim[i], output[i], config)['word_bbox'], -1, (0, 255, 0), 2)
				cv2.drawContours(cur_image, np.array(annots[i]['bbox']), -1, (0, 0, 255), 2)

				plt.imsave(
					config.save_path + '/Test_' + str(iteration) + '/' + image_name[i],
					cur_image.astype(np.uint8))

				score_calc = calculate_fscore(
						resize_bbox(original_dim[i], output[i], config)['word_bbox'][:, :, 0, :],
						np.array(annots[i]['bbox']),
						text_target=annots[i]['text'],
					)

				f_score.append(
					score_calc['f_score']
				)
				true_positive += score_calc['true_positive']
				false_positive += score_calc['false_positive']
				num_positive += score_calc['num_positive']

				# --------------- PostProcessing for creating the targets for the next iteration ---------------- #

			all_accuracy.append(np.mean(f_score))

			precision = true_positive / (true_positive + false_positive)
			recall = true_positive / num_positive

			iterator.set_description(
				'F-score: ' + str(np.mean(all_accuracy)) + '| Cumulative F-score: '
				+ str(2*precision*recall/(precision + recall)))

		torch.cuda.empty_cache()

	return 2*precision*recall/(precision + recall), precision, recall
