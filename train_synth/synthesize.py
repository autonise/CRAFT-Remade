import train_synth.config as config
from src.model import UNetWithResnet50Encoder
from train_synth.dataloader import DataLoaderEval, DataLoaderEvalICDAR2013
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from src.utils.parallel import DataParallelModel
from src.utils.utils import generate_bbox, get_weighted_character_target
import cv2
import json


DATA_DEBUG = False

if DATA_DEBUG:
	config.num_cuda = '0'
	config.batchsize['test'] = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def seed():
	# This removes randomness, makes everything deterministic

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


def synthesize(dataloader, model, base_path_affinity, base_path_character):

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)

		for no, (image, image_name, original_dim) in enumerate(iterator):

			if DATA_DEBUG:
				continue

			if config.use_cuda:
				image = image.cuda()

			output = model(image)

			if type(output) == list:
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()
			original_dim = original_dim.cpu().numpy()

			for i in range(output.shape[0]):

				max_dim = original_dim[i].max()
				resizing_factor = 768/max_dim
				before_pad_dim = [int(original_dim[i][0]*resizing_factor), int(original_dim[i][1]*resizing_factor)]

				output[i, :, :, :] = np.uint8(output[i, :, :, :]*255)

				character_bbox = cv2.resize(output[i, 0, (768 - before_pad_dim[0])//2:(768 - before_pad_dim[0])//2+ before_pad_dim[0], (768 - before_pad_dim[1])//2:(768 - before_pad_dim[1])//2 + before_pad_dim[1]], (original_dim[i][1], original_dim[i][0]))/255
				affinity_bbox = cv2.resize(output[i, 1, (768 - before_pad_dim[0])//2:(768 - before_pad_dim[0])//2+ before_pad_dim[0], (768 - before_pad_dim[1])//2:(768 - before_pad_dim[1])//2 + before_pad_dim[1]], (original_dim[i][1], original_dim[i][0]))/255

				plt.imsave(
					base_path_character+'/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(character_bbox > config.threshold_character),
					cmap='gray')

				plt.imsave(
					base_path_affinity+'/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(affinity_bbox > config.threshold_affinity),
					cmap='gray')


def synthesize_with_score(dataloader, model, base_target_path):

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)

		for no, (image, image_name, original_dim, item) in enumerate(iterator):

			annots = []

			for i in item:
				annot = dataloader.dataset.gt['annots'][dataloader.dataset.imnames[i]]
				annots.append(annot)

			if DATA_DEBUG:
				continue

			if config.use_cuda:
				image = image.cuda()

			output = model(image)

			if type(output) == list:
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()
			original_dim = original_dim.cpu().numpy()

			for i in range(output.shape[0]):

				max_dim = original_dim[i].max()
				resizing_factor = 768/max_dim
				before_pad_dim = [int(original_dim[i][0]*resizing_factor), int(original_dim[i][1]*resizing_factor)]

				output[i, :, :, :] = np.uint8(output[i, :, :, :]*255)

				character_bbox = cv2.resize(output[i, 0, (768 - before_pad_dim[0])//2:(768 - before_pad_dim[0])//2+ before_pad_dim[0], (768 - before_pad_dim[1])//2:(768 - before_pad_dim[1])//2 + before_pad_dim[1]], (original_dim[i][1], original_dim[i][0]))/255
				affinity_bbox = cv2.resize(output[i, 1, (768 - before_pad_dim[0])//2:(768 - before_pad_dim[0])//2+ before_pad_dim[0], (768 - before_pad_dim[1])//2:(768 - before_pad_dim[1])//2 + before_pad_dim[1]], (original_dim[i][1], original_dim[i][0]))/255

				generated_targets = generate_bbox(
					character_bbox, affinity_bbox,
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity)

				if 'error_message' in generated_targets.keys():
					print('There was an error while generating the target of ', image_name[i])
					print('Error:', generated_targets['error_message'])
					continue

				generated_targets = get_weighted_character_target(generated_targets, {'bbox': annots[i]['bbox'], 'text': annots[i]['text']}, dataloader.dataset.unknown)

				with open(base_target_path + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.json', 'w') as f:
					json.dump(generated_targets, f)


def main(folder_path, base_path_character=None, base_path_affinity=None, model_path=None, model=None):

	os.makedirs(base_path_affinity, exist_ok=True)
	os.makedirs(base_path_character, exist_ok=True)

	if base_path_character is None:
		base_path_character = '/'.join(folder_path.split('/')[:-1])+'/target_character'
	if base_path_affinity is None:
		base_path_affinity = '/'.join(folder_path.split('/')[:-1])+'/target_affinity'

	infer_dataloader = DataLoaderEval(folder_path)

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=2,
		shuffle=True, num_workers=2)

	if model is None:
		model = UNetWithResnet50Encoder()
		model = DataParallelModel(model)

		if config.use_cuda:
			model = model.cuda()

		saved_model = torch.load(model_path)
		model.load_state_dict(saved_model['state_dict'])

	synthesize(infer_dataloader, model, base_path_affinity, base_path_character)


def generator(folder_path, base_target_path, model_path=None, model=None):

	os.makedirs(base_target_path, exist_ok=True)

	infer_dataloader = DataLoaderEvalICDAR2013(folder_path)

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=2,
		shuffle=True, num_workers=2)

	if model is None:
		model = UNetWithResnet50Encoder()
		model = DataParallelModel(model)

		if config.use_cuda:
			model = model.cuda()

		saved_model = torch.load(model_path)
		model.load_state_dict(saved_model['state_dict'])

	synthesize_with_score(infer_dataloader, model, base_target_path)
