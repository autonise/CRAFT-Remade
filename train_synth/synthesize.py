import train_synth.config as config
from src.model import UNetWithResnet50Encoder

from src.utils.parallel import DataParallelModel
from src.utils.utils import generate_word_bbox, get_weighted_character_target



from src.utils.utils import calculate_batch_fscore
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
from shapely.geometry import box, Polygon
import cv2
import json



DATA_DEBUG = False


os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)  # Specify which GPU you want to use


def seed():
	"""
	This removes randomness, makes everything deterministic
	:return: None
	"""

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


def synthesize(dataloader, model, base_path_affinity, base_path_character, base_path_bbox):

	"""

	Given a path to a set of images, and path to a pre-trained model, generate the character heatmap and affinity heatmap

	:param dataloader: A Pytorch dataloader for loading and resizing the images of the folder
	:param model: A pre-trained model
	:param base_path_affinity: Path where to store the predicted affinity heatmap
	:param base_path_character: Path where to store the predicted character heatmap
	:param base_path_bbox: Path where to store the word_bbox overlapped on images
	:return: None
	"""

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)

		for no, (image, image_name, original_dim) in enumerate(iterator):

			if config.use_cuda:
				image = image.cuda()

			output = model(image)

			if type(output) == list:

				# If using custom DataParallelModel this is necessary to convert the list to tensor
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()
			original_dim = original_dim.cpu().numpy()

			for i in range(output.shape[0]):

				# --------- Resizing it back to the original image size and saving it ----------- #

				image_i = (image[i].data.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

				max_dim = original_dim[i].max()
				resizing_factor = 768/max_dim
				before_pad_dim = [int(original_dim[i][0]*resizing_factor), int(original_dim[i][1]*resizing_factor)]

				output[i, :, :, :] = np.uint8(output[i, :, :, :]*255)

				height_pad = (768 - before_pad_dim[0])//2
				width_pad = (768 - before_pad_dim[1])//2

				image_i = cv2.resize(
					image_i[height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])
				)

				character_bbox = cv2.resize(
					output[i, 0, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])
				)/255

				affinity_bbox = cv2.resize(
					output[i, 1, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])
				)/255

				predicted_bbox = generate_word_bbox(
					character_bbox,
					affinity_bbox,
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity)['word_bbox']

				predicted_bbox = [np.array(predicted_bbox_i) for predicted_bbox_i in predicted_bbox]

				cv2.drawContours(image_i, predicted_bbox, -1, (0, 255, 0), 2)

				plt.imsave(
					base_path_bbox + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.png',
					image_i)

				plt.imsave(
					base_path_character + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.png',
					np.float32(character_bbox > config.threshold_character),
					cmap='gray')

				plt.imsave(
					base_path_affinity+'/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(affinity_bbox > config.threshold_affinity),
					cmap='gray')


def synthesize_with_score(dataloader, model, base_target_path):

	"""
	Given a path to a set of images(icdar 2013 dataset), and path to a pre-trained model, generate the character heatmap
	and affinity heatmap and a json of all the annotations
	:param dataloader: dataloader for icdar 2013 dataset
	:param model: pre-trained model
	:param base_target_path: path where to store the predictions
	:return:
	"""

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)

		for no, (image, image_name, original_dim, item) in enumerate(iterator):

			annots = []

			for i in item:
				annot = dataloader.dataset.gt['annots'][dataloader.dataset.imnames[i]]
				annots.append(annot)

			if config.use_cuda:
				image = image.cuda()

			output = model(image)

			if type(output) == list:
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()
			original_dim = original_dim.cpu().numpy()

			for i in range(output.shape[0]):

				# --------- Resizing it back to the original image size and saving it ----------- #

				image_i = (image[i].data.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

				max_dim = original_dim[i].max()
				resizing_factor = 768/max_dim
				before_pad_dim = [int(original_dim[i][0]*resizing_factor), int(original_dim[i][1]*resizing_factor)]

				plt.imsave(
					base_target_path + '_affinity/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(output[i, 1, :, :] > config.threshold_affinity),
					cmap='gray')
				plt.imsave(
					base_target_path + '_character/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(output[i, 0, :, :] > config.threshold_character), cmap='gray')

				output[i, :, :, :] = np.uint8(output[i, :, :, :]*255)

				height_pad = (768 - before_pad_dim[0]) // 2
				width_pad = (768 - before_pad_dim[1]) // 2

				image_i = cv2.resize(
					image_i[height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])
				)

				character_bbox = cv2.resize(
					output[i, 0, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0]))/255

				affinity_bbox = cv2.resize(
					output[i, 1, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0]))/255

				generated_targets = generate_word_bbox(
					character_bbox, affinity_bbox,
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity)

				if 'error_message' in generated_targets.keys():
					print('There was an error while generating the target of ', image_name[i])
					print('Error:', generated_targets['error_message'])
					continue

				generated_targets = get_weighted_character_target(
					generated_targets, {'bbox': annots[i]['bbox'], 'text': annots[i]['text']},
					dataloader.dataset.unknown,
					config.threshold_fscore)

				cv2.drawContours(image_i, [np.array(word_bbox) for word_bbox in generated_targets['word_bbox']], -1, (0, 255, 0), 2)

				plt.imsave(base_target_path + '_word_bbox/'+'.'.join(image_name[i].split('.')[:-1])+'.png', image_i)

				f_score=calculate_batch_fscore(generated_targets,original_annotations[i])

				dataloader.set_description(f_score)
				with open(base_target_path + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.json', 'w') as f:
					json.dump(generated_targets, f)


def main(
		folder_path,
		base_path_character=None,
		base_path_affinity=None,
		base_path_bbox=None,
		model_path=None,
		model=None):

	"""
	Entry function for synthesising character and affinity heatmap on images given in a folder using a pre-trained model
	:param folder_path: Path of folder where the images are
	:param base_path_character: Path where to store the character heatmap
	:param base_path_affinity: Path where to store the affinity heatmap
	:param base_path_bbox: Path where to store the generated word_bbox overlapped on the image
	:param model_path: Path where the pre-trained model is stored
	:param model: If model is provided directly use it instead of loading it
	:return:
	"""

	os.makedirs(base_path_affinity, exist_ok=True)
	os.makedirs(base_path_character, exist_ok=True)
	os.makedirs(base_path_bbox, exist_ok=True)

	if base_path_character is None:
		base_path_character = '/'.join(folder_path.split('/')[:-1])+'/target_character'
	if base_path_affinity is None:
		base_path_affinity = '/'.join(folder_path.split('/')[:-1])+'/target_affinity'
	if base_path_bbox is None:
		base_path_bbox = '/'.join(folder_path.split('/')[:-1])+'/word_bbox'

	# Dataloader to pre-process images given in the folder

	infer_dataloader = DataLoaderEval(folder_path)

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=2,
		shuffle=True, num_workers=2)

	if model is None:

		# If model has not been provided, loading it from the path provided

		model = UNetWithResnet50Encoder()
		model = DataParallelModel(model)

		# ToDo - Can't run pre-trained models on CPU

		if config.use_cuda:
			model = model.cuda()

		saved_model = torch.load(model_path)
		model.load_state_dict(saved_model['state_dict'])

	synthesize(infer_dataloader, model, base_path_affinity, base_path_character, base_path_bbox)


def generator_(base_target_path, model_path=None, model=None):

	from train_weak_supervision.dataloader import DataLoaderEvalICDAR2013

	"""
	Generator function to generate weighted heat-maps for weak-supervision training
	:param base_target_path: Path where to store the generated annotations
	:param model_path: If model is not provided then load from model_path
	:param model: Pytorch Model can be directly provided ofr inference
	:return: None
	"""

	os.makedirs(base_target_path, exist_ok=True)
	os.makedirs(base_target_path+'_affinity', exist_ok=True)
	os.makedirs(base_target_path+'_character', exist_ok=True)
	os.makedirs(base_target_path+'_word_bbox', exist_ok=True)

	# Dataloader to pre-process images given in the dataset and provide annotations to generate weight

	infer_dataloader = DataLoaderEvalICDAR2013()

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=2,
		shuffle=True, num_workers=2)

	if model is None:

		# If model has not been provided, loading it from the path provided

		model = UNetWithResnet50Encoder()
		model = DataParallelModel(model)

		if config.use_cuda:
			model = model.cuda()

		saved_model = torch.load(model_path)
		model.load_state_dict(saved_model['state_dict'])

	synthesize_with_score(infer_dataloader, model, base_target_path)
