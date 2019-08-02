import train_synth.config as config
from src.model import UNetWithResnet50Encoder
from train_synth.dataloader import DataLoaderEval
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from src.utils.parallel import DataParallelModel

DATA_DEBUG = False

if DATA_DEBUG:
	config.num_cuda = '0'
	config.batchsize['test'] = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def synthesize(dataloader, model, base_path_affinity, base_path_character):

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)

		for no, (image, image_name) in enumerate(iterator):

			if DATA_DEBUG:
				continue

			if config.use_cuda:
				image, weight, weight_affinity = image.cuda(), weight.cuda(), weight_affinity.cuda()

			output = model(image)

			if type(output) == list:
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()

			for i in range(output.shape[0]):

				character_bbox = output[i, 0, :, :]
				affinity_bbox = output[i, 1, :, :]

				plt.imsave(
					base_path_character+'/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(character_bbox > config.threshold_character),
					cmap='gray')

				plt.imsave(
					base_path_affinity+'/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(affinity_bbox > config.threshold_affinity),
					cmap='gray')


def seed():
	# This removes randomness, makes everything deterministic

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


def main(folder_path, base_path_character=None, base_path_affinity=None, model_path=None, model=None):

	os.makedirs(base_path_affinity, exist_ok=True)
	os.makedirs(base_path_character, exist_ok=True)

	if base_path_character is None:
		base_path_character = '/'.join(folder_path.split('/')[:-1])+'/target_character'
	if base_path_affinity is None:
		base_path_affinity = '/'.join(folder_path.split('/')[:-1])+'/target_affinity'

	infer_dataloader = DataLoaderEval(folder_path)

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=16,
		shuffle=True, num_workers=8)

	if model is None:
		model = UNetWithResnet50Encoder()

		if config.use_cuda:
			model = model.cuda()

		saved_model = torch.load(model_path)
		model.load_state_dict(saved_model['state_dict'])

	synthesize(infer_dataloader, model, base_path_affinity, base_path_character)
