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


def save(data, output, target, target_affinity, epoch, no):
	output = output.data.cpu().numpy()
	data = data.data.cpu().numpy()
	target = target.data.cpu().numpy()
	target_affinity = target_affinity.data.cpu().numpy()

	batchsize = output.shape[0]

	base = 'test_synthesis/' + str(epoch) + '_' + str(no) + '/'

	os.makedirs(base, exist_ok=True)

	for i in range(batchsize):
		os.makedirs(base + str(i), exist_ok=True)
		character_bbox = output[i, 0, :, :]
		affinity_bbox = output[i, 1, :, :]

		plt.imsave(base + str(i) + '/image.png', data[i].transpose(1, 2, 0))

		plt.imsave(base + str(i) + '/target_characters.png', target[i, :, :], cmap='gray')
		plt.imsave(base + str(i) + '/target_affinity.png', target_affinity[i, :, :], cmap='gray')

		plt.imsave(base + str(i) + '/pred_characters.png', character_bbox, cmap='gray')
		plt.imsave(base + str(i) + '/pred_affinity.png', affinity_bbox, cmap='gray')

		plt.imsave(
			base + str(i) + '/pred_characters_thresh.png',
			np.float32(character_bbox > config.threshold_character), cmap='gray')
		plt.imsave(
			base + str(i) + '/pred_affinity_thresh.png', np.float32(affinity_bbox > config.threshold_affinity),
			cmap='gray')


def synthesize(dataloader, model):

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)

		for no, (image, weight, weight_affinity) in enumerate(iterator):

			if DATA_DEBUG:
				continue

			if config.use_cuda:
				image, weight, weight_affinity = image.cuda(), weight.cuda(), weight_affinity.cuda()

			output = model(image)

			if type(output) == list:
				output = torch.cat(output, dim=0)
			save(image, output, weight, weight_affinity, 0, no)  # Change


def seed():
	# This removes randomness, makes everything deterministic

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


def main(model_path, folder_path):
	seed()

	model = UNetWithResnet50Encoder()
	model = DataParallelModel(model)

	infer_dataloader = DataLoaderEval(folder_path)  # Change this

	if config.use_cuda:
		model = model.cuda()

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=10,
		shuffle=True, num_workers=16)  # Change this

	saved_model = torch.load(model_path)
	model.load_state_dict(saved_model['state_dict'])

	synthesize(infer_dataloader, model)
