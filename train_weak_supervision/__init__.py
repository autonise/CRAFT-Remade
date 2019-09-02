from .trainer import train, test
import train_weak_supervision.config as config
from train_synth.synthesize import generator_
from src.utils.parallel import DataParallelModel
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def get_initial_model_optimizer(path):

	"""
	Function to load pre-trained optimizer and model
	:param path: path to the model
	:return: model, optimizer
	"""

	if config.model_architecture == "UNET_ResNet":
		from src.UNET_ResNet import UNetWithResnet50Encoder
		model = UNetWithResnet50Encoder()
		model = DataParallelModel(model)
		saved_model = torch.load(path)
		model.load_state_dict(saved_model['state_dict'])

	else:
		from src.craft_model import CRAFT
		model = CRAFT()
		model = DataParallelModel(model)
		saved_model = torch.load(path)
		if 'state_dict' in saved_model.keys():
			model.load_state_dict(saved_model['state_dict'])
		else:
			model.load_state_dict(saved_model)

	if config.use_cuda:
		model = model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr[0])

	# ToDo - Check the effects of using a new optimizer after every iteration or using the previous iteration optimizer

	if config.model_architecture == "UNET_ResNet":
		optimizer.load_state_dict(saved_model['optimizer'])

	return model, optimizer


def generate_target(model, iteration):

	"""
	Generate the target after every iteration

	:param model: Pytorch model UNet-ResNet
	:param iteration: weak-supervision current iteration
	:return: None
	"""

	generator_(base_target_path=config.target_path + '/' + str(iteration), model=model)
	torch.cuda.empty_cache()


def save_model(model, optimizer, state, iteration=None, loss=None, accuracy=None):

	"""
	Function to save the model and optimizer state dict
	:param model: Pytorch model
	:param optimizer: Adam Optimizer
	:param state: 'intermediate' or 'final'
	:param iteration: weak-supervision current iteration
	:param accuracy: F-score while training
	:param loss: MSE loss with OHNM while training
	:return: None
	"""

	os.makedirs(config.save_path + '/' + str(iteration), exist_ok=True)

	if state == 'intermediate':
		torch.save(
			{
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, config.save_path + '/' + str(iteration) + '/model.pkl')

		np.save(config.save_path + '/' + str(iteration) + '/loss_plot.npy', loss)

		np.save(config.save_path + '/' + str(iteration) + '/accuracy_plot.npy', accuracy)

		plt.plot(loss)
		plt.savefig(config.save_path + '/' + str(iteration) + '/loss_plot.png')
		plt.clf()

		plt.plot(accuracy)
		plt.savefig(config.save_path + '/' + str(iteration) + '/accuracy_plot.png')
		plt.clf()

	elif state == 'final':
		torch.save(
			{
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, config.save_path + '/final_model.pkl')
