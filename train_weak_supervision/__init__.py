from .trainer import train
from src.model import UNetWithResnet50Encoder
import train_weak_supervision.config as config
from train_synth.synthesize import generator_
from src.utils.parallel import DataParallelModel
import torch


def get_initial_model_optimizer(path):

	"""
	Function to load pre-trained optimizer and model
	:param path: path to the model
	:return: model, optimizer
	"""

	model = UNetWithResnet50Encoder()

	if config.use_cuda:
		model = model.cuda()

	model = DataParallelModel(model)

	model.load_state_dict(torch.load(path)['state_dict'])

	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr[1])

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


def save_model(model, optimizer, state, iteration=None):

	"""
	Function to save the model and optimizer state dict
	:param model: Pytorch model
	:param optimizer: Adam Optimizer
	:param state: 'intermediate' or 'final'
	:param iteration: weak-supervision current iteration
	:return: None
	"""

	if state == 'intermediate':
		torch.save(
			{
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, config.save_path + '/' + str(iteration) + '_model.pkl')
	elif state == 'final':
		torch.save(
			{
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, config.save_path + '/final_model.pkl')
