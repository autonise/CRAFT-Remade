import click
import os
import torch
import numpy as np
import random


def seed(config):

	# This removes randomness, makes everything deterministic

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


@click.group()
def main():
	pass


@main.command()
@click.option('-mode', '--mode', help='Train or Test or Synthesize', required=True)
@click.option('-model', '--model', help='Path to Model for Testing', required=False)
@click.option('-folder', '--folder', help='Path to Evaluation Folder', required=False)
def train_synth(mode, model=None, folder=None):

	"""
	Training, Synthesizing, Testing using strong supervision on Synth-Text dataset
	:param mode: 'train', 'test', 'synthesize'
	:param model: Path to Model for Testing (only required if mode = 'test', 'synthesize'
	:param folder: Path to folder to synthesize
	:return: None
	"""

	mode = mode.lower()

	if mode == 'train':
		from train_synth import train
		train.main()

	elif mode == 'test':
		from train_synth import test
		if model is None:
			print('Please Enter the model path')
		else:
			test.main(model)

	elif mode == 'synthesize':

		from train_synth import synthesize

		if model is None:
			print('Please Enter the model path')

		elif folder is None:
			print('Please Enter the path of the folder you want to generate the targets for')

		else:
			print('Will generate the predictions at: ', '/'.join(folder.split('/')[:-1])+'/target_affinity')
			print('Will generate the predictions at: ', '/'.join(folder.split('/')[:-1])+'/target_character')
			print('Will generate the predictions at: ', '/'.join(folder.split('/')[:-1]) + '/word_bbox')

			os.makedirs('/'.join(folder.split('/')[:-1])+'/target_affinity', exist_ok=True)
			os.makedirs('/'.join(folder.split('/')[:-1])+'/target_character', exist_ok=True)
			os.makedirs('/'.join(folder.split('/')[:-1])+'/word_bbox', exist_ok=True)

			synthesize.main(
				folder,
				model_path=model,
				base_path_character='/'.join(folder.split('/')[:-1])+'/target_character',
				base_path_affinity='/'.join(folder.split('/')[:-1])+'/target_affinity',
				base_path_bbox='/'.join(folder.split('/')[:-1])+'/word_bbox',)

	else:
		print('Invalid Mode')


@main.command()
@click.option('-model', '--model', help='Path to Model trained on SYNTH', required=True)
@click.option('-iter', '--iterations', help='Number of Iterations to do', required=True)
def weak_supervision(model, iterations):

	"""
	Training weak supervision on icdar 2013 dataset
	:param model: Path to Pre-trained model on Synth-Text using the function train_synth
	:param iterations: Number of iterations to train on icdar 2013
	:return: None
	"""

	from train_weak_supervision.__init__ import get_initial_model_optimizer, generate_target, train, save_model, test
	from train_weak_supervision import config

	seed(config)

	# ToDo - Check the effects of using optimizer of Synth-Text or starting from a random optimizer

	model, optimizer = get_initial_model_optimizer(model)

	"""
	Steps - 
		1) Using the pre-trained model generate the targets
		2) Fine-tune the model on icdar 2013 dataset using weak-supervision
		3) Saving the model and again repeating process 1-3
		4) Saving the final model	
	"""

	for iteration in range(int(iterations)):

		print('Generating for iteration:', iteration)
		generate_target(model, iteration)

		# ToDo - Check the effects of using a new optimizer after every iteration or using the previous iteration optimizer

		print('Testing for iteration:', iteration)
		f_score_test = test(model)
		print('Test Results for iteration:', iteration, ' | F-score: ', f_score_test)

		print('Fine-tuning for iteration:', iteration)
		model, optimizer = train(model, optimizer, iteration)

		print('Saving for iteration:', iteration)
		save_model(model, optimizer, 'intermediate', iteration)

	save_model(model, optimizer, 'final')


if __name__ == "__main__":

	main()
