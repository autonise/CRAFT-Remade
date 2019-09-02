import click
import os
import torch
import numpy as np
import random


def seed(config=None):

	# This removes randomness, makes everything deterministic
	# ToDo - Make everything seeded, currently dataloader is random

	if config is None:
		import config

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


@click.group()
def main():
	seed()
	pass


@main.command()
def train_synth():

	"""
	Training using strong supervision on Synth-Text dataset
	:return: None
	"""

	from train_synth import train
	train.main()


@main.command()
@click.option('-model', '--model', help='Path to Model', required=True)
def test_synth(model):

	"""
	Training, Synthesizing, Testing using strong supervision on Synth-Text dataset
	:param model: Path to trained model
	:return: None
	"""

	from train_synth import test
	test.main(model)


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

	# ToDo - Check the effects of using optimizer of Synth-Text or starting from a random optimizer

	model, optimizer = get_initial_model_optimizer(model)

	print('Number of parameters in the model:', sum(p.numel() for p in model.parameters() if p.requires_grad))

	"""
	Steps - 
		1) Using the pre-trained model generate the targets
		2) Fine-tune the model on icdar 2013 dataset using weak-supervision
		3) Saving the model and again repeating process 1-3
		4) Saving the final model	
	"""

	skip_iterations = [0]

	for iteration in range(int(iterations)):

		if iteration not in skip_iterations:

			print('Generating for iteration:', iteration)
			generate_target(model, iteration)

			print('Testing for iteration:', iteration)
			f_score_test = test(model)
			print('Test Results for iteration:', iteration, ' | F-score: ', f_score_test)

		print('Fine-tuning for iteration:', iteration)
		model, optimizer, loss, accuracy = train(model, optimizer, iteration)

		print('Saving for iteration:', iteration)
		save_model(model, optimizer, 'intermediate', iteration, loss=loss, accuracy=accuracy)

	save_model(model, optimizer, 'final')


@main.command()
@click.option('-model', '--model', help='Path to Model trained on SYNTH', required=True)
@click.option('-folder', '--folder', help='Path to the image folder', required=True)
def synthesize(model, folder):

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
			base_path_bbox='/'.join(folder.split('/')[:-1])+'/word_bbox')


@main.command()
@click.option('-dataset', '--dataset', help='name of the dataset you want to pre-process(IC13, IC15)', required=True)
def pre_process(dataset):

	valid_choice = ['ic13', 'ic15']
	if dataset.lower() not in valid_choice:
		print('Invalid Dataset', dataset.lower(), ', currently available:', valid_choice)
		exit()

	if dataset.lower() == 'ic13':
		import config

		if \
			config.dataset_pre_process['ic13']['train']['target_json_path'] is None or \
			config.dataset_pre_process['ic13']['train']['target_folder_path'] is None or \
			config.dataset_pre_process['ic13']['test']['target_json_path'] is None or \
			config.dataset_pre_process['ic13']['test']['target_folder_path'] is None:
			print(
				'Change the config.py file. '
				'Add the path to the output json file and the target folder path. Detailed instructions in ReadMe.md')
		else:
			from src.utils.data_structure_ic13 import icdar2013_test, icdar2013_train

			icdar2013_test(
				config.dataset_pre_process['ic13']['test']['target_folder_path'],
				config.dataset_pre_process['ic13']['test']['target_json_path']
			)

			icdar2013_train(
				config.dataset_pre_process['ic13']['train']['target_folder_path'],
				config.dataset_pre_process['ic13']['train']['target_json_path']
			)
	elif dataset.lower() == 'ic15':
		import config

		if \
			config.dataset_pre_process['ic15']['train']['target_json_path'] is None or \
			config.dataset_pre_process['ic15']['train']['target_folder_path'] is None or \
			config.dataset_pre_process['ic15']['test']['target_json_path'] is None or \
			config.dataset_pre_process['ic15']['test']['target_folder_path'] is None:
			print(
				'Change the config.py file. '
				'Add the path to the output json file and the target folder path. Detailed instructions in ReadMe.md')
		else:
			from src.utils.data_structure_ic15 import icdar2015_test, icdar2015_train

			icdar2015_test(
				config.dataset_pre_process['ic15']['test']['target_folder_path'],
				config.dataset_pre_process['ic15']['test']['target_json_path']
			)

			icdar2015_train(
				config.dataset_pre_process['ic15']['train']['target_folder_path'],
				config.dataset_pre_process['ic15']['train']['target_json_path']
			)


if __name__ == "__main__":

	main()
