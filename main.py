import click
import os

@click.group()
def main():
	pass


@main.command()
@click.option('-mode', '--mode', help='Train or Test or Synthesize', required=True)
@click.option('-model', '--model', help='Path to Model for Testing', required=False)
@click.option('-folder', '--folder', help='Path to Evaluation Folder', required=False)
def train_synth(mode, model=None, folder=None):

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
			# Check if this test function works
	elif mode == 'synthesize':
		from train_synth import synthesize
		if model is None:
			print('Please Enter the model path')
		elif folder is None:
			print('Please Enter the path of the folder you want to generate the targets for')
		else:
			print('Will generate the predictions at: ', '/'.join(folder.split('/')[:-1])+'/target_affinity')
			print('Will generate the predictions at: ', '/'.join(folder.split('/')[:-1])+'/target_character')

			os.makedirs('/'.join(folder.split('/')[:-1])+'/target_affinity', exist_ok=True)
			os.makedirs('/'.join(folder.split('/')[:-1])+'/target_character', exist_ok=True)

			synthesize.main(
				folder,
				model_path=model,
				base_path_character='/'.join(folder.split('/')[:-1])+'/target_character',
				base_path_affinity='/'.join(folder.split('/')[:-1])+'/target_affinity')
	else:
		print('Invalid Mode')


@main.command()
@click.option('-model', '--model', help='Path to Model trained on SYNTH', required=True)
@click.option('-iter', '--iterations', help='Number of Iterations to do', required=True)
def weak_supervision(model, iter):

	from train_weak_supervision.__init__ import get_initial_model, generate_target, train, save_model

	model, optimizer = get_initial_model_optimizer(model)

	for iteration in range(int(iter)):

		generate_target(model, iteration)
		model, optimizer = train(model, optimizer, iteration)
		save_model(model, optimizer, 'intermediate', iteration)

	save_model(model, optimizer, 'final')


if __name__ == "__main__":

	main()