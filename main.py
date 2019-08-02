import click


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
		model = model.loewr()
		from train_synth import test
		if model is None:
			print('Please Enter the model path')
		else:
			test.main(model)
			# Check if this test function works
	elif mode == 'synthesize':
		model = model.lower()
		from train_synth import synthesize
		if model is None:
			print('Please Enter the model path')
		elif folder is None:
			print('Please Enter the path of the folder you want to generate the targets for')
		else:
			print('Will generate the predictions at: ', '/'.join(folder.split('/')[:-1])+'/target')
			synthesize.main(model, folder)
			# Write this synthesize function
	else:
		print('Invalid Mode')


@main.command()
@click.option('-model', '--model', help='Path to Model trained on SYNTH', required=True)
@click.option('-iter', '--iterations', help='Number of Iterations to do', required=True)
def train_synth(model, iter):

	model = model.lower()
	iter = int(iter)

	# ToDo Write the Code

	return
