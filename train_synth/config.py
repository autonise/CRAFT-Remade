"""
Config containing all hardcoded parameters for training strong supervised model on synth-text
"""
from config import *

num_cuda = "0,1,2,3"
save_path = '/home/SharedData/Mayank/Models/SYNTH'
use_cuda = True

batch_size = {
	'train': 4*len(num_cuda.split(',')),
	'test': 8*len(num_cuda.split(',')),
}

num_workers = {
	'train': 0,
	'test': 16
}

pretrained = True
pretrained_path = '/home/SharedData/Mayank/Models/SYNTH/11999_model.pkl'
pretrained_loss_plot_training = '/home/SharedData/Mayank/Models/SYNTH/loss_plot_training.npy'

optimizer_iteration = 4//len(num_cuda.split(','))

lr = {
	1: 1e-4,
	8000*optimizer_iteration: 5e-5,
	12000*optimizer_iteration: 1e-5,
	16000*optimizer_iteration: 5e-6,
	24000*optimizer_iteration: 1e-6,
	36000*optimizer_iteration: 5e-7,
}

num_epochs_strong_supervision = 1.5

periodic_fscore = 300*optimizer_iteration
periodic_output = 3000*optimizer_iteration
periodic_save = 3000*optimizer_iteration

visualize_generated = True
visualize_freq = 21000

weight_threshold = 0.5

model_architecture = 'craft'
