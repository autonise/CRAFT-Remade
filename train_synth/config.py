"""
Config containing all hardcoded parameters for training strong supervised model on synth-text
"""

num_cuda = "0"
seed = 0
save_path = '/home/SharedData/Mayank/Models/SYNTH'
use_cuda = True

batch_size = {
	'train': 3,
	'test': 3,
}

DEBUG = False
if DEBUG:
	batch_size['train'] = 1
	num_cuda = '0'

pretrained = False
pretrained_path = ''
pretrained_loss_plot_training = 'model/loss_plot_training.npy'

lr = {
	1: 5e-5,
	10000: 2.5e-5,
	20000: 1e-5,
	40000: 5e-6,
	60000: 1e-6,
}

periodic_fscore = 100
periodic_output = 1000
periodic_save = 10000

threshold_character = 0.4
threshold_affinity = 0.4
threshold_fscore = 0.5

DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

ICDAR2013_path = '/home/SharedData/Mayank/ICDAR2013'
