"""
Config containing all hardcoded parameters for training weak supervised model on datasets like icdar 2013
"""
from config import *

seed = 0

use_cuda = True
num_cuda = "0,1"

prob_synth = 1/6

check_path = 'Temporary/'

batch_size = {
	'train': 4*len(num_cuda.split(',')),
	'test': 8*len(num_cuda.split(',')),
}

num_workers = {
	'train': 16,
	'test': 16
}


lr = {
	0: 1e-4,
	1: 5e-5,
	2: 5e-5,
	3: 1e-5,
	4: 1e-5,
	5: 1e-5,
	6: 5e-6,
	7: 5e-6,
	8: 5e-6,
	9: 5e-6,
	10: 1e-6,
	11: 1e-6,
	12: 1e-6,
	13: 1e-6,
	14: 5e-7,
	15: 5e-7,
	16: 5e-7,
	17: 5e-7,
	18: 1e-7,
	19: 1e-7,
}

optimizer_iterations = 4//len(num_cuda.split(','))
iterations = batch_size['train']*12500*optimizer_iterations
check_iterations = 2500*optimizer_iterations
calc_f_score = 50*optimizer_iterations
change_lr = 500*optimizer_iterations
test_now = 500*optimizer_iterations

model_architecture = 'craft'

data_augmentation = {
	'crop_size': [[1024, 1024], [768, 768], [512, 512]],
	'crop_size_prob': [0.7, 0.2, 0.1],
}
