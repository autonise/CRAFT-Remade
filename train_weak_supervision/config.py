"""
Config containing all hardcoded parameters for training weak supervised model on datasets like icdar 2013
"""
from config import *

seed = 0

use_cuda = True
num_cuda = "0,1,2,3"

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
	0: 1e-6,
	1: 1e-6,
	2: 5e-7,
	3: 5e-7,
	4: 5e-7,
	5: 5e-7,
	6: 1e-7,
	7: 1e-7,
	8: 1e-7,
	9: 1e-7,
}

optimizer_iterations = 1
iterations = batch_size['train']*2500*optimizer_iterations
check_iterations = 1249
calc_f_score = 100

model_architecture = 'craft'

data_augmentation = {
	'crop_size': [[1024, 1024], [768, 768], [512, 512]],
	'crop_size_prob': [0.7, 0.2, 0.1],
}
