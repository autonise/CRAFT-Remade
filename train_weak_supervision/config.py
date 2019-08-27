"""
Config containing all hardcoded parameters for training weak supervised model on datasets like icdar 2013
"""
from config import *

seed = 0

use_cuda = True
num_cuda = "0,1,2,3"

prob_synth = 0

check_path = 'Temporary/'

batch_size = {
	'train': 4,
	'test': 3,
}

num_workers = {
	'train': 8,
	'test': 8
}


lr = {
	0: 1e-4,
	1: 1e-4,
	2: 1e-4,
	3: 1e-5,
	4: 5e-5,
	5: 5e-5,
	6: 5e-5,
	7: 5e-5,
	8: 1e-5,
	9: 1e-5,
	10: 1e-5,
	11: 1e-5,
	12: 5e-6,
	13: 5e-6,
	14: 5e-6,
	15: 5e-6,
	16: 1e-6,
	17: 1e-6,
	18: 1e-6,
	19: 1e-6,
}

optimizer_iterations = 4
iterations = batch_size['train']*2500*optimizer_iterations
check_iterations = 200

model_architecture = 'craft'

data_augmentation = {
	'crop_size': [[1024, 1024], [768, 768], [512, 512]],
	'crop_size_prob': [0.7, 0.2, 0.1],
}