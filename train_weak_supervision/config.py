"""
Config containing all hardcoded parameters for training weak supervised model on datasets like icdar 2013
"""

seed = 0

use_cuda = True
num_cuda = "0"

save_path = '/home/SharedData/Mayank/Models/WeakSupervision/ICDAR2015'

images_path = '/home/SharedData/Mayank/ICDAR2015/Images'
target_path = '/home/SharedData/Mayank/ICDAR2015/Generated'

prob_synth = 0

DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

DataLoaderICDAR2013_Synthesis = '/home/SharedData/Mayank/ICDAR2015/Save/'

ICDAR2013_path = '/home/SharedData/Mayank/ICDAR2015'

batch_size = {
	'train': 3,
	'test': 3,
}

lr = {
	0: 1e-5,
	1: 1e-5,
	2: 1e-5,
	3: 1e-5,
	4: 5e-6,
	5: 5e-6,
	6: 5e-6,
	7: 5e-6,
	8: 1e-6,
	9: 1e-6,
	10: 1e-6,
	11: 1e-6,
	12: 5e-7,
	13: 5e-7,
	14: 5e-7,
	15: 5e-7,
	16: 1e-7,
	17: 1e-7,
	18: 1e-7,
	19: 1e-7,
}

threshold_character = 0.4
threshold_affinity = 0.4
threshold_fscore = 0.5

iterations = batch_size['train']*500

model_architecture = 'craft'
