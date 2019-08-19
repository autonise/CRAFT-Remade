"""
Config containing all hardcoded parameters for training weak supervised model on datasets like icdar 2013
"""

seed = 0

use_cuda = True
num_cuda = "0"

save_path = '/home/SharedData/Mayank/Models/WeakSupervision/ICDAR2015'

images_path = '/home/SharedData/Mayank/ICDAR2015/Images'
target_path = '/home/SharedData/Mayank/ICDAR2015/Generated'

prob_synth = 1/6

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
	0: 1e-4,
	1: 1e-4,
	2: 1e-4,
	3: 1e-4,
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

threshold_character = 0.4
threshold_affinity = 0.4
threshold_fscore = 0.5

iterations = batch_size['train']*500
