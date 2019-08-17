"""
Config containing all hardcoded parameters for training weak supervised model on datasets like icdar 2013
"""

use_cuda = True
num_cuda = "0"

save_path = '/home/prantik/Documents/Datasets/Text-Detection/Models/WeakSupervision/ICDAR2013'

images_path = '/home/prantik/Documents/Datasets/Text-Detection/ICDAR2013/Images'
target_path = '/home/prantik/Documents/Datasets/Text-Detection/ICDAR2013/Generated'

prob_synth = 0.8

DataLoaderSYNTH_base_path = '/home/prantik/Documents/Datasets/Text-Detection/SynthText/Images'
DataLoaderSYNTH_mat = '/home/prantik/Documents/Datasets/Text-Detection/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/prantik/Documents/Datasets/Text-Detection/SynthText/train_synthesis/'

DataLoaderICDAR2013_Synthesis = '/home/prantik/Documents/Datasets/Text-Detection/ICDAR2013/Save/s'

ICDAR2013_path = '/home/prantik/Documents/Datasets/Text-Detection/ICDAR2013'

batch_size = {
	'train': 3,
	'test': 3,
}

lr = {
	1: 1e-5,
}

periodic_fscore = 10

threshold_character = 0.4
threshold_affinity = 0.4
threshold_fscore = 0.5

iterations = 1000
