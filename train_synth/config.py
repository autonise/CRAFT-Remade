num_cuda = "0,1,2,3"
seed = 0
save_path = '/home/SharedData/Mayank/Models/SYNTH'
use_cuda = True

batchsize = {
	'train': 12,
	'test': 1
}

DEBUG = False
if DEBUG:
	batchsize['train'] = 1
	num_cuda = '0'

pretrained = False
pretrained_path = '/home/Krishna.Wadhwani/Dataset/Programs/CRAFT-Remade/Stage-1/model/41958_model.pkl'

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

threshold_character = 0.5
threshold_affinity = 0.5
threshold_fscore = 0.5
threshold_first_character = 0.5
threshold_boundary = 0.5
