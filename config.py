seed = 0

DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

DataLoader_Other_Synthesis = '/home/SharedData/Mayank/ICDAR2015/Save/'

Other_Dataset_Path = '/home/SharedData/Mayank/ICDAR2015'

save_path = '/home/SharedData/Mayank/Models/WeakSupervision/ICDAR2015'

images_path = '/home/SharedData/Mayank/ICDAR2015/Images'
target_path = '/home/SharedData/Mayank/ICDAR2015/Generated'

threshold_character = 0.4
threshold_affinity = 0.4
threshold_word = 0.7
threshold_fscore = 0.5

dataset_pre_process = {
	'ic13': {
		'train': {
			'target_json_path': None,
			'target_folder_path': None,
		},
		'test': {
			'target_json_path': None,
			'target_folder_path': None,
		}
	},
	'ic15': {
		'train': {
			'target_json_path': None,
			'target_folder_path': None,
		},
		'test': {
			'target_json_path': None,
			'target_folder_path': None,
		}
	}
}