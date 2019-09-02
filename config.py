seed = 0

dataset_name = 'ICDAR2013_ICDAR2017'
test_dataset_name = 'ICDAR2013'

DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

DataLoader_Other_Synthesis = '/home/SharedData/Mayank/'+dataset_name+'/Save/'
Other_Dataset_Path = '/home/SharedData/Mayank/'+dataset_name
save_path = '/home/SharedData/Mayank/Models/WeakSupervision/'+dataset_name
images_path = '/home/SharedData/Mayank/'+dataset_name+'/Images'
target_path = '/home/SharedData/Mayank/'+dataset_name+'/Generated'

Test_Dataset_Path = '/home/SharedData/Mayank/'+test_dataset_name

threshold_character = 0.5
threshold_affinity = 0.5
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
