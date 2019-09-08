seed = 0

dataset_name = 'ICDAR2015'
test_dataset_name = 'ICDAR2015'

DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

DataLoader_Other_Synthesis = '/home/SharedData/Mayank/'+dataset_name+'/Save/'
Other_Dataset_Path = '/home/SharedData/Mayank/'+dataset_name
save_path = '/home/SharedData/Mayank/Models/WeakSupervision/'+dataset_name
images_path = '/home/SharedData/Mayank/'+dataset_name+'/Images'
target_path = '/home/SharedData/Mayank/'+dataset_name+'/Generated'

Test_Dataset_Path = '/home/SharedData/Mayank/'+test_dataset_name

# threshold_character = 0.401288268
# threshold_affinity = 0.457833362

threshold_character = 0.4
threshold_affinity = 0.4

# threshold_character_upper = threshold_character + 0.2
# threshold_affinity_upper = threshold_affinity + 0.2

threshold_character_upper = 0.4
threshold_affinity_upper = 0.4

scale_character = 25/18.67
scale_affinity = 25/18.3

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
