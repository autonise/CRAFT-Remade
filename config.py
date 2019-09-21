import math

seed = 0

THRESHOLD_POSITIVE = 0.1
THRESHOLD_NEGATIVE = 0

threshold_point = 25
window = 120

sigma = 18.5  # 0.401288268
sigma_aff = 20  # 0.457833362

# sigma = 15.25  # 0.401288268
# sigma_aff = 13.3  # 0.457833362

boundary_character = math.exp(-1/2*(threshold_point**2)/(sigma**2))
boundary_affinity = math.exp(-1/2*(threshold_point**2)/(sigma_aff**2))

threshold_character = boundary_character + 0.03
threshold_affinity = boundary_affinity + 0.03

threshold_character_upper = boundary_character + 0.1
threshold_affinity_upper = boundary_affinity + 0.1

scale_character = math.sqrt(math.log(boundary_character)/math.log(threshold_character_upper))
scale_affinity = math.sqrt(math.log(boundary_affinity)/math.log(threshold_affinity_upper))

dataset_name = 'ICDAR2013_ICDAR2017'
test_dataset_name = 'ICDAR2013'

print(
	'Boundary character value = ', boundary_character,
	'| Threshold character value = ', threshold_character,
	'| Threshold character upper value = ', threshold_character_upper
)
print(
	'Boundary affinity value = ', boundary_affinity,
	'| Threshold affinity value = ', threshold_affinity,
	'| Threshold affinity upper value = ', threshold_affinity_upper
)
print('Scale character value = ', scale_character, '| Scale affinity value = ', scale_affinity)
print('Sigma Character:', sigma, 'Sigma Affinity:', sigma_aff)
print('Training Dataset = ', dataset_name, '| Testing Dataset = ', test_dataset_name)

DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

DataLoader_Other_Synthesis = '/home/SharedData/Mayank/'+dataset_name+'/Save/'
Other_Dataset_Path = '/home/SharedData/Mayank/'+dataset_name
save_path = '/home/SharedData/Mayank/Models/WeakSupervision/'+dataset_name
images_path = '/home/SharedData/Mayank/'+dataset_name+'/Images'
target_path = '/home/SharedData/Mayank/'+dataset_name+'/Generated'

Test_Dataset_Path = '/home/SharedData/Mayank/'+test_dataset_name

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

start_iteration = 0
skip_iterations = []
horizontal_rectangle = True
