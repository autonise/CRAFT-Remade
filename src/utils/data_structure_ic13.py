import os
import numpy as np
import json


def icdar2013_test(
		image_path='Challenge2_Test_Task12_Images',
		base_path='Challenge2_Test_Task1_GT',
		output_path='test_gt.json'):

	"""
	This function converts the icdar 2013 challenge 2 images to the format which we need
	to train our weak supervision model.
	:param image_path: Put the path to the test images
	:param base_path: Put your path to ground truth folder for icdar 2013 data set
	:param output_path: Will convert the ground truth to a json format at the location output_path
	:return: None
	"""

	all_transcriptions = os.listdir(image_path)

	# Sometimes the bbox is marked but the text is not marked.
	# This is the symbol(unknown = '###') that was used in icdar 2013 to denote that text annotation is missing
	unknown = '###'

	""" The json format for storing the dataset is 

			all_annots = {
				'unknown' : '###',  # String which denotes the unannotated bbox
				'annots' : {
					'image_name_1' : {
						'bbox' : [word_bbox_1, word_bbox_2, ...],  # word_bbox is a list of shape [4, 2]
						'test' : [text_string_1, text_string_2, ...],  # text_string is the string corresponding to the bbox
					}
				}  
			}
	"""

	# These are pre-processing steps specific to icdar 2013 challenge 2

	all_annots = {'unknown': unknown, 'annots': {}}

	for image_name in all_transcriptions:

		i = 'gt_'+'.'.join(image_name.split('.')[:-1])+'.txt'
		all_annots['annots'][image_name] = {}

		cur_annot = []
		cur_text = []

		with open(base_path + '/' + i, 'r') as f:
			for no, f_i in enumerate(f):
				x, text = f_i[:-1].split(',')[0:4], ','.join(f_i[:-1].split(',')[4:])[2:-1]
				annots = [[x[0], x[1]], [x[0], x[3]], [x[2], x[3]], [x[2], x[1]]]
				cur_annot.append(np.array(annots, dtype=np.int32).reshape([4, 2]).tolist())
				cur_text.append(text)

		all_annots['annots'][image_name]['bbox'] = cur_annot
		all_annots['annots'][image_name]['text'] = cur_text

	with open(output_path, 'w') as f:
		json.dump(all_annots, f)


def icdar2013_train(
		image_path='ch2_training_images',
		base_path='ch2_training_localization_transcription_gt',
		output_path='train_gt.json'):
	"""
	This function converts the icdar 2013 challenge 2 images to the format which we need
	to train our weak supervision model.
	More data set conversion functions would be written here
	:param image_path: Put the path to the test images
	:param base_path: Put your path to ground truth folder for icdar 2013 data set
	:param output_path: Will convert the ground truth to a json format at the location output_path
	:return: None
	"""

	all_transcriptions = os.listdir(image_path)

	# Sometimes the bbox is marked but the text is not marked.
	# This is the symbol(unknown = '###') that was used in icdar 2013 to denote that text annotation is missing
	unknown = '###'

	""" The json format for storing the dataset is 

			all_annots = {
				'unknown' : '###',  # String which denotes the unannotated bbox
				'annots' : {
					'image_name_1' : {
						'bbox' : [word_bbox_1, word_bbox_2, ...],  # word_bbox is a list of shape [4, 2]
						'test' : [text_string_1, text_string_2, ...],  # text_string is the string corresponding to the bbox
					}
				}  
			}
	"""

	# These are pre-processing steps specific to icdar 2013 challenge 2

	all_annots = {'unknown': unknown, 'annots': {}}

	for image_name in all_transcriptions:

		i = 'gt_'+'.'.join(image_name.split('.')[:-1])+'.txt'
		all_annots['annots'][image_name] = {}

		cur_annot = []
		cur_text = []

		with open(base_path + '/' + i, 'r') as f:
			lines = f.readlines()
			for no,line in enumerate(lines):
				line = line.strip().split(",")
				annots = line[:8]
				text = ','.join(line[8:])
				if no == 0:
					annots[0] = annots[0][1:]
				cur_annot.append(np.array(annots, dtype=np.int32).reshape([4, 2]).tolist())
				cur_text.append(text)
			all_annots['annots'][image_name]['bbox'] = cur_annot
			all_annots['annots'][image_name]['text'] = cur_text

	with open(output_path, 'w') as f:
		json.dump(all_annots, f)


if __name__ == "__main__":

	icdar2013_test()
	icdar2013_train()
