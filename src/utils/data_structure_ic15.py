import os
import numpy as np
import json


def icdar2015_test(base_path='Challenge4_Test_Task1_GT', output_path='Images/test_gt.json'):
	"""
	This function converts the icdar 2013 challenge 2 images to the format which we need
	to train our weak supervision model.
	More data set conversion functions would be written here
	:param base_path: Put your path to ground truth folder for icdar 2015 data set
	:param output_path: Will convert the ground truth to a json format at the location output_path
	:return: None
	"""

	all_transcriptions = os.listdir(base_path)

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

	for i in all_transcriptions:

		image_name = i[3:].split('.')[0] + '.jpg'
		all_annots['annots'][image_name] = {}

		cur_annot = []
		cur_text = []

		with open(base_path + '/' + i, 'r') as f:
			for no, f_i in enumerate(f):
				annots, text = f_i[:-1].split(',')[0:8], ','.join(f_i[:-1].split(',')[8:])
				cur_annot.append(np.array(annots, dtype=np.int32).reshape([4, 2]).tolist())
				cur_text.append(text)

		all_annots['annots'][image_name]['bbox'] = cur_annot
		all_annots['annots'][image_name]['text'] = cur_text

	with open(output_path, 'w') as f:
		json.dump(all_annots, f)


def icdar2015_train(base_path='ch4_training_localization_transcription_gt', output_path='Images/train_gt.json'):
	"""
	This function converts the icdar 2013 challenge 2 images to the format which we need
	to train our weak supervision model.
	More data set conversion functions would be written here
	:param base_path: Put your path to ground truth folder for icdar 2015 data set
	:param output_path: Will convert the ground truth to a json format at the location output_path
	:return: None
	"""

	all_transcriptions = os.listdir(base_path)

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

	for i in all_transcriptions:

		image_name = i[3:].split('.')[0] + '.jpg'
		all_annots['annots'][image_name] = {}

		cur_annot = []
		cur_text = []

		with open(base_path + '/' + i, 'r') as f:
			for no, f_i in enumerate(f):
				annots, text = f_i[:-1].split(',')[0:8], ','.join(f_i[:-1].split(',')[8:])

				if no == 0:
					annots[0] = annots[0][1:]
				cur_annot.append(np.array(annots, dtype=np.int32).reshape([4, 2]).tolist())
				cur_text.append(text)

		all_annots['annots'][image_name]['bbox'] = cur_annot
		all_annots['annots'][image_name]['text'] = cur_text

	with open(output_path, 'w') as f:
		json.dump(all_annots, f)
