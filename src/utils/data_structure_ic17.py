import os
import json
import numpy as np

path_train = 'Images/train'
path_val = 'Images/test'

path_train_gt = 'ch8_training_localization_transcription_gt_v2'
path_val_gt = 'ch8_validation_localization_transcription_gt_v2'

path_out_images = 'Images'
path_out_gt = 'Images/train_gt.json'

unknown = '###'


def clean_annots(base_path, base_image_path):

	"""
			{
				unknown: '###',
				annots: {
							'img_1.png': {
											'word_bbox': [],
											'text': ['asdf', 'asdf', '###'],
											'language': ['latin', 'arabic']
							}
				}
			}
		"""


	annots = {
				'unknown': '###',
				'annots': {}
	}
		
	all_annots = os.listdir(base_path)

	all_image_path = os.listdir(base_image_path)

	for image_name in all_image_path:

		gt_name = 'gt_'+image_name[:-3]+'txt'

		with open(base_path+'/'+gt_name, 'r') as f:

			list_bbox = []
			list_language = []
			list_text = []

			for i in f:

				all_ = i[:-1].split(',')

				bbox = np.array(all_[0:8]).reshape([4, 2]).tolist()
				language = all_[8]
				text = all_[9]

				list_bbox.append(bbox)
				list_language.append(language)
				list_text.append(text)

			annots['annots'][image_name] = {
											'bbox': list_bbox,
											'text': list_text,
											'language': list_language
			}
					

	return annots


# clean_images()
with open('train_gt.json', 'w') as f:
	json.dump(clean_annots(path_train_gt, path_train), f)


with open('test_gt.json', 'w') as f:
	json.dump(clean_annots(path_val_gt, path_val), f)