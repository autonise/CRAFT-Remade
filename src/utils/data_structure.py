import os
import numpy as np
import json


def ICDAR2013():

		base_path = '/home/SharedData/Mayank/ICDAR2013/ch2_training_localization_transcription_gt'
		output_path = '/home/SharedData/Mayank/ICDAR2013/gt.json'

		all_transcriptions = os.listdir(base_path)

		unknown = '###'

		all_annots = {'unknown': unknown, 'annots': {}}

		for i in all_transcriptions:

			image_name = i[3:].split('.')[0]+'.jpg'
			all_annots['annots'][image_name] = {}

			cur_annot = []
			cur_text = []

			with open(base_path+'/'+i, 'r') as f:
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

ICDAR2013()