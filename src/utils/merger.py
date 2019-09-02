import json
import os
import shutil


def merge(datasets):

	base_path = '_'.join(datasets)
	os.makedirs(base_path, exist_ok=True)
	os.makedirs(base_path+'/Images/train', exist_ok=True)
	os.makedirs(base_path+'/Images/test', exist_ok=True)

	def merge_image():

		for dataset in datasets:

			images_train = os.listdir(dataset + '/Images/train')

			for img in images_train:
				shutil.copyfile(dataset + '/Images/train/' + img, base_path + '/Images/train/' + dataset + '_' + img)

			images_test = os.listdir(dataset + '/Images/test')

			for img in images_test:
				shutil.copyfile(dataset + '/Images/test/' + img, base_path + '/Images/test/' + dataset + '_' + img)

	def merge_gt():

		common_unknown = '###'

		combined_gt_train = {"unknown": common_unknown, "annots": {}}
		combined_gt_test = {"unknown": common_unknown, "annots": {}}

		for dataset in datasets:

			with open(dataset + '/Images/train_gt.json', 'r') as f:
				gt_train = json.load(f)

			dataset_unknown = gt_train['unknown']

			for img in gt_train["annots"].keys():
				cleaned_text = []
				for cur_text in gt_train['annots'][img]['text']:

					if cur_text == dataset_unknown:
						cleaned_text.append(common_unknown)
					else:
						cleaned_text.append(cur_text)

				combined_gt_train["annots"][dataset + '_' + img] = {
					'bbox': gt_train["annots"][img]['bbox'], 'text': cleaned_text,
					'dataset': dataset}

			with open(base_path + '/Images/train_gt.json', 'w') as f:
				json.dump(combined_gt_train, f)

			with open(dataset + '/Images/test_gt.json', 'r') as f:
				gt_test = json.load(f)

			dataset_unknown = gt_test['unknown']

			for img in gt_test["annots"].keys():
				cleaned_text = []
				for cur_text in gt_test['annots'][img]['text']:

					if cur_text == dataset_unknown:
						cleaned_text.append(common_unknown)
					else:
						cleaned_text.append(cur_text)

				combined_gt_test["annots"][dataset + '_' + img] = {'bbox': gt_test["annots"][img]['bbox'], 'text': cleaned_text, 'dataset': dataset}

			with open(base_path + '/Images/test_gt.json', 'w') as f:
				json.dump(combined_gt_test, f)

	merge_image()
	merge_gt()


if __name__ == "__main__":

	dataset_paths = ['ICDAR2013', 'ICDAR2017']

	merge(dataset_paths)
