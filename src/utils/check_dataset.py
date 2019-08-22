import json
import numpy as np
import cv2
import math

base_path = '/home/SharedData/Mayank/ICDAR2015/Images/train_gt.json'

with open(base_path, 'r') as f:
	gt = json.load(f)

unknown = gt['unknown']

unknown_bbox = []
known_bbox = []


def return_height_width(bbox):
	bbox = bbox.copy()
	bbox = np.array(bbox).reshape([len(bbox), 1, 2])
	bbox = cv2.minAreaRect(bbox)
	bbox = cv2.boxPoints(bbox)
	dist = [np.sqrt(np.sum(np.square(bbox[i] - bbox[(i+1) % 4]))) for i in range(4)]
	return np.min(dist), np.max(dist)


for image_i in gt['annots'].keys():
	for bbox_i in range(len(gt['annots'][image_i]['bbox'])):
		if gt['annots'][image_i]['text'][bbox_i] == unknown:
			unknown_bbox.append(gt['annots'][image_i]['bbox'][bbox_i])
		else:
			known_bbox.append(gt['annots'][image_i]['bbox'][bbox_i])

print('Unknown Bbox:', len(unknown_bbox), 'Known Bbox:', len(known_bbox))

unknown_dim = [[], []]
known_dim = [[], []]

for bbox_i in unknown_bbox:
	height, width = return_height_width(bbox_i)
	unknown_dim[0].append(height)
	unknown_dim[1].append(width)

for bbox_i in known_bbox:
	height, width = return_height_width(bbox_i)
	known_dim[0].append(height)
	known_dim[1].append(width)

print('Unknown Mean Height:', np.mean(unknown_dim[0]))
print('Unknown Mean Width:', np.mean(unknown_dim[1]))
print('Known Mean Height:', np.mean(known_dim[0]))
print('Known Mean Width:', np.mean(known_dim[1]))
