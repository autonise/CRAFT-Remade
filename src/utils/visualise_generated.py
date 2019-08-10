import sys
import os
import json
import cv2
import numpy as np

base_path = sys.argv[1]
iteration = int(sys.argv[2])

imnames = sorted(os.listdir(base_path+'/Images'))

for im in imnames:
	image = cv2.imread(base_path+'/Images/'+im)
	with open(base_path+'/Generated/'+str(iteration)+'/'+'.'.join(im.split('.')[:-1])+'.json') as f:
		annotation = json.load(f)

	cv2.drawContours(image, [np.array(i) for i in annotation['word_bbox']], -1, (255, 255, 255), 3)
	cv2.imwrite(im, image)
