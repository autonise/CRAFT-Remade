from shapely.geometry import Polygon
import numpy as np
import train_synth.config as config
# ToDO pass the config everytime when it is required. Don't make the utils folder specific to some config
import cv2
import networkx as nx

def generate_bbox(weight, weight_affinity, character_threshold=config.threshold_character, affinity_threshold=config.threshold_affinity):
	assert weight.max() <= 1, 'Weight has not been normalised'
	assert weight.min() >= 0, 'Weight has not been normalised'

	weight[weight > character_threshold] = 255
	weight[weight != 255] = 0

	assert weight_affinity.max() <= 1, 'Weight Affinity has not been normalised'
	assert weight_affinity.min() >= 0, 'Weight Affinity has not been normalised'

	weight_affinity[weight_affinity > affinity_threshold] = 255
	weight_affinity[weight_affinity != 255] = 0

	weight = weight.astype(np.uint8)
	weight_affinity = weight_affinity.astype(np.uint8)

	word_bbox = []
	characters = []
	joins = []
	

	for i in range(weight.shape[0]):

		all_characters, hierarchy = cv2.findContours(weight[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		all_joins, hierarchy = cv2.findContours(weight_affinity[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		characters.append(np.copy(all_characters))
		joins.append(np.copy(all_joins))

		if len(all_characters) > 1000:
			word_bbox.append(np.zeros([0]))
			continue
		if len(all_joins) > 1000:
			word_bbox.append(np.zeros([0]))
			continue

		for ii in range(len(all_characters)):
			rect = cv2.minAreaRect(all_characters[ii])
			all_characters[ii] = cv2.boxPoints(rect)

		for ii in range(len(all_joins)):
			rect = cv2.minAreaRect(all_joins[ii])
			all_joins[ii] = cv2.boxPoints(rect)


		word_bbox.append(np.array(join(all_characters, all_joins)))

	to_return={}
	to_return['word_bbox']=word_bbox
	to_return['characters']=characters
	to_return['joins']=joins
	return 

def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def calc_iou(poly1, poly2):

	a = Polygon(poly1.reshape([poly1.shape[0], 2]))
	b = Polygon(poly2.reshape([poly2.shape[0], 2]))

	union_area = a.union(b).area

	if union_area == 0:
		return 0

	return a.intersection(b).area/union_area


def calculate_fscore(pred, target, text_pred=None, text_target=None, threshold=0.5):

	if text_pred is None:
		check_text = False
	else:
		check_text = True

	already_done = np.zeros([len(target)], dtype=np.bool)

	false_positive = 0

	for no, i in enumerate(pred):

		found = False

		for j in range(len(target)):
			if already_done[j]:
				continue
			iou = calc_iou(i, target[j])
			if iou > threshold:
				if check_text:
					if text_pred[no] == text_target[j]:
						already_done[j] = True
						found = True
						break
				else:
					already_done[j] = True
					found = True
					break

		if not found:
			false_positive += 1

	true_positive = np.sum(already_done.astype(np.float32))

	if true_positive == 0:
		return 0

	precision = true_positive/(true_positive + false_positive)
	recall = true_positive/target.shape[0]

	return 2*precision*recall/(precision + recall)


def calculate_batch_fscore(pred, target, text_pred=None, text_target=None, threshold=0.5):

	f_score = 0
	for i in range(len(pred)):
		if text_pred is not None:
			f_score += calculate_fscore(pred[i], target[i], text_pred[i], text_target[i], threshold)
		else:
			f_score += calculate_fscore(pred[i], target[i], threshold=threshold)

	if len(pred) == 0 and len(target) == 0:
		return 1
	elif len(pred) == 0:
		return 0

	return f_score/len(pred)


def get_smooth_polygon(word_contours):

	"""
	:param word_contours:   Contours to be joined to get one word in order(The contours are consecutive)
	:return:
	"""

	all_word_contours = np.concatenate(word_contours, axis=0)
	convex_hull = cv2.convexHull(all_word_contours).astype(np.int32)
	if convex_hull.shape[0] == 1:
		convex_hull = np.repeat(convex_hull, 4, axis=0)
	if convex_hull.shape[0] == 2:
		convex_hull = np.repeat(convex_hull, 2, axis=0)
	return convex_hull  # Will have shape - number_of_points, 1, 2


def join(characters, joints):

	all_joined = characters + joints

	all_joined = np.array(all_joined).reshape([len(all_joined), 4, 2])

	graph = nx.Graph()
	graph.add_nodes_from(nx.path_graph(all_joined.shape[0]))

	for contour_i in range(all_joined.shape[0]-1):
		for contour_j in range(contour_i+1, all_joined.shape[0]):
			value = Polygon(all_joined[contour_i]).intersection(Polygon(all_joined[contour_j])).area
			if value > 0:
				graph.add_edge(contour_i, contour_j)

	all_words = nx.connected_components(graph)
	all_word_contours = []

	for word_idx in all_words:
		all_word_contours.append(get_smooth_polygon(all_joined[list(word_idx)]))

	return all_word_contours


def get_word_poly(weight, weight_affinity, character_threshold=config.threshold_character, affinity_threshold=config.threshold_affinity):

	assert weight.max() <= 1, 'Weight has not been normalised'
	assert weight.min() >= 0, 'Weight has not been normalised'

	weight[weight > character_threshold] = 255
	weight[weight != 255] = 0

	assert weight_affinity.max() <= 1, 'Weight Affinity has not been normalised'
	assert weight_affinity.min() >= 0, 'Weight Affinity has not been normalised'

	weight_affinity[weight_affinity > affinity_threshold] = 255
	weight_affinity[weight_affinity != 255] = 0

	weight = weight.astype(np.uint8)
	weight_affinity = weight_affinity.astype(np.uint8)

	word_bbox = []

	for i in range(weight.shape[0]):

		all_characters, hierarchy = cv2.findContours(weight[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		all_joins, hierarchy = cv2.findContours(weight_affinity[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if len(all_characters) > 1000:
			word_bbox.append(np.zeros([0]))
			continue
		if len(all_joins) > 1000:
			word_bbox.append(np.zeros([0]))
			continue

		for ii in range(len(all_characters)):
			rect = cv2.minAreaRect(all_characters[ii])
			all_characters[ii] = cv2.boxPoints(rect)

		for ii in range(len(all_joins)):
			rect = cv2.minAreaRect(all_joins[ii])
			all_joins[ii] = cv2.boxPoints(rect)

		word_bbox.append(np.array(join(all_characters, all_joins)))

	return word_bbox


if __name__ == "__main__":

	import matplotlib.pyplot as plt

	affinity = plt.imread('/home/Krishna.Wadhwani/Dataset/Programs/CRAFT-Remade/Stage-1/train_synthesis/0_1/0/target_affinity.png')[:, :, 0][None, :, :]
	character = plt.imread('/home/Krishna.Wadhwani/Dataset/Programs/CRAFT-Remade/Stage-1/train_synthesis/0_1/0/target_characters.png')[:, :, 0][None, :, :]

	image = plt.imread('/home/Krishna.Wadhwani/Dataset/Programs/CRAFT-Remade/Stage-1/train_synthesis/0_1/0/image.png')[:, :, 0:3]*255
	image = image.astype(np.uint8)

	word_bbox = get_word_poly(character, affinity)
	cv2.drawContours(image, word_bbox[0], -1, (0, 255, 0), 3)

	plt.imsave('out.png', image)
