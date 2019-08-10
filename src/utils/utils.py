from shapely.geometry import Polygon
import numpy as np
import train_synth.config as config
# ToDO pass the config everytime when it is required. Don't make the utils folder specific to some config
import cv2
import networkx as nx


def weighing_function(orig_length, cur_length):

	"""

	:param orig_length: Length of the expected word bounding box
	:param cur_length: Length of the predicted word bounding box
	:return:
	"""

	return (orig_length - min(orig_length, abs(orig_length - cur_length)))/orig_length


def get_weighted_character_target(generated_targets, original_annotation, unkown_symbol):

	"""

	:param generated_targets: {
			'word_bbox': np.array(join(all_characters, all_joins)),
			'characters': np.copy(all_characters),
		}
	:param original_annotation: {'bbox': [[4, 2]], 'text': []}
	:param unkown_symbol: The symbol(string) which denotes that the text was not annotated
	:return:
	"""

	aligned_generated_targets = {
		'word_bbox': original_annotation['bbox'],
		'characters': [[] for i in original_annotation['text']],
		'text': original_annotation['text'],
		'weights': [0 for i in original_annotation['text']]
	}

	for orig_no, orig_annot in enumerate(original_annotation['bbox']):

		found_no = -1

		for no, gen_t in enumerate(generated_targets['word_bbox']):

			if calc_iou(np.array(gen_t), np.array(orig_annot)) > 0.5:
				found_no = no
				break

		if found_no == -1:
			# ToDo - Write the code for breaking the word_bbox in equal parts and
			#  appending to the aligned_generated_targets['characters']

			aligned_generated_targets['weights'][orig_no] = 0.5
		elif original_annotation['text'][found_no] == unkown_symbol:

			aligned_generated_targets['weights'][orig_no] = 0.5
			for i in generated_targets['characters'][found_no]:
				aligned_generated_targets['characters'][orig_no].append(i)
		else:
			aligned_generated_targets['weights'][orig_no] = weighing_function(len(original_annotation['text'][found_no]), len(generated_targets['characters'][no]))
			for i in generated_targets['characters'][found_no]:
				aligned_generated_targets['characters'][orig_no].append(i)

	return aligned_generated_targets


def remove_small_predictions(image):

	"""

	:param image: Predicted character or affinity heat map in uint8
	:return:
	"""

	kernel = np.ones((5, 5), np.uint8)
	image = cv2.erode(image, kernel, iterations=2)
	image = cv2.dilate(image, kernel, iterations=3)

	return image


def generate_bbox(weight, weight_affinity, character_threshold=config.threshold_character, affinity_threshold=config.threshold_affinity):

	"""

	:param weight: Character Heatmap (Range between 0 and 1)
	:param weight_affinity: Affinity Heatmap (Range between 0 and 1)
	:param character_threshold: Threshold above which we say pixel belongs to a character
	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
	:return:
	"""

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

	weight = remove_small_predictions(weight)
	weight_affinity = remove_small_predictions(weight_affinity)

	all_characters, hierarchy = cv2.findContours(weight, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	all_joins, hierarchy = cv2.findContours(weight_affinity, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if len(all_characters) > 1000:

		return {
			'error_message': 'Number of characters too high'
		}

	if len(all_joins) > 1000:
		return {
			'error_message': 'Number of characters too high'
		}

	for ii in range(len(all_characters)):
		rect = cv2.minAreaRect(all_characters[ii])
		all_characters[ii] = cv2.boxPoints(rect)

	for ii in range(len(all_joins)):
		rect = cv2.minAreaRect(all_joins[ii])
		all_joins[ii] = cv2.boxPoints(rect)

	all_word_bbox, all_characters_bbox = join_with_characters(all_characters, all_joins)
	word_bbox = [i.tolist() for i in np.array(all_word_bbox)]
	char_bbox = [i.tolist() for i in np.array(all_characters_bbox)]

	return {
		'word_bbox': word_bbox,
		'characters': char_bbox
	}


def order_points(pts):

	"""
	Orders the 4 co-ordinates of a bounding box
	:param pts: numpy array with shape [4, 2]
	:return:
	"""

	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def calc_iou(poly1, poly2):

	"""

	:param poly1: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:param poly2: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:return:
	"""

	a = Polygon(poly1.reshape([poly1.shape[0], 2]))
	b = Polygon(poly2.reshape([poly2.shape[0], 2]))

	union_area = a.union(b).area

	if union_area == 0:
		return 0

	return a.intersection(b).area/union_area


def calculate_fscore(pred, target, text_pred=None, text_target=None, threshold=0.5):

	"""

	:param pred: numpy array with shape [num_words, 4, 2]
	:param target: numpy array with shape [num_words, 4, 2]
	:param text_pred: predicted text (Not useful in CRAFT implementation)
	:param text_target: target text (Not useful in CRAFT implementation)
	:param threshold: overlap iou threshold over which we say the pair is positive
	:return:
	"""

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

	"""

	:param pred: list of numpy array having shape [num_words, 4, 2]
	:param target: list of numpy array having shape [num_words, 4, 2]
	:param text_pred: list of predicted text
	:param text_target: list of target text
	:param threshold: threshold value for iou above which we say a pair of bbox are positive
	:return:
	"""

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
	:param word_contours: Contours to be joined to get one word in order(The contours are consecutive)
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

	"""

	:param characters: list of character bbox
	:param joints: list of join(affinity) bbox
	:return:
	"""

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


def join_with_characters(characters, joints):

	"""

	:param characters: list of character bbox
	:param joints: list of join(affinity) bbox
	:return:
	"""

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
	all_character_contours = []

	for word_idx in all_words:
		if len(all_joined[np.array(list(word_idx))[list(np.where(np.array(list(word_idx)) < len(characters)))[0]]]) == 0:
			continue

		all_word_contours.append(get_smooth_polygon(all_joined[list(word_idx)]))
		all_character_contours.append(all_joined[np.array(list(word_idx))[list(np.where(np.array(list(word_idx)) < len(characters)))[0]]])

	return all_word_contours, all_character_contours


def get_word_poly(weight, weight_affinity, character_threshold=config.threshold_character, affinity_threshold=config.threshold_affinity):

	"""

	:param weight: heatmap of characters
	:param weight_affinity: heatmap of affinity
	:param character_threshold: threshold above which we say a pixel is character
	:param affinity_threshold: threshold above which we say a pixel is affinity
	:return:
	"""

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
