from shapely.geometry import Polygon
import numpy as np
import cv2
import networkx as nx


def weighing_function(orig_length, cur_length):

	"""
	Function to generate the weight value given the predicted text-length and the expected text-length
	The intuition is that if the predicted text-length is far from the expected text-length then weight should be
	small to that word-bbox.

	:param orig_length: Length of the expected word bounding box
	:param cur_length: Length of the predicted word bounding box
	:return:
	"""

	return (orig_length - min(orig_length, abs(orig_length - cur_length)))/orig_length


def cutter(word_bbox, num_characters):

	"""
	Given a word_bbox of 4 co-ordinates and the number of characters inside it,
	generates equally spaced character bbox
	:param word_bbox: list of shape [4, 2] or a numpy array of shape [4, 2]
	:param num_characters: integer denoting number of characters inside the word_bbox
	:return: numpy array containing bbox of all the characters, dtype = np.float32, shape = [num_characters, 4, 2]
	"""

	if type(word_bbox) == list:
		word_bbox = np.array(word_bbox.copy())

	x = [np.sqrt(np.sum(np.square(word_bbox[i, :] - word_bbox[(i+1) % 4, :]))) for i in range(4)]

	width1_0 = np.argmax(x)
	width1_1 = (width1_0 + 1) % 4
	width2_0 = (width1_0 + 3) % 4

	width = x[width1_0]

	direction = (word_bbox[width1_1] - word_bbox[width1_0])/width
	character_width = width/num_characters

	char_bbox = np.zeros([num_characters, 4, 2])
	co_ordinates = np.zeros([num_characters + 1, 2, 2])

	co_ordinates[0, 0] = word_bbox[width1_0]
	co_ordinates[0, 1] = word_bbox[width2_0]

	for i in range(1, num_characters + 1):
		co_ordinates[i, 0] = co_ordinates[i - 1, 0] + direction*character_width
		co_ordinates[i, 1] = co_ordinates[i - 1, 1] + direction*character_width
		char_bbox[i-1, 0] = co_ordinates[i - 1, 0]
		char_bbox[i-1, 1] = co_ordinates[i, 0]
		char_bbox[i-1, 2] = co_ordinates[i, 1]
		char_bbox[i-1, 3] = co_ordinates[i - 1, 1]

	return char_bbox


def get_weighted_character_target(generated_targets, original_annotation, unknown_symbol, threshold):

	"""

	Function to generate targets using weak-supervision which will be used to fine-tune the model trained using
	Synthetic data.

	:param generated_targets: {
			'word_bbox': list of shape [num_words, 4, 1, 2] containing word-bbox of generated targets,
			'characters': list of shape [num_words, num_characters in each word, 4, 2], containing char-bbox of
																									generated targets,
		}
	:param original_annotation: {
			'bbox': list of shape [num_words, 4, 2] containing word-bbox of original annotations,
			'text': list of shape [num_words] containing text in original annotations
		}
	:param unknown_symbol: The symbol(string) which denotes that the text was not annotated
	:param threshold: overlap IOU value above which we consider prediction as positive
	:return: aligned_generated_targets: {
			'word_bbox': contains the word-bbox which have been annotated and present in original annotations,
			'characters': will contain the character-bbox generated using weak-supervision,
			'text' : list of all annotated text present in original_annotation,
			'weights' : list of values between 0 and 1 denoting weight of that particular word-bbox in the loss while
						training weak-supervised model
		}
	"""
	aligned_generated_targets = {
		'word_bbox': original_annotation['bbox'],
		'characters': [[] for _ in original_annotation['text']],
		'text': original_annotation['text'],
		'weights': [0 for _ in original_annotation['text']]
	}

	for orig_no, orig_annot in enumerate(original_annotation['bbox']):

		found_no = -1

		for no, gen_t in enumerate(generated_targets['word_bbox']):

			if calc_iou(np.array(gen_t), np.array(orig_annot)) > threshold:
				# If IOU between predicted and original word-bbox is > threshold then it is termed positive
				found_no = no
				break

		if found_no == -1:

			"""
				If the current original annotation was not predicted by the model then we create equi-spaced character
				bbox and give a weight of 0.5 to the word-bbox
			"""

			# ToDo - What should be done in this scenario?
			#  If not found and unknown_symbol is there this will create problems

			characters = cutter(orig_annot, len(original_annotation['text'][orig_no]))
			aligned_generated_targets['characters'][orig_no] = characters.astype(np.int32).tolist()
			aligned_generated_targets['weights'][orig_no] = 0.5

		elif original_annotation['text'][orig_no] == unknown_symbol:

			"""
				If the current original annotation was predicted by the model but the text-annotation is not present 
				then we create character bbox using predictions and give a weight of 0.5 to the word-bbox
			"""

			aligned_generated_targets['weights'][orig_no] = 0.5
			aligned_generated_targets['characters'][orig_no] = generated_targets['characters'][found_no]
		else:

			"""
				If the current original annotation was predicted by the model and the text-annotation is present 
				then we find the weight using the weighing_function and if it is less than 0.5 then create equi-spaced 
				character bbox otherwise use the generated char-bbox using predictions and give a weight of value of 
				weighing_function to the word-bbox
			"""

			weight = weighing_function(len(original_annotation['text'][orig_no]), len(generated_targets['characters'][found_no]))

			if weight < 0.5:
				characters = cutter(orig_annot, len(original_annotation['text'][orig_no]))
				aligned_generated_targets['characters'][orig_no] = characters.astype(np.int32).tolist()
				aligned_generated_targets['weights'][orig_no] = 0.5
			else:
				aligned_generated_targets['weights'][orig_no] = weight
				aligned_generated_targets['characters'][orig_no] = generated_targets['characters'][found_no]

	return aligned_generated_targets


def remove_small_predictions(image):

	"""
	This function is used to erode small stray character predictions but does not reduce the thickness of big predictions
	:param image: Predicted character or affinity heat map in uint8
	:return: image with less stray contours
	"""

	kernel = np.ones((5, 5), np.uint8)
	image = cv2.erode(image, kernel, iterations=2)
	image = cv2.dilate(image, kernel, iterations=3)

	return image


def generate_word_bbox(character_heatmap, affinity_heatmap, character_threshold, affinity_threshold):

	"""
	Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
	character bbox and word-bbox

	:param character_heatmap: Character Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
	:param affinity_heatmap: Affinity Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
	:param character_threshold: Threshold above which we say pixel belongs to a character
	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
	:return: {
		'word_bbox': word_bbox,
		'characters': char_bbox,
	}
	"""

	assert character_heatmap.max() <= 1, 'Weight has not been normalised'
	assert character_heatmap.min() >= 0, 'Weight has not been normalised'

	# character_heatmap being thresholded

	character_heatmap[character_heatmap > character_threshold] = 255
	character_heatmap[character_heatmap != 255] = 0

	assert affinity_heatmap.max() <= 1, 'Weight Affinity has not been normalised'
	assert affinity_heatmap.min() >= 0, 'Weight Affinity has not been normalised'

	# affinity_heatmap being thresholded

	affinity_heatmap[affinity_heatmap > affinity_threshold] = 255
	affinity_heatmap[affinity_heatmap != 255] = 0

	character_heatmap = character_heatmap.astype(np.uint8)
	affinity_heatmap = affinity_heatmap.astype(np.uint8)

	# Thresholded heat-maps being removed of stray contours

	character_heatmap = remove_small_predictions(character_heatmap)
	affinity_heatmap = remove_small_predictions(affinity_heatmap)

	# Finding the character and affinity contours

	all_characters, hierarchy = cv2.findContours(character_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	all_joins, hierarchy = cv2.findContours(affinity_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# In the beginning of training number of character predictions might be too high because the model performs poorly
	# Hence the check below does not waste time in calculating the f-score

	if len(all_characters) > 1000:

		return {
			'error_message': 'Number of characters too high'
		}

	if len(all_joins) > 1000:
		return {
			'error_message': 'Number of characters too high'
		}

	# Converting all affinity-contours and character-contours to affinity-bbox and character-bbox

	for ii in range(len(all_characters)):
		rect = cv2.minAreaRect(all_characters[ii])
		all_characters[ii] = cv2.boxPoints(rect)

	for ii in range(len(all_joins)):
		rect = cv2.minAreaRect(all_joins[ii])
		all_joins[ii] = cv2.boxPoints(rect)

	# The function join_with_characters joins the character_bbox using affinity_bbox to create word-bbox

	all_word_bbox, all_characters_bbox = join(all_characters, all_joins, return_characters=True)
	word_bbox = [i.tolist() for i in np.array(all_word_bbox)]
	char_bbox = [i.tolist() for i in np.array(all_characters_bbox)]

	return {
		'word_bbox': word_bbox,
		'characters': char_bbox
	}


def generate_word_bbox_batch(batch_character_heatmap, batch_affinity_heatmap, character_threshold, affinity_threshold):

	"""

	Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
	character bbox and word-bbox for the entire batch

	:param batch_character_heatmap: Batch Character Heatmap, numpy array, dtype=np.float32,
									shape = [batch_size, height, width], value range [0, 1]
	:param batch_affinity_heatmap: Batch Affinity Heatmap, numpy array, dtype=np.float32,
									shape = [batch_size, height, width], value range [0, 1]
	:param character_threshold: Threshold above which we say pixel belongs to a character
	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
	:return: word_bbox
	"""

	word_bbox = []
	batch_size = batch_affinity_heatmap.shape[0]

	for i in range(batch_size):

		returned = generate_word_bbox(
			batch_character_heatmap[i],
			batch_affinity_heatmap[i],
			character_threshold,
			affinity_threshold)

		if 'error_message' in returned.keys():
			word_bbox.append(np.zeros([0]))

		word_bbox.append(np.array(returned['word_bbox']))

	return word_bbox


def order_points(pts):

	"""
	Orders the 4 co-ordinates of a bounding box, top-left, top-right, bottom-right, bottom-left
	:param pts: numpy array with shape [4, 2]
	:return: numpy array, shape = [4, 2], ordered bbox
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
	Function to calculate IOU of two bbox

	:param poly1: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:param poly2: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:return: float representing the IOU
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
	Function to calculate the F-score of an entire batch. If lets say the model also predicted text,
	then a positive would be word_bbox IOU > threshold and exact text-match

	:param pred: list of numpy array having shape [num_words, 4, 2]
	:param target: list of numpy array having shape [num_words, 4, 2]
	:param text_pred: list of predicted text, (not useful for CRAFT)
	:param text_target: list of target text, (not useful for CRAFT)
	:param threshold: threshold value for iou above which we say a pair of bbox are positive
	:return:
	"""

	f_score = 0
	for i in range(len(pred)):
		if text_pred is not None:
			f_score += calculate_fscore(pred[i], target[i], text_pred[i], text_target[i], threshold)
		else:
			f_score += calculate_fscore(pred[i], target[i], threshold=threshold)

	# The corner case of there being no text in the image

	if len(pred) == 0 and len(target) == 0:
		return 1

	# The corner case of len(pred) being zero so f_score/len(pred) would give nan, though actually it should be 0

	elif len(pred) == 0:
		return 0

	return f_score/len(pred)


def get_smooth_polygon(word_contours):

	"""
	Takes many points and finds a convex hull of them. Used to get word-bbox
	from the characters and affinity bbox compromising the word-bbox

	:param word_contours: Contours to be joined to get one word in order (The contours are consecutive)
	:return: Numpy array of shape = [number_of_points, 1, 2]
	"""

	all_word_contours = np.concatenate(word_contours, axis=0)
	convex_hull = cv2.convexHull(all_word_contours).astype(np.int32)
	if convex_hull.shape[0] == 1:
		convex_hull = np.repeat(convex_hull, 4, axis=0)
	if convex_hull.shape[0] == 2:
		convex_hull = np.repeat(convex_hull, 2, axis=0)
	return convex_hull


def join(characters, joints, return_characters=False):

	"""
	Function to create word-polygon from character-bbox and affinity-bbox.
	A graph is created in which all the character-bbox and affinity-bbox are treated as nodes and there is an edge
	between them if IOU of the contours > 0. Then connected components are found and a convex hull is taken over all
	the nodes(char-bbox and affinity-bbox) compromising the word-polygon. This way the word-polygon is found

	:param characters: list of character bbox
	:param joints: list of join(affinity) bbox
	:param return_characters: a bool to toggle returning of character bbox corresponding to each word-polygon
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
		all_character_contours.append(
			all_joined[np.array(list(word_idx))[list(np.where(np.array(list(word_idx)) < len(characters)))[0]]])

	if return_characters:
		return all_word_contours, all_character_contours
	else:
		return all_word_contours
