from shapely.geometry import Polygon
import numpy as np
import cv2
from .data_manipulation import two_char_bbox_to_affinity
import math


def order_points(box):

	x_sorted_arg = np.argsort(box[:, 0])
	if box[x_sorted_arg[0], 1] > box[x_sorted_arg[1], 1]:
		tl = x_sorted_arg[1]
	else:
		tl = x_sorted_arg[0]

	ordered_bbox = np.array([box[(tl + i) % 4] for i in range(4)])

	return ordered_bbox


def _init_fn(worker_id):

	"""
	Function to make the pytorch dataloader deterministic
	:param worker_id: id of the parallel worker
	:return:
	"""

	np.random.seed(0 + worker_id)


def resize_bbox(original_dim, output, config):

	max_dim = original_dim.max()
	resizing_factor = 768 / max_dim
	before_pad_dim = [int(original_dim[0] * resizing_factor), int(original_dim[1] * resizing_factor)]

	output = np.uint8(output * 255)

	height_pad = (768 - before_pad_dim[0]) // 2
	width_pad = (768 - before_pad_dim[1]) // 2

	character_bbox = cv2.resize(
		output[0, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
		(original_dim[1]//2, original_dim[0]//2)) / 255

	affinity_bbox = cv2.resize(
		output[1, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
		(original_dim[1]//2, original_dim[0]//2)) / 255

	generated_targets = generate_word_bbox(
		character_bbox,
		affinity_bbox,
		character_threshold=config.threshold_character,
		affinity_threshold=config.threshold_affinity,
		word_threshold=config.threshold_word,
		character_threshold_upper=config.threshold_character_upper,
		affinity_threshold_upper=config.threshold_affinity_upper,
		scaling_affinity=config.scale_affinity,
		scaling_character=config.scale_character,
	)

	generated_targets['word_bbox'] = generated_targets['word_bbox']*2
	generated_targets['characters'] = [i*2 for i in generated_targets['characters']]
	generated_targets['affinity'] = [i*2 for i in generated_targets['affinity']]

	return generated_targets


def poly_to_rect(bbox_contour):

	return cv2.boxPoints(
		cv2.minAreaRect(np.array(bbox_contour, dtype=np.int64).squeeze())).reshape([4, 1, 2]).astype(np.int64)


def weighing_function(orig_length, cur_length):

	"""
	Function to generate the weight value given the predicted text-length and the expected text-length
	The intuition is that if the predicted text-length is far from the expected text-length then weight should be
	small to that word-bbox.

	:param orig_length: Length of the expected word bounding box
	:param cur_length: Length of the predicted word bounding box
	:return:
	"""

	if orig_length == 0:
		if cur_length == 0:
			return 1
		return 0

	return (orig_length - min(orig_length, abs(orig_length - cur_length)))/orig_length


def cutter(word_bbox, num_characters):

	"""
	Given a word_bbox of 4 co-ordinates and the number of characters inside it,
	generates equally spaced character bbox
	:param word_bbox: numpy array of shape [4, 1, 2], dtype=np.int64
	:param num_characters: integer denoting number of characters inside the word_bbox
	:return:
		numpy array containing bbox of all the characters, dtype = np.float32, shape = [num_characters, 4, 1, 2],
		numpy array containing affinity between characters, dtype = np.float32, shape = [num_characters, 4, 1, 2],
	"""

	if Polygon(word_bbox[:, 0, :]).area == 0:
		return np.zeros([0, 4, 1, 2], dtype=np.int64), np.zeros([0, 4, 1, 2], dtype=np.int64)

	word_bbox[:, 0, :] = order_points(word_bbox[:, 0, :])

	edge_length = [np.sqrt(np.sum(np.square(word_bbox[i, 0, :] - word_bbox[(i+1) % 4, 0, :]))) for i in range(4)]

	if edge_length[0]*2 < edge_length[1]:
		tl, tr, br, bl = 1, 2, 3, 0
	else:
		tl, tr, br, bl = 0, 1, 2, 3

	if edge_length[0] == 0:
		return np.zeros([0, 4, 1, 2], dtype=np.int64), np.zeros([0, 4, 1, 2], dtype=np.int64)

	width_0 = edge_length[tl]
	width_1 = edge_length[tr]

	if width_0 != 0:
		direction_0 = (word_bbox[tr, 0] - word_bbox[tl, 0]) / width_0
	else:
		direction_0 = np.float32([0, 0])

	if width_1 != 0:
		direction_1 = (word_bbox[br, 0] - word_bbox[bl, 0])/width_1
	else:
		direction_1 = np.float32([0, 0])

	character_width_0 = width_0/num_characters
	character_width_1 = width_1/num_characters

	char_bbox = np.zeros([num_characters, 4, 1, 2], dtype=np.int64)
	affinity_bbox = np.zeros([num_characters - 1, 4, 1, 2], dtype=np.int64)
	co_ordinates = np.zeros([num_characters + 1, 2, 2], dtype=np.int64)

	co_ordinates[0, 0] = word_bbox[tl, 0]
	co_ordinates[0, 1] = word_bbox[bl, 0]

	for i in range(1, num_characters + 1):

		co_ordinates[i, 0] = co_ordinates[i - 1, 0] + direction_0*character_width_0
		co_ordinates[i, 1] = co_ordinates[i - 1, 1] + direction_1*character_width_1

		char_bbox[i-1, 0, 0] = co_ordinates[i - 1, 0]
		char_bbox[i-1, 1, 0] = co_ordinates[i, 0]
		char_bbox[i-1, 2, 0] = co_ordinates[i, 1]
		char_bbox[i-1, 3, 0] = co_ordinates[i - 1, 1]

	for i in range(num_characters - 1):

		affinity_bbox[i] = two_char_bbox_to_affinity(char_bbox[i], char_bbox[i+1])

	return char_bbox.astype(np.int64), affinity_bbox.astype(np.int64)


def get_weighted_character_target(generated_targets, original_annotation, unknown_symbol, threshold, weight_threshold):

	"""

	Function to generate targets using weak-supervision which will be used to fine-tune the model trained using
	Synthetic data.

	:param generated_targets: {
			'word_bbox': word_bbox, type=np.array, dtype=np.int64, shape=[num_words, 4, 1, 2]
			'characters': char_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_characters, 4, 1, 2]
			'affinity': affinity_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_affinity, 4, 1, 2]
		}
	:param original_annotation: {
			'bbox': list of shape [num_words, 4, 2] containing word-bbox of original annotations,
			'text': list of shape [num_words] containing text in original annotations
		}
	:param unknown_symbol: The symbol(string) which denotes that the text was not annotated
	:param threshold: overlap IOU value above which we consider prediction as positive
	:param weight_threshold: threshold of predicted_char/target_chars above which we say the prediction will be used as
								a target in the next iteration
	:return: aligned_generated_targets: {
			'word_bbox': contains the word-bbox which have been annotated and present in original annotations,
							type = np.array, dtype=np.int64, shape = [num_words, 4, 1, 2]
			'characters': will contain the character-bbox generated using weak-supervision,
							type = list of np.array, shape = [num_words, num_character_in_each_word, 4, 1, 2]
			'affinity': will contain the affinity-bbox generated using weak-supervision,
							type = list of np.array, shape = [num_words, num_affinity_in_each_word, 4, 1, 2]
			'text' : list of all annotated text present in original_annotation,
							type = list, shape = [num_words]
			'weights' : list containing list for character and affinity having values between 0 and 1 denoting weight
						of that particular word-bbox for character and affinity respectively in the loss while
						training weak-supervised model
							type = list, shape = [num_words, 2]
		}
	"""

	original_annotation['bbox'] = np.array(original_annotation['bbox'], dtype=np.int64)[:, :, None, :]
	# Converting original annotations to the shape [num_words, 4, 1, 2]

	assert len(original_annotation['bbox']) == len(original_annotation['text']), \
		'Number of word Co-ordinates do not match with number of text'

	num_words = len(original_annotation['text'])

	aligned_generated_targets = {
		'word_bbox': original_annotation['bbox'],
		'characters': [[] for _ in range(num_words)],
		'affinity': [[] for _ in range(num_words)],
		'text': original_annotation['text'],
		'weights': [0 for _ in range(num_words)]
	}

	for orig_no, orig_annot in enumerate(original_annotation['bbox']):

		# orig_annot - co-ordinates of one word-bbox, type = np.array, dtype = np.int64, shape = [4, 1, 2]

		found_no = -1

		for no, gen_t in enumerate(generated_targets['word_bbox']):

			# ToDo - For ic13, as the bbox are always horizontal we should use better calc_iou

			if calc_iou(np.array(gen_t), np.array(orig_annot)) > threshold:
				# If IOU between predicted and original word-bbox is > threshold then it is termed positive
				found_no = no
				break

		if original_annotation['text'][orig_no] == unknown_symbol or len(original_annotation['text'][orig_no]) == 0:

			"""
				If the current original annotation was predicted by the model but the text-annotation is not present 
				then we create character bbox using predictions and give a weight of 0.5 to the word-bbox
			"""

			# Some annotation in 2017 has length of word == 0. Hence adding max

			assert np.all(np.array(orig_annot.shape) == np.array([4, 1, 2])), str(orig_annot.shape) + ' error in original annot'

			aligned_generated_targets['characters'][orig_no] = orig_annot[None]
			aligned_generated_targets['affinity'][orig_no] = orig_annot[None]
			aligned_generated_targets['weights'][orig_no] = [0, 0]

		elif found_no == -1:

			"""
				If the current original annotation was not predicted by the model then we create equi-spaced character
				bbox and give a weight of 0.5 to the word-bbox
			"""

			characters, affinity = cutter(word_bbox=orig_annot, num_characters=len(original_annotation['text'][orig_no]))

			aligned_generated_targets['characters'][orig_no] = characters
			aligned_generated_targets['affinity'][orig_no] = affinity
			aligned_generated_targets['weights'][orig_no] = [0.5, 0.5]

		else:

			"""
				If the current original annotation was predicted by the model and the text-annotation is present 
				then we find the weight using the weighing_function and if it is less than 0.5 then create equi-spaced 
				character bbox otherwise use the generated char-bbox using predictions and give a weight of value of 
				weighing_function to the word-bbox
			"""

			weight_char = weighing_function(
				len(original_annotation['text'][orig_no]), len(generated_targets['characters'][found_no]))
			weight_aff = weighing_function(
				len(original_annotation['text'][orig_no])-1, len(generated_targets['affinity'][found_no]))

			applied_weight = [0, 0]

			if weight_char <= weight_threshold:

				characters, affinity = cutter(orig_annot, len(original_annotation['text'][orig_no]))

				aligned_generated_targets['characters'][orig_no] = characters
				applied_weight[0] = 0.5

			else:

				aligned_generated_targets['characters'][orig_no] = generated_targets['characters'][found_no]
				applied_weight[0] = weight_char

			if weight_aff <= weight_threshold:

				characters, affinity = cutter(orig_annot, len(original_annotation['text'][orig_no]))

				aligned_generated_targets['affinity'][orig_no] = affinity
				applied_weight[1] = 0.5

			else:

				aligned_generated_targets['affinity'][orig_no] = generated_targets['affinity'][found_no]
				applied_weight[1] = weight_aff

			aligned_generated_targets['weights'][orig_no] = applied_weight

	return aligned_generated_targets


def generate_word_bbox(
		character_heatmap,
		affinity_heatmap,
		character_threshold,
		affinity_threshold,
		word_threshold,
		character_threshold_upper,
		affinity_threshold_upper,
		scaling_character,
		scaling_affinity
	):

	"""
	Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
	character bbox and word-bbox

	:param character_heatmap: Character Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
	:param affinity_heatmap: Affinity Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
	:param character_threshold: Threshold above which we say pixel belongs to a character
	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
	:param word_threshold: Threshold of any pixel above which we say a group of characters for a word
	:param character_threshold_upper: Threshold above which we differentiate the characters
	:param affinity_threshold_upper: Threshold above which we differentiate the affinity
	:param scaling_character: how much to scale the character bbox
	:param scaling_affinity: how much to scale the affinity bbox
	:return: {
		'word_bbox': word_bbox, type=np.array, dtype=np.int64, shape=[num_words, 4, 1, 2] ,
		'characters': char_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_characters, 4, 1, 2] ,
		'affinity': affinity_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_affinity, 4, 1, 2] ,
	}
	"""

	img_h, img_w = character_heatmap.shape

	""" labeling method """
	ret, text_score = cv2.threshold(character_heatmap, character_threshold, 1, 0)
	ret, link_score = cv2.threshold(affinity_heatmap, affinity_threshold, 1, 0)

	text_score_comb = np.clip(text_score + link_score, 0, 1)

	n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		text_score_comb.astype(np.uint8),
		connectivity=4)

	det = []
	mapper = []
	for k in range(1, n_labels):

		try:
			# size filtering
			size = stats[k, cv2.CC_STAT_AREA]
			if size < 10:
				continue

			where = labels == k

			# thresholding
			if np.max(character_heatmap[where]) < word_threshold:
				continue

			# make segmentation map
			seg_map = np.zeros(character_heatmap.shape, dtype=np.uint8)
			seg_map[where] = 255
			seg_map[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area

			x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
			w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
			niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
			sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
			# boundary check
			if sx < 0:
				sx = 0
			if sy < 0:
				sy = 0
			if ex >= img_w:
				ex = img_w
			if ey >= img_h:
				ey = img_h

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
			seg_map[sy:ey, sx:ex] = cv2.dilate(seg_map[sy:ey, sx:ex], kernel)

			# make box
			np_contours = np.roll(np.array(np.where(seg_map != 0)), 1, axis=0).transpose().reshape(-1, 2)
			rectangle = cv2.minAreaRect(np_contours)
			box = cv2.boxPoints(rectangle)

			if Polygon(box).area == 0:
				continue

			# align diamond-shape
			w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
			box_ratio = max(w, h) / (min(w, h) + 1e-5)
			if abs(1 - box_ratio) <= 0.1:
				l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
				t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
				box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

			# make clock-wise order
			start_idx = box.sum(axis=1).argmin()
			box = np.roll(box, 4 - start_idx, 0)
			box = np.array(box)

			det.append(box)
			mapper.append(k)

		except:
			# ToDo - Understand why there is a ValueError: math domain error in line
			#  niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)

			continue

	ret, text_score = cv2.threshold(character_heatmap, character_threshold_upper, 1, 0)
	ret, link_score = cv2.threshold(affinity_heatmap, affinity_threshold_upper, 1, 0)

	char_contours, _ = cv2.findContours(text_score.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	affinity_contours, _ = cv2.findContours(link_score.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	char_contours = scale_bbox(char_contours, scaling_character)
	affinity_contours = scale_bbox(affinity_contours, scaling_affinity)

	char_contours = link_to_word_bbox(char_contours, det)
	affinity_contours = link_to_word_bbox(affinity_contours, det)

	return {
		'word_bbox': np.array(det, dtype=np.int32).reshape([len(det), 4, 1, 2]),
		'characters': char_contours,
		'affinity': affinity_contours,
	}


def scale_bbox(contours, scale):

	mean = [np.array(i)[:, 0, :].mean(axis=0) for i in contours]
	centered_contours = [np.array(contours[i]) - mean[i][None, None, :] for i in range(len(contours))]
	scaled_contours = [centered_contours[i]*scale for i in range(len(centered_contours))]
	shifted_back = [scaled_contours[i] + mean[i][None, None, :] for i in range(len(scaled_contours))]
	shifted_back = [i.astype(np.int32) for i in shifted_back]

	return shifted_back


def link_to_word_bbox(to_find, word_bbox):

	if len(word_bbox) == 0:
		return [np.zeros([0, 4, 1, 2], dtype=np.int32)]
	word_sorted_character = [[] for _ in word_bbox]

	for cont_i, cont in enumerate(to_find):

		if cont.shape[0] < 4:
			continue

		rectangle = cv2.minAreaRect(cont)
		box = cv2.boxPoints(rectangle)

		if Polygon(box).area == 0:
			continue

		ordered_bbox = order_points(box)

		a = Polygon(cont.reshape([cont.shape[0], 2])).buffer(0)

		if a.area == 0:
			continue

		ratio = np.zeros([len(word_bbox)])

		# Polygon intersection usd for checking ratio

		for word_i, word in enumerate(word_bbox):
			b = Polygon(word.reshape([word.shape[0], 2])).buffer(0)
			ratio[word_i] = a.intersection(b).area/a.area

		word_sorted_character[np.argmax(ratio)].append(ordered_bbox)

	word_sorted_character = [
		np.array(word_i, dtype=np.int32).reshape([len(word_i), 4, 1, 2]) for word_i in word_sorted_character]

	return word_sorted_character


def generate_word_bbox_batch(
		batch_character_heatmap,
		batch_affinity_heatmap,
		character_threshold,
		affinity_threshold,
		word_threshold):

	"""

	Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
	character bbox and word-bbox for the entire batch

	:param batch_character_heatmap: Batch Character Heatmap, numpy array, dtype=np.float32,
									shape = [batch_size, height, width], value range [0, 1]
	:param batch_affinity_heatmap: Batch Affinity Heatmap, numpy array, dtype=np.float32,
									shape = [batch_size, height, width], value range [0, 1]
	:param character_threshold: Threshold above which we say pixel belongs to a character
	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
	:param word_threshold: Threshold above which we say a group of characters compromise a word
	:return: word_bbox
	"""

	word_bbox = []
	batch_size = batch_affinity_heatmap.shape[0]

	for i in range(batch_size):

		returned = generate_word_bbox(
			batch_character_heatmap[i],
			batch_affinity_heatmap[i],
			character_threshold,
			affinity_threshold,
			word_threshold)

		word_bbox.append(returned['word_bbox'])

	return word_bbox


def calc_iou(poly1, poly2):

	"""
	Function to calculate IOU of two bbox

	:param poly1: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:param poly2: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:return: float representing the IOU
	"""

	a = Polygon(poly1.reshape([poly1.shape[0], 2])).buffer(0)
	b = Polygon(poly2.reshape([poly2.shape[0], 2])).buffer(0)

	union_area = a.union(b).area

	if union_area == 0:
		return 0

	return a.intersection(b).area/union_area


def calculate_fscore(pred, target, text_target, unknown='###', text_pred=None, threshold=0.5):

	"""

	:param pred: numpy array with shape [num_words, 4, 2]
	:param target: numpy array with shape [num_words, 4, 2]
	:param text_target: list of the target text
	:param unknown: do not care text bbox
	:param text_pred: predicted text (Not useful in CRAFT implementation)
	:param threshold: overlap iou threshold over which we say the pair is positive
	:return:
	"""

	assert len(text_target) == target.shape[0], 'Some error in text target'

	if pred.shape[0] == target.shape[0] == 0:
		return {
			'f_score': 1,
			'precision': 1,
			'recall': 1,
			'false_positive': 0,
			'true_positive': 0,
			'num_positive': 0
		}

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

	if text_target is not None:
		true_positive = np.sum(already_done.astype(np.float32)[np.where(np.array(text_target) != unknown)[0]])
	else:
		true_positive = np.sum(already_done.astype(np.float32))

	if text_target is not None:
		num_positive = (np.where(np.array(text_target) != unknown)[0]).shape[0]
	else:
		num_positive = len(target)

	if true_positive == 0:
		return {
			'f_score': 0,
			'precision': 0,
			'recall': 0,
			'false_positive': false_positive,
			'true_positive': true_positive,
			'num_positive': num_positive
		}

	precision = true_positive/(true_positive + false_positive)
	recall = true_positive / num_positive

	return {
		'f_score': 2*precision*recall/(precision + recall),
		'precision': precision,
		'recall': recall,
		'false_positive': false_positive,
		'true_positive': true_positive,
		'num_positive': num_positive
	}


def calculate_batch_fscore(pred, target, text_target, unknown='###', text_pred=None, threshold=0.5):

	"""
	Function to calculate the F-score of an entire batch. If lets say the model also predicted text,
	then a positive would be word_bbox IOU > threshold and exact text-match

	:param pred: list of numpy array having shape [num_words, 4, 2]
	:param target: list of numpy array having shape [num_words, 4, 2]
	:param text_target: list of target text, (not useful for CRAFT)
	:param text_pred: list of predicted text, (not useful for CRAFT)
	:param unknown: text specifying do not care scenario
	:param threshold: threshold value for iou above which we say a pair of bbox are positive
	:return:
	"""
	if text_target is None:
		text_target = [''.join(['_' for __ in range(len(target[_]))]) for _ in range(len(pred))]

	f_score = 0
	precision = 0
	recall = 0

	for i in range(len(pred)):
		if text_pred is not None:
			stats = calculate_fscore(pred[i], target[i], text_target[i], unknown, text_pred[i], threshold)
			f_score += stats['f_score']
			precision += stats['precision']
			recall += stats['recall']
		else:
			stats = calculate_fscore(pred[i], target[i], text_target[i], unknown, threshold=threshold)
			f_score += stats['f_score']
			precision += stats['precision']
			recall += stats['recall']

	return f_score/len(pred), precision/len(pred), recall/len(pred)


def get_smooth_polygon(word_contours):

	"""
	Takes many points and finds a convex hull of them. Used to get word-bbox
	from the characters and affinity bbox compromising the word-bbox

	:param word_contours: Contours to be joined to get one word in order (The contours are consecutive)
	:return: Numpy array of shape = [number_of_points, 1, 2]
	"""

	all_word_contours = np.concatenate(word_contours, axis=0)
	convex_hull = cv2.convexHull(all_word_contours).astype(np.int64)
	if convex_hull.shape[0] == 1:
		convex_hull = np.repeat(convex_hull, 4, axis=0)
	if convex_hull.shape[0] == 2:
		convex_hull = np.repeat(convex_hull, 2, axis=0)

	return convex_hull
