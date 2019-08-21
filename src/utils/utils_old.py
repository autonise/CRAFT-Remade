# import networkx as nx

# Replacing with CRAFT's implementation
# def remove_small_predictions(image):
#
# 	"""
# 	This function is used to erode small stray character predictions but does not reduce the thickness of big predictions
# 	:param image: Predicted character or affinity heat map in uint8
# 	:return: image with less stray contours
# 	"""
#
# 	# kernel = np.ones((5, 5), np.uint8)
# 	# image = cv2.erode(image, kernel, iterations=2)
# 	# image = cv2.dilate(image, kernel, iterations=3)
#
# 	return image

# Replacing generate_word_bbox with craft implementation which is a bit modified
#
# def generate_word_bbox(character_heatmap, affinity_heatmap, character_threshold, affinity_threshold):
#
# 	"""
# 	Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
# 	character bbox and word-bbox
#
# 	:param character_heatmap: Character Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
# 	:param affinity_heatmap: Affinity Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
# 	:param character_threshold: Threshold above which we say pixel belongs to a character
# 	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
# 	:return: {
# 		'word_bbox': word_bbox, type=np.array, dtype=np.int64, shape=[num_words, 4, 1, 2] ,
# 		'characters': char_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_characters, 4, 1, 2] ,
# 		'affinity': affinity_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_affinity, 4, 1, 2] ,
# 	}
# 	"""
#
# 	assert character_heatmap.max() <= 1, 'Weight has not been normalised'
# 	assert character_heatmap.min() >= 0, 'Weight has not been normalised'
#
# 	# character_heatmap being thresholded
#
# 	character_heatmap[character_heatmap > character_threshold] = 255
# 	character_heatmap[character_heatmap != 255] = 0
#
# 	assert affinity_heatmap.max() <= 1, 'Weight Affinity has not been normalised'
# 	assert affinity_heatmap.min() >= 0, 'Weight Affinity has not been normalised'
#
# 	# affinity_heatmap being thresholded
#
# 	affinity_heatmap[affinity_heatmap > affinity_threshold] = 255
# 	affinity_heatmap[affinity_heatmap != 255] = 0
#
# 	character_heatmap = character_heatmap.astype(np.uint8)
# 	affinity_heatmap = affinity_heatmap.astype(np.uint8)
#
# 	# Thresholded heat-maps being removed of stray contours
#
# 	character_heatmap = remove_small_predictions(character_heatmap)
# 	affinity_heatmap = remove_small_predictions(affinity_heatmap)
#
# 	# Finding the character and affinity contours
#
# 	all_characters, hierarchy = cv2.findContours(character_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 	all_joins, hierarchy = cv2.findContours(affinity_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# 	# In the beginning of training number of character predictions might be too high because the model performs poorly
# 	# Hence the check below does not waste time in calculating the f-score
#
# 	if len(all_characters) > 1000 or len(all_joins) > 1000:
# 		return {
# 			'word_bbox': np.zeros([0, 4, 1, 2]),
# 			'characters': [],
# 			'affinity': []
# 		}
#
# 	# Converting all affinity-contours and character-contours to affinity-bbox and character-bbox
#
# 	for ii in range(len(all_characters)):
# 		rect = cv2.minAreaRect(all_characters[ii])
# 		all_characters[ii] = cv2.boxPoints(rect)[:, None, :]
#
# 	for ii in range(len(all_joins)):
# 		rect = cv2.minAreaRect(all_joins[ii])
# 		all_joins[ii] = cv2.boxPoints(rect)[:, None, :]
#
# 	all_characters = np.array(all_characters, dtype=np.int64).reshape([len(all_characters), 4, 1, 2])
# 	all_joins = np.array(all_joins, dtype=np.int64).reshape([len(all_joins), 4, 1, 2])
#
# 	# The function join_with_characters joins the character_bbox using affinity_bbox to create word-bbox
#
# 	all_word_bbox, all_characters_bbox, all_affinity_bbox = join(all_characters, all_joins, return_characters=True)
#
# 	return {
# 		'word_bbox': all_word_bbox,
# 		'characters': all_characters_bbox,
# 		'affinity': all_affinity_bbox
# 	}


# Replacing join with craft's implementation
#
# def join(characters, joints, return_characters=False):
#
# 	"""
# 	Function to create word-polygon from character-bbox and affinity-bbox.
# 	A graph is created in which all the character-bbox and affinity-bbox are treated as nodes and there is an edge
# 	between them if IOU of the contours > 0. Then connected components are found and a convex hull is taken over all
# 	the nodes(char-bbox and affinity-bbox) compromising the word-polygon. This way the word-polygon is found
#
# 	:param characters: type=np.array, dtype=np.int64, shape=[num_chars, 4, 1, 2]
# 	:param joints: type=np.array, dtype=np.int64, shape=[num_affinity, 4, 1, 2]
# 	:param return_characters: a bool to toggle returning of character bbox corresponding to each word-polygon
# 	:return:
# 		all_word_contours: type = np.array, dtype=np.int64, shape=[num_words, 4, 1, 2]
# 		all_character_contours: type = list of np.array, dtype=np.int64, shape =[num_words, num_chars, 4, 1, 2]
# 		all_affinity_contours: type = list of np.array, dtype=np.int64, shape =[num_words, num_chars, 4, 1, 2]
# 	"""
#
# 	all_joined = np.concatenate([characters, joints], axis=0)
#
# 	graph = nx.Graph()
# 	graph.add_nodes_from(nx.path_graph(all_joined.shape[0]))
#
# 	for contour_i in range(all_joined.shape[0]-1):
# 		for contour_j in range(contour_i+1, all_joined.shape[0]):
# 			if Polygon(all_joined[contour_i, :, 0, :]).intersection(Polygon(all_joined[contour_j, :, 0, :])).area > 0:
# 				graph.add_edge(contour_i, contour_j)
#
# 	all_words = nx.connected_components(graph)
#
# 	all_word_contours = []
# 	all_character_contours = []
# 	all_affinity_contours = []
#
# 	for word_idx in all_words:
#
# 		index_chars = np.array(list(word_idx))[list(np.where(np.array(list(word_idx)) < len(characters)))[0]]
#
# 		if len(index_chars) == 0:
# 			continue
#
# 		index_affinity = np.array(list(word_idx))[list(np.where(np.array(list(word_idx)) >= len(characters)))[0]]
#
# 		# Converting the characters and affinity to a polygon and then converting it to rectangle.
# 		# In the future can be changed to produce polygon outputs
# 		all_word_contours.append(poly_to_rect(get_smooth_polygon(all_joined[list(word_idx)])))
#
# 		all_character_contours.append(all_joined[index_chars])
#
# 		if len(index_affinity) != 0:
# 			all_affinity_contours.append(all_joined[index_affinity])
# 		else:
# 			all_affinity_contours.append(np.zeros([0, 4, 1, 2]))
#
# 	all_word_contours = np.array(all_word_contours, dtype=np.int64).reshape([len(all_word_contours), 4, 1, 2])
#
# 	if return_characters:
# 		return all_word_contours, all_character_contours, all_affinity_contours
# 	else:
# 		return all_word_contours
