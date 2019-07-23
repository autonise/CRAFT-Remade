import config
from model import UNetWithResnet50Encoder, Criterian
from dataloader import DataLoaderSYNTH
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from utils.parallel import DataParallelModel, DataParallelCriterion
from utils.utils import calculate_batch_fscore, get_word_poly

DATA_DEBUG = False

if DATA_DEBUG:
	config.num_cuda = '0'
	config.batchsize['test'] = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(data, output, target, target_affinity, epoch, no):

	output = output.data.cpu().numpy()
	data = data.data.cpu().numpy()
	target = target.data.cpu().numpy()
	target_affinity = target_affinity.data.cpu().numpy()

	batchsize = output.shape[0]

	base = 'test_synthesis/'+str(epoch)+'_'+str(no)+'/'

	os.makedirs(base, exist_ok=True)

	for i in range(batchsize):
		
		os.makedirs(base+str(i), exist_ok=True)
		character_bbox = output[i, 0, :, :]
		affinity_bbox = output[i, 1, :, :]

		plt.imsave(base+str(i) + '/image.png', data[i].transpose(1, 2, 0))

		plt.imsave(base+str(i) + '/target_characters.png', target[i, :, :], cmap='gray')
		plt.imsave(base+str(i) + '/target_affinity.png', target_affinity[i, :, :], cmap='gray')

		plt.imsave(base + str(i) + '/pred_characters.png', character_bbox, cmap='gray')
		plt.imsave(base + str(i) + '/pred_affinity.png', affinity_bbox, cmap='gray')

		plt.imsave(base + str(i) + '/pred_characters_thresh.png', np.float32(character_bbox>config.threshold_character), cmap='gray')
		plt.imsave(base + str(i) + '/pred_affinity_thresh.png', np.float32(affinity_bbox>config.threshold_affinity), cmap='gray')


def test(dataloader, model):

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)
		all_loss = []
		all_accuracy = []

		for no, (image, weight, weight_affinity) in enumerate(iterator):

			if DATA_DEBUG:
				continue

			if config.use_cuda:
				image, weight, weight_affinity = image.cuda(), weight.cuda(), weight_affinity.cuda()

			output = model(image)
			loss = lossCriterian(output, weight, weight_affinity).mean()

			all_loss.append(loss.item())
			if type(output) == list:
				output = torch.cat(output, dim=0)
			predicted_bbox = get_word_poly(output[:, 0, :, :].data.cpu().numpy(), output[:, 1, :, :].data.cpu().numpy())
			target_bbox = get_word_poly(weight.data.cpu().numpy(), weight_affinity.data.cpu().numpy())
			all_accuracy.append(calculate_batch_fscore(predicted_bbox, target_bbox, threshold=config.threshold_fscore))

			iterator.set_description(
				'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
					len(iterator)) +
				'] Average Loss:' + str(int(np.array(all_loss)[-min(1000, len(all_loss)):].mean()*100000000)/100000000) +
				'| Average F-Score: ' + str(int(np.array(all_accuracy)[-min(1000, len(all_accuracy)):].mean()*100000000)/100000000)
			)

			if no % config.periodic_output == 0 and no != 0:
				if type(output) == list:
					output = torch.cat(output, dim=0)
				save(image, output, weight, weight_affinity, epoch, no)

		return all_loss


def seed():

	# This removes randomness, makes everything deterministic

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

	seed()

	epoch = 0

	model = UNetWithResnet50Encoder()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])

	print('Total number of trainable parameters: ', params)

	model = DataParallelModel(model)
	lossCriterian = DataParallelCriterion(Criterian())

	test_dataloader = DataLoaderSYNTH('test')

	if config.use_cuda:
		model = model.cuda()

	test_dataloader = DataLoader(
		test_dataloader, batch_size=config.batchsize['test'],
		shuffle=True, num_workers=16)

	saved_model = torch.load(config.pretrained_path_test)
	model.load_state_dict(saved_model['state_dict'])

	all_loss = test(test_dataloader, model)
