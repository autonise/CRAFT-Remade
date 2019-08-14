from src.model import UNetWithResnet50Encoder, Criterian
from .dataloader import DataLoaderSYNTH
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import random
import train_synth.config as config

from src.utils.parallel import DataParallelModel, DataParallelCriterion
from src.utils.utils import calculate_batch_fscore, generate_word_bbox_batch


os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(data, output, target, target_affinity, epoch, no):

	"""
	Saving the synthesised outputs in between the training
	:param data: image as tensor
	:param output: predicted output from the model as tensor
	:param target: character heatmap target as tensor
	:param target_affinity: affinity heatmap target as tensor
	:param no: current iteration number
	:param epoch: current epoch
	:return: None
	"""

	output = output.data.cpu().numpy()
	data = data.data.cpu().numpy()
	target = target.data.cpu().numpy()
	target_affinity = target_affinity.data.cpu().numpy()

	batch_size = output.shape[0]

	base = config.DataLoaderSYNTH_Train_Synthesis+str(epoch)+'_'+str(no)+'/'

	os.makedirs(base, exist_ok=True)

	for i in range(batch_size):

		os.makedirs(base+str(i), exist_ok=True)
		character_bbox = output[i, 0, :, :]
		affinity_bbox = output[i, 1, :, :]

		plt.imsave(base+str(i) + '/image.png', data[i].transpose(1, 2, 0))

		plt.imsave(base+str(i) + '/target_characters.png', target[i, :, :], cmap='gray')
		plt.imsave(base+str(i) + '/target_affinity.png', target_affinity[i, :, :], cmap='gray')

		plt.imsave(base + str(i) + '/pred_characters.png', character_bbox, cmap='gray')
		plt.imsave(base + str(i) + '/pred_affinity.png', affinity_bbox, cmap='gray')

		# Thresholding the character and affinity heatmap

		plt.imsave(
			base + str(i) + '/pred_characters_thresh.png',
			np.float32(character_bbox > config.threshold_character),
			cmap='gray')
		plt.imsave(
			base + str(i) + '/pred_affinity_thresh.png',
			np.float32(affinity_bbox > config.threshold_affinity),
			cmap='gray')


def train(dataloader, loss_criterian, model, optimizer, starting_no, epoch, all_loss, all_accuracy):

	"""
	Function to test
	:param dataloader: Pytorch dataloader
	:param loss_criterian: Loss function with OHNM using MSE Loss
	:param model: Pytorch model of UNet-ResNet
	:param optimizer: Adam Optimizer
	:param starting_no: how many items to skip in the dataloader
	:param epoch: current epoch
	:param all_loss: list of all loss values
	:param all_accuracy: list of all f-scores
	:return: all iteration loss values
	"""

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	def change_lr(no_i):

		for i in config.lr:
			if i == no_i:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer.param_groups:
					param_group['lr'] = config.lr[i]

	for no, (image, weight, weight_affinity) in enumerate(iterator):

		if no < starting_no:
			continue

		if epoch == 0:
			change_lr(no)

		if config.pretrained:
			if no == starting_no:
				dataloader.start = True
				continue
			elif no < starting_no:
				continue

		if config.use_cuda:
			image, weight, weight_affinity = image.cuda(), weight.cuda(), weight_affinity.cuda()

		output = model(image)
		loss = loss_criterian(output, weight, weight_affinity).mean()

		all_loss.append(loss.item())

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if len(all_accuracy) == 0:
			iterator.set_description(
				'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
					len(iterator)) +
				'] Average Loss:' + str(int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000000) / 100000000))

		else:

			iterator.set_description(
				'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
					len(iterator)) +
				'] Average Loss:' + str(int(np.array(all_loss)[-min(1000, len(all_loss)):].mean()*100000000)/100000000) +
				'| Average F-Score: ' + str(int(np.array(all_accuracy)[-min(1000, len(all_accuracy)):].mean()*100000000)/100000000)
			)

		if no >= 1000:

			# Calculating the f-score after some iterations because initially there are a lot of stray contours

			if no % config.periodic_fscore == 0 and no != 0:

				if type(output) == list:
					output = torch.cat(output, dim=0)

				predicted_bbox = generate_word_bbox_batch(
					output[:, 0, :, :].data.cpu().numpy(),
					output[:, 1, :, :].data.cpu().numpy(),
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity)

				target_bbox = generate_word_bbox_batch(
					weight.data.cpu().numpy(),
					weight_affinity.data.cpu().numpy(),
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity)

				all_accuracy.append(calculate_batch_fscore(predicted_bbox, target_bbox, threshold=config.threshold_fscore))

		if no % config.periodic_output == 0 and no != 0:

			if type(output) == list:
				output = torch.cat(output, dim=0)

			save(image, output, weight, weight_affinity, epoch, no)

		if no % config.periodic_save == 0 and no != 0:

			torch.save(
				{
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict()
				}, config.save_path + '/' + str(no) + '_model.pkl')

			np.save(config.save_path + '/loss_plot_training.npy', all_loss)
			plt.plot(all_loss)
			plt.savefig(config.save_path + '/loss_plot_training.png')
			plt.clf()

	return all_loss


def seed():

	# This removes randomness, makes everything deterministic

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


def main():

	seed()

	copyfile('train_synth/config.py', config.save_path + '/config.py')

	model = UNetWithResnet50Encoder()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])

	print('Total number of trainable parameters: ', params)

	model = DataParallelModel(model)
	loss_criterian = DataParallelCriterion(Criterian())

	train_dataloader = DataLoaderSYNTH('train')

	if config.use_cuda:
		model = model.cuda()

	train_dataloader = DataLoader(
		train_dataloader, batch_size=config.batch_size['train'],
		shuffle=True, num_workers=24)

	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr[1])

	if config.pretrained:
		saved_model = torch.load(config.pretrained_path)
		model.load_state_dict(saved_model['state_dict'])
		optimizer.load_state_dict(saved_model['optimizer'])
		starting_no = int(config.pretrained_path.split('/')[-1].split('_')[0])
		all_loss = np.load(config.pretrained_loss_plot_training).tolist()

	else:
		starting_no = 0
		all_loss = []

	all_accuracy = []

	for epoch in range(1, 2):
		if epoch != 0:
			starting_no = 0
		all_loss += train(
			train_dataloader, loss_criterian, model, optimizer, starting_no=starting_no, epoch=epoch,
			all_loss=all_loss, all_accuracy=all_accuracy)

	torch.save(
		{
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict()
		}, config.save_path + '/final_model.pkl')

	np.save(config.save_path + '/loss_plot_training.npy', all_loss)
	plt.plot(all_loss)
	plt.savefig(config.save_path + '/loss_plot_training.png')
	plt.clf()


if __name__ == "__main__":

	main()
