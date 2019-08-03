def train(model, optimizer, iteration):

	# ToDo - Read the paper again and note down the important points
	# ToDo - Create the dataloader to create weighted mask for the
	#  characters and also generate the target given the iterations

	model.train()
	optimizer.zero_grad()
	iterator = tqdm(dataloader)

	def change_lr(no):
		#Change learning rate while training
		for i in config.lr:
			if i == no:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer.param_groups:
					param_group['lr'] = config.lr[i]

	for no, (image, weight, weight_affinity) in enumerate(iterator):

		if no < STARTING_NO:
			continue

		if epoch == 0:
			change_lr(no)

		if DATA_DEBUG:
			continue

		if config.pretrained:
			if no == STARTING_NO:
				dataloader.start = True
				continue
			elif no < STARTING_NO:
				continue

		if config.use_cuda:
			image, weight, weight_affinity = image.cuda(), weight.cuda(), weight_affinity.cuda()

		output = model(image)
		loss = lossCriterian(output, weight, weight_affinity).mean()

		all_loss.append(loss.item())

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if len(all_accuracy) == 0:
			iterator.set_description(
				'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
					len(iterator)) +
				'] Average Loss:' + str(
					int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000000) / 100000000))

		else:

			iterator.set_description(
				'Loss:' + str(int(loss.item() * 100000000) / 100000000) + ' Iterations:[' + str(no) + '/' + str(
					len(iterator)) +
				'] Average Loss:' + str(
					int(np.array(all_loss)[-min(1000, len(all_loss)):].mean() * 100000000) / 100000000) +
				'| Average F-Score: ' + str(
					int(np.array(all_accuracy)[-min(1000, len(all_accuracy)):].mean() * 100000000) / 100000000)
			)

		if no >= 1000:
			if no % config.periodic_fscore == 0 and no != 0:
				if type(output) == list:
					output = torch.cat(output, dim=0)
				predicted_bbox = get_word_poly(output[:, 0, :, :].data.cpu().numpy(),
				                               output[:, 1, :, :].data.cpu().numpy())
				target_bbox = get_word_poly(weight.data.cpu().numpy(), weight_affinity.data.cpu().numpy())
				all_accuracy.append(
					calculate_batch_fscore(predicted_bbox, target_bbox, threshold=config.threshold_fscore))

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