import numpy as np
import matplotlib.pyplot as plt
import sys

"""
	A program to generate a moving average plot from the loss plot
"""

if __name__ == "__main__":
	
	
	if len(sys.argv) != 3:
		print('Usage: python running_mean path_to_loss.npy path_to_moving_average.png')
		exit(0)

	x = np.load(sys.argv[1])
	window = 1000

	average = list()
	average.append(x[0:window].mean())

	for i in range(window, x.shape[0]):
		average.append((average[-1]*window - x[i - window] + x[i])/window)

	plt.plot(average)
	plt.savefig(sys.argv[2])
