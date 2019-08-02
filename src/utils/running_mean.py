import numpy as np

x = np.load('loss_plot_training.npy')
window = 1000

average = []
average.append(x[0:window].mean())

for i in range(window, x.shape[0]):
	average.append((average[-1]*window - x[i - window] + x[i])/window)

import matplotlib.pyplot as plt

plt.plot(average)
plt.savefig('moving_average.png')