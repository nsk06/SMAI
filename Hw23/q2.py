import random
import numpy as np
import math
import matplotlib.pyplot as plt

x = [[-1,0],[1,1],[1,3],[2,2],[5,-1],[2,2],[-1,9],[10,-10],[0,0],[1,-2]]
errors = [] 
for k in range(1,11):
	y = []
	for i in range(k):
		y.append(i)
	for i in range(k,10):
		r = random.randrange(0,k,1)
		y.append(r)
	change = 1
	mean_coord = np.zeros((k,2))
	while(change):
		mean_coord = np.zeros((k,2))
		set_size = np.zeros((k))
		change = 0
		for i in range(10):
			mean_coord[y[i]][0] += x[i][0]
			mean_coord[y[i]][1] += x[i][1]
			set_size[y[i]] += 1
		for i in range(k):
			mean_coord[i][0] /= set_size[i]
			mean_coord[i][1] /= set_size[i]
		for i in range(10):
			mini = y[i]
			min_d = (x[i][0] - mean_coord[y[i]][0])**2 + (x[i][1] - mean_coord[y[i]][1])**2
			for j in range(k):
				dis = (x[i][0] - mean_coord[j][0])**2 + (x[i][1] - mean_coord[j][1])**2
				if dis < min_d:
					min_d = dis
					mini = j
					change = 1
			y[i] = mini

	error = k*k/30
	for i in range(10):
		error += math.sqrt((x[i][0] - mean_coord[y[i]][0])**2 + (x[i][1] - mean_coord[y[i]][1])**2)/10
	errors.append(error)

plt.plot(range(1,11),errors,'r')
plt.title("K v/s Regularized Error")
plt.show()