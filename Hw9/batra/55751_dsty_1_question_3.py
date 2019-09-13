import numpy as np
import matplotlib.pyplot as plt

def part_a(s,k):
	samples = []

	for i in range(s):
		samples.append(np.random.normal(0,1,k))

	MLE = []
	for sample in samples:
		MLE.append(np.mean(sample))

	Var = np.cov(MLE)
	
	return Var




s = 1000
varying_k = []
for k in range(1,1000):
	varying_k.append(part_a(s,k))


plt.plot(varying_k)
plt.grid(True)
plt.title("varying k")
plt.show()


k = 1000
varying_s = []
for s in range(1,1000):
	varying_s.append(part_a(s,k))


plt.plot(varying_s)
plt.grid(True)
plt.title("varying s")
plt.show()

