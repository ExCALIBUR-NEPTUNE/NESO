import matplotlib.pyplot as plt
import numpy as np

E = np.loadtxt("E.txt")
print(E)

for i in range(0, len(E[:, 0])):
    plt.plot(E[i, :])
plt.savefig("E_vs_time.pdf")
plt.clf()

Esquared = []
for i in range(0, len(E[:, 0])):
    Esquared.append(sum(E[i, :] ** 2))

plt.semilogy(Esquared)
plt.savefig("E2_vs_time.pdf")
