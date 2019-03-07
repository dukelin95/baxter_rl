import matplotlib.pyplot as plt
import numpy as np

np.load('errors.npy')
s0, s1, e1, w1 = list(zip(*graph))

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
ax1.set_title("s0")
ax2.set_title("s1")
ax3.set_title("e1")
ax4.set_title("w1")
ax1.plot(s0)
ax2.plot(s1)
ax3.plot(e1)
ax4.plot(w1)