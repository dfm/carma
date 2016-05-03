import time
import numpy as np
import matplotlib.pyplot as pl

from carma._carma import carma_log_likelihood

np.random.seed(1234)

sigma = 0.1
a = np.array([-0.5 + 0.01j, -0.5 - 0.01j, -1.0 + 0.1j, -1.0 - 0.1j])
b = np.array([-0.5 + 0.1j, -0.5 - 0.1j])

N = 2 ** np.arange(5, 20)
dts = np.empty(len(N))

x = np.sort(10*np.random.rand(N.max())) - 5
yerr = 0.01 * np.ones_like(x)
y = 5.0 * np.sin(x)

for i, n in enumerate(N):
    strt = time.time()
    ll = carma_log_likelihood(sigma, a, b, x[:n], y[:n], yerr[:n])
    dts[i] = time.time() - strt

    print(n, dts[i], ll)

pl.loglog(N, dts, "-ok")
pl.ylabel("time in seconds")
pl.xlabel("number of datapoints")
pl.savefig("benchmark.png")
