import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfc

# Exercise 2 - Task 2

MC = int(1e5)     # Monte Carlo simulations
N = 50            # Samples
A = 1             # DC
sigma_squared = 1 # Noise variance
cost = 2          # Symmetric costs C01 = C10 = 2

# Generate noise samples
w_n = np.sqrt(sigma_squared) * np.random.randn(MC, N)

# Hypotheses
H0 = w_n
H1 = (1/np.sqrt(N)) * np.ones((MC, N)) + w_n

# Randomly pick a realization from H0 or H1
idx = np.random.randint(0, 2, size=(MC, 1))  # 0 or 1 for each trial
samples = np.where(idx == 1, H1, H0)
idx = idx[:, 0]   # Flatten to 1D

# Decide H0 or H1
gamma = A / (2 * np.sqrt(N))     # Threshold
sample_av = np.mean(samples, axis=1)

decisions = (sample_av > gamma).astype(int)
outcome = decisions - idx        # 1=FA, -1=Miss, 0=Correct

# Empirical probabilities
PFA = np.sum(outcome == 1) / np.sum(idx == 0)
PM  = np.sum(outcome == -1) / np.sum(idx == 1)

r_bayes = 0.5 * cost * PFA + 0.5 * cost * PM

p_error = (np.sum(outcome == 1) + np.sum(outcome == -1)) / MC

# Theoretical probability of error
qfunc = lambda x: 0.5 * erfc(x / np.sqrt(2))
p_error_th = qfunc(A / (2 * np.sqrt(sigma_squared)))

print("Empirical False Alarm Probability:", PFA)
print("Empirical Miss Probability:", PM)
print("Bayesian Risk:", r_bayes)
print("Empirical Error Probability:", p_error)
print("Theoretical Error Probability:", p_error_th)

# Task 3

ts = sample_av
ts_H0 = ts[idx == 0]
ts_H1 = ts[idx == 1]

# Histogram-based PDFs (by ChatGPT)
nbins = 80
sim_pdf_H0, edges = np.histogram(ts_H0, bins=nbins, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])

sim_pdf_H1, _ = np.histogram(ts_H1, bins=edges, density=True)

# Theoretical PDFs
mu0 = 0
mu1 = A / np.sqrt(N)
sigma_ts = np.sqrt(sigma_squared / N)

th_pdf_H0 = norm.pdf(centers, mu0, sigma_ts)
th_pdf_H1 = norm.pdf(centers, mu1, sigma_ts)

# Plot (by ChatGPT)
plt.figure(figsize=(10, 6))
plt.plot(centers, sim_pdf_H0, 'b', linewidth=1.4, label='Simulated H0')
plt.plot(centers, sim_pdf_H1, 'r', linewidth=1.4, label='Simulated H1')

plt.plot(centers, th_pdf_H0, '--b', linewidth=1.4, label='Theoretical H0')
plt.plot(centers, th_pdf_H1, '--r', linewidth=1.4, label='Theoretical H1')

plt.axvline(gamma, color='k', linewidth=2, label='Threshold')

plt.xlabel('Test statistic (mean(X))')
plt.ylabel('PDF')
plt.title('Simulated and Theoretical PDFs of Test Statistic')
plt.legend()
plt.grid(True)
plt.show()
