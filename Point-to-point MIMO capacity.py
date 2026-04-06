import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ================= PARAMETERS =================
Nt = 8
Nr = 8
channel_realizations = 100   # increase later (CVX is slow)
SNR_dB = np.arange(-10, 31, 10)
N0 = 1

# Results storage
cvx_ld_rate = np.zeros(len(SNR_dB))
cvx_rate = np.zeros(len(SNR_dB))
wf_rate = np.zeros(len(SNR_dB))
eq_rate = np.zeros(len(SNR_dB))

# ================= MAIN LOOP =================
for s, snr in enumerate(SNR_dB):
    P = N0 * 10**(snr / 10)
    print("\n" + "="*60)
    print(f"STARTING SNR = {snr} dB   (P = {P:.3f})")
    print("="*60)

    for rel in tqdm(range(channel_realizations), desc=f"SNR {snr} dB"):

        t0 = time.time()

        # ================= CHANNEL =================
        H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)

        # ================= SVD =================
        U, S, Vh = np.linalg.svd(H)
        gains = S**2
        NL = min(Nt, Nr)

        # ================= LOG-DET CVX =================
        Q = cp.Variable((Nt, Nt), hermitian=True)
        objective = cp.Maximize(cp.log_det(np.eye(Nr) + H @ Q @ H.conj().T / N0) / np.log(2))
        constraints = [Q >> 0, cp.trace(Q) <= P]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        Q_opt = Q.value
        rate_ld = np.real(np.log2(np.linalg.det(np.eye(Nr) + H @ Q_opt @ H.conj().T / N0)))
        cvx_ld_rate[s] += rate_ld

        # ================= SVD CVX =================
        p = cp.Variable(NL)
        objective = cp.Maximize(cp.sum(cp.log(1 + cp.multiply(gains, p) / N0)) / np.log(2))
        constraints = [cp.sum(p) <= P, p >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        p_opt = p.value
        rate_svd = np.sum(np.log2(1 + gains * p_opt / N0))
        cvx_rate[s] += rate_svd

        # ================= WATERFILLING =================
        inv_gains = N0 / gains
        wf_low = np.min(inv_gains)
        wf_high = P + np.max(inv_gains)

        while abs(wf_low - wf_high) > 1e-10:
            wf = (wf_low + wf_high) / 2
            p_wf = np.maximum(wf - inv_gains, 0)
            if np.sum(p_wf) > P:
                wf_high = wf
            else:
                wf_low = wf

        rate_wf = np.sum(np.log2(1 + gains * p_wf / N0))
        wf_rate[s] += rate_wf

        # ================= EQUAL POWER =================
        p_eq = P / NL
        rate_eq = np.sum(np.log2(1 + gains * p_eq / N0))
        eq_rate[s] += rate_eq

        # ================= VERBOSE DEBUG PRINT =================
        if rel % 5 == 0:   # print every 5 realizations
            print(f"\n[DEBUG] Realization {rel}")
            print(f"  logdet = {rate_ld:.6f}")
            print(f"  SVD-CVX = {rate_svd:.6f}")
            print(f"  WF      = {rate_wf:.6f}")
            print(f"  EQ      = {rate_eq:.6f}")
            print(f"  time = {time.time() - t0:.3f} s")

            # Check mismatch
            print(f"  |logdet - WF| = {abs(rate_ld - rate_wf):.3e}")
            print(f"  |SVD - WF|    = {abs(rate_svd - rate_wf):.3e}")

# ================= AVERAGING =================
cvx_ld_rate /= channel_realizations
cvx_rate /= channel_realizations
wf_rate /= channel_realizations
eq_rate /= channel_realizations

# ================= PLOT =================
plt.figure(figsize=(9,6))
plt.plot(SNR_dB, cvx_ld_rate, '-go', label='log-det (CVX)')
plt.plot(SNR_dB, cvx_rate, '-b.', label='SVD (CVX)')
plt.plot(SNR_dB, wf_rate, '--r*', label='Waterfilling')
plt.plot(SNR_dB, eq_rate, '--kd', label='Equal Power')

plt.title("Point-to-Point MIMO Capacity")
plt.xlabel("SNR [dB]")
plt.ylabel("Ergodic Sum Rate [bits/s/Hz]")
plt.grid(True)
plt.legend()
plt.show()
