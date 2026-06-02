import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

SNR_dB = np.arange(-20, 45, 5)
SNR = 10**(SNR_dB / 10)

Nreal = 500

configs = np.array([
    [4, 4],
    [4, 8],
    [8, 8]
])

cfg_names = ['8x8', '4x8', '4x4']

# ============================================================
# Main loop
# ============================================================

for cfg in range(len(configs)):

    Nt = configs[cfg, 0]
    Nr = configs[cfg, 1]

    print(f'Running {Nt}x{Nr} ...')

    ratio_mf = np.zeros(len(SNR))
    ratio_zf = np.zeros(len(SNR))
    ratio_mmse = np.zeros(len(SNR))
    ratio_sic = np.zeros(len(SNR))

    # ========================================================
    # SNR loop
    # ========================================================

    for s, rho in enumerate(SNR):

        Csum = 0.0

        Rmf_sum = 0.0
        Rzf_sum = 0.0
        Rmmse_sum = 0.0
        Rsic_sum = 0.0

        # ====================================================
        # Monte Carlo
        # ====================================================

        for mc in range(Nreal):

            # Rayleigh channel
            H = (np.random.randn(Nr, Nt)
                 + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)

            # =================================================
            # Capacity (stable implementation)
            # =================================================

            M = np.eye(Nr) + (rho / Nt) * (H @ H.conj().T)

            sign, logdet = np.linalg.slogdet(M)

            C = logdet / np.log(2)

            Csum += C

            # =================================================
            # 1) Matched Filter
            # =================================================

            Rmf = 0.0

            for k in range(Nt):

                hk = H[:, k]

                signal_power = (rho / Nt) * np.abs(
                    hk.conj().T @ hk
                )**2

                interference_power = 0.0

                for j in range(Nt):

                    if j != k:

                        hj = H[:, j]

                        interference_power += (
                            (rho / Nt)
                            * np.abs(hk.conj().T @ hj)**2
                        )

                noise_power = np.linalg.norm(hk)**2

                gamma_mf = signal_power / (
                    interference_power + noise_power
                )

                Rmf += np.log2(1 + gamma_mf)

            Rmf_sum += Rmf

            # =================================================
            # 2) ZF / Decorrelator
            # =================================================

            G = np.linalg.pinv(H.conj().T @ H)

            gamma_zf = np.maximum(
                (rho / Nt) / np.real(np.diag(G)),
                0
            )

            Rzf = np.sum(np.log2(1 + gamma_zf))

            Rzf_sum += Rzf

            # =================================================
            # 3) MMSE
            # =================================================

            A = np.linalg.pinv(
                np.eye(Nt)
                + (rho / Nt) * (H.conj().T @ H)
            )

            gamma_mmse = np.maximum(
                1 / np.real(np.diag(A)) - 1,
                0
            )

            Rmmse = np.sum(np.log2(1 + gamma_mmse))

            Rmmse_sum += Rmmse

            # =================================================
            # 4) MMSE-SIC
            # =================================================

            Hsic = H.copy()

            Rsic = 0.0

            while Hsic.shape[1] > 0:

                ns = Hsic.shape[1]

                Asic = np.linalg.pinv(
                    np.eye(ns)
                    + (rho / Nt)
                    * (Hsic.conj().T @ Hsic)
                )

                gamma_sic = np.maximum(
                    1 / np.real(np.diag(Asic)) - 1,
                    0
                )

                idx = np.argmax(gamma_sic)

                gmax = gamma_sic[idx]

                Rsic += np.log2(1 + gmax)

                Hsic = np.delete(Hsic, idx, axis=1)

            Rsic_sum += Rsic

        # ====================================================
        # Ratios
        # ====================================================

        Cerg = Csum / Nreal

        ratio_mf[s] = Rmf_sum / (Nreal * Cerg)
        ratio_zf[s] = Rzf_sum / (Nreal * Cerg)
        ratio_mmse[s] = Rmmse_sum / (Nreal * Cerg)
        ratio_sic[s] = Rsic_sum / (Nreal * Cerg)

        print(
            f'SNR={SNR_dB[s]:3d} dB completed'
        )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        SNR_dB,
        ratio_sic,
        linewidth=2,
        label='MMSE-SIC'
    )

    plt.plot(
        SNR_dB,
        ratio_mmse,
        '--',
        linewidth=2,
        label='MMSE'
    )

    plt.plot(
        SNR_dB,
        ratio_mf,
        ':',
        linewidth=2,
        label='Matched Filter'
    )

    plt.plot(
        SNR_dB,
        ratio_zf,
        '-.',
        linewidth=2,
        label='Decorrelator'
    )

    plt.grid(True)

    plt.xlabel('SNR (dB)')
    plt.ylabel(f'R / C_{{{Nt}{Nr}}}')

    plt.ylim([0, 1.05])

    plt.title(f'MIMO {cfg_names[cfg]}')

    plt.legend()

    plt.tight_layout()

    plt.show()