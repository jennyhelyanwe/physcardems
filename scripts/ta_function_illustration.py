import numpy as np
import matplotlib.pyplot as plt

def ta(tau, Ta_d, Ta_peak, t_peak, a, sigma, T):
    tau = np.asarray(tau, float)
    s = np.clip(tau / t_peak, 0, 1)
    rise = np.sin(0.5 * np.pi * s**a) ** 2
    g = lambda x: np.exp(-0.5 * (x / sigma) ** 2)
    e1 = g(T - t_peak)
    fall = (g(tau - t_peak) - e1) / (1 - e1)
    S = np.where(tau < t_peak, rise, fall)
    return Ta_d + (Ta_peak - Ta_d) * S

# Illustrative values; Ta_d exaggerated so it is visible
T, Ta_d, Ta_peak, t_peak, a, sigma = 800, 2.0, 27.0, 170, 1.3, 120
tau = np.linspace(0, T, 1000)

fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

# Panel 1: annotated base curve
ax = axs[0]
ax.plot(tau, ta(tau, Ta_d, Ta_peak, t_peak, a, sigma, T), "k", lw=2)
ax.axhline(Ta_peak, ls=":", c="grey"); ax.axhline(Ta_d, ls=":", c="grey")
ax.axvline(t_peak, ls=":", c="grey")
ax.annotate("$Ta_{peak}$", (T * 0.7, Ta_peak + 0.6))
ax.annotate("$Ta_d$", (T * 0.7, Ta_d + 0.6))
ax.annotate("$t_{peak}$", (t_peak + 10, 4))
ax.axvspan(0, t_peak, color="C0", alpha=0.08)
ax.axvspan(t_peak, T, color="C3", alpha=0.08)
ax.text(t_peak / 2, 15, "sin$^2$ rise", ha="center", c="C0")
ax.text((t_peak + T) / 2, 15, "Gaussian decay", ha="center", c="C3")
ax.set_title("Prescribed Ta profile")
ax.set_xlabel(r"$\tau$, time since local activation (ms)"); ax.set_ylabel("Ta (kPa)")

# Panel 2: effect of a
ax = axs[1]
for av in [0.6, 1.0, 1.6, 2.5]:
    ax.plot(tau, ta(tau, Ta_d, Ta_peak, t_peak, av, sigma, T), label=f"a = {av}")
ax.axvline(t_peak, ls=":", c="grey")
ax.set_title("a: shape of the rise (small a = front-loaded)")
ax.set_xlabel(r"$\tau$ (ms)"); ax.legend()

# Panel 3: effect of sigma
ax = axs[2]
for sv in [60, 100, 150, 250]:
    ax.plot(tau, ta(tau, Ta_d, Ta_peak, t_peak, a, sv, T), label=f"σ = {sv} ms")
ax.axvline(t_peak, ls=":", c="grey")
ax.set_title("σ: rate of relaxation")
ax.set_xlabel(r"$\tau$ (ms)"); ax.legend()

for ax in axs:
    ax.set_xlim(0, T); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("ta_profile_illustration.png", dpi=200)