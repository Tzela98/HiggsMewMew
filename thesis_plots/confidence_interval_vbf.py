import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)

# Data
classic_fit = {
    "median": 3.64,
    "1_sigma": (2.60, 5.09),
    "2_sigma": (1.96, 6.84)
}

ml_fit = {
    "median": 2.21,
    "1_sigma": (1.57, 3.12),
    "2_sigma": (1.17, 4.27)
}

ml_dimuon_fit = {
    "median": 2.93,
    "1_sigma": (2.09, 4.13),
    "2_sigma": (1.55, 5.57)
}

# Plotting
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)

# Plot classic fit
ax.errorbar(
    1, classic_fit["median"],
    yerr=[[classic_fit["median"] - classic_fit["1_sigma"][0]], [classic_fit["1_sigma"][1] - classic_fit["median"]]],
    fmt='_', color='black', capsize=14, label='Classic Fit'
)
ax.errorbar(
    1, classic_fit["median"],
    yerr=[[classic_fit["median"] - classic_fit["2_sigma"][0]], [classic_fit["2_sigma"][1] - classic_fit["median"]]],
    fmt='_', color='black', alpha=0.9, capsize=28
)

# Plot Asymptotic Limits fit (swapped with ML Fit)
ax.errorbar(
    2, ml_dimuon_fit["median"],
    yerr=[[ml_dimuon_fit["median"] - ml_dimuon_fit["1_sigma"][0]], [ml_dimuon_fit["1_sigma"][1] - ml_dimuon_fit["median"]]],
    fmt='_', color='black', capsize=14, label='Asymptotic Limits'
)
ax.errorbar(
    2, ml_dimuon_fit["median"],
    yerr=[[ml_dimuon_fit["median"] - ml_dimuon_fit["2_sigma"][0]], [ml_dimuon_fit["2_sigma"][1] - ml_dimuon_fit["median"]]],
    fmt='_', color='black', alpha=0.9, capsize=28
)

# Plot ML fit (swapped with Asymptotic Limits)
ax.errorbar(
    3, ml_fit["median"],
    yerr=[[ml_fit["median"] - ml_fit["1_sigma"][0]], [ml_fit["1_sigma"][1] - ml_fit["median"]]],
    fmt='_', color='black', capsize=14, label='ML Fit'
)
ax.errorbar(
    3, ml_fit["median"],
    yerr=[[ml_fit["median"] - ml_fit["2_sigma"][0]], [ml_fit["2_sigma"][1] - ml_fit["median"]]],
    fmt='_', color='black', alpha=0.9, capsize=28
)

# Customization
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([r'$m_{\mu_1 \mu_2}\ \mathrm{Fit}$', r'$m_{\mu_1 \mu_2}\ \mathrm{Fit + NN Cut}$', 'NN Output Fit'])
ax.set_ylabel(r'$\mu$', rotation=90)

# Adding shaded regions for sigma intervals
ax.fill_between([0.85, 1.15], classic_fit["1_sigma"][0], classic_fit["1_sigma"][1], color='orangered', alpha=0.7)
ax.fill_between([1.85, 2.15], ml_dimuon_fit["1_sigma"][0], ml_dimuon_fit["1_sigma"][1], color='orangered', alpha=0.7)
ax.fill_between([2.85, 3.15], ml_fit["1_sigma"][0], ml_fit["1_sigma"][1], color='orangered', alpha=0.7)
ax.fill_between([0.85, 1.15], classic_fit["2_sigma"][0], classic_fit["2_sigma"][1], color='orangered', alpha=0.4)
ax.fill_between([1.85, 2.15], ml_dimuon_fit["2_sigma"][0], ml_dimuon_fit["2_sigma"][1], color='orangered', alpha=0.4)
ax.fill_between([2.85, 3.15], ml_fit["2_sigma"][0], ml_fit["2_sigma"][1], color='orangered', alpha=0.4)

# Adding custom legend for the shaded regions
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orangered', alpha=0.7, label=r'$1\sigma\ \mathrm{Interval}$'),
    Patch(facecolor='orangered', alpha=0.4, label=r'$2\sigma\ \mathrm{Interval}$')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS \ Simulation}}$', loc='left', pad=10, fontsize=24)
plt.title(r'59.7 fb$^{-1}$ at 13 TeV (2018)', loc='right', pad=10, fontsize=18)
plt.ylim(0, 7.5)
plt.hlines(1, 0.5, 3.5, linestyles='dashed', color='black')
plt.xlim(0.5, 3.5)

plt.savefig('thesis_plots/plots_vbf/confidence_intervals_vbf.png')