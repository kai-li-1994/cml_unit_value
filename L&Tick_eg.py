# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:50:34 2025

@author: k.li@cml.leidenuniv.nl
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Helper Functions ----------

def compute_tangents(x, bic, min_idx):
    return [
        (bic[i] - bic[min_idx]) / abs(x[i] - x[min_idx]) if i != min_idx else 0
        for i in range(len(bic))
    ]

def compute_tangents_split(x, bic, min_idx):
    tangents_pre = []
    tangents_post = []
    for i in range(len(bic)):
        if i < min_idx:
            slope = (bic[i] - bic[min_idx]) / abs(x[i] - x[min_idx])
            tangents_pre.append(slope)
        elif i > min_idx:
            slope = (bic[i] - bic[min_idx]) / abs(x[i] - x[min_idx])
            tangents_post.append(slope)
    return tangents_pre, tangents_post

# ---------- L-shape Data ----------
x_l = np.arange(2, 15)
bic_l = np.array([
    17000, 16000, 15000, 14000, 12800,
    12700, 12700, 12600, 12550, 12600,
    12600, 12700, 12800
])
min_l_idx = np.argmin(bic_l)
tangents_l = compute_tangents(x_l, bic_l, min_l_idx)
pre_slopes_l = tangents_l[:min_l_idx]
max_pre_slope_l = max(pre_slopes_l) if pre_slopes_l else 1

l_adjust_idx = None
ratios_l = []
for i in range(min_l_idx):
    ratio = tangents_l[i] / max_pre_slope_l if max_pre_slope_l else 0
    ratios_l.append(ratio)
    if l_adjust_idx is None and ratio < 0.2:
        l_adjust_idx = i

# ---------- Tick-shape Data ----------
x_t = np.arange(2, 15)
bic_t = np.array([
    14800, 14400, 14000, 13920, 13910, 13905, 13902, 13800,
    14300, 14800, 15300, 15800, 16300
])
min_t_idx = np.argmin(bic_t)
tangents_t = compute_tangents(x_t, bic_t, min_t_idx)
tangents_t_pre, tangents_t_post = compute_tangents_split(x_t, bic_t, min_t_idx)
max_post_slope = max(tangents_t_post) if tangents_t_post else 1

tick_adjust_idx = None
ratios_t = []
for i in range(min_t_idx):
    ratio = tangents_t[i] / max_post_slope if max_post_slope else 0
    ratios_t.append(ratio)
    if tick_adjust_idx is None and ratio < 0.2:
        tick_adjust_idx = i
# Combine both L-shape and Tick-shape into a single figure with annotations
mpl.rcParams['pdf.fonttype'] = 42  
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# --- L-shape subplot ---
axs[0].plot(x_l, bic_l, marker='o', color='blue', label='BIC', markersize=6)
axs[0].axvline(x_l[min_l_idx], color='red', linestyle='--', label='Global Min')
slope_labeled1,slope_labeled2 = False, False # flag to label 'Slope (tangent)' only once
if l_adjust_idx is not None:
    axs[0].axvline(x_l[l_adjust_idx], color='orange', linestyle=':', label='L-shape Adj.')

for i in range(len(tangents_l)):
    if i == min_l_idx:
        continue
    if i < min_l_idx:
        mid_x = (x_l[i] + x_l[min_l_idx]) / 2
        mid_y = (bic_l[i] + bic_l[min_l_idx]) / 2
        slope_text = f"{tangents_l[i]:.0f}"
        ratio = tangents_l[i] / max_pre_slope_l if max_pre_slope_l else 0
        ratio_text = f"{100*ratio:.0f}%"
        color = "orange" if ratio < 0.2 else "gray"

        # Set label only once for each slope group
        if ratio < 0.2:
            label = 'Slope (tangent<20%)' if not slope_labeled2 else None
            slope_labeled2 = True
        else:
            label = 'Slope (tangent≥20%)' if not slope_labeled1 else None
            slope_labeled1 = True

        axs[0].plot([x_l[i], x_l[min_l_idx]], [bic_l[i], bic_l[min_l_idx]],
                    color=color, alpha=0.5, label=label)
        axs[0].text(mid_x, mid_y + 40, f"{slope_text}\n({ratio_text})", fontsize=8,
                    ha='center', color=color)

axs[0].set_title("L-shape pattern (Early saturation)")
axs[0].set_xlabel("Number of Components")
axs[0].set_ylabel("BIC")
axs[0].grid(True, color='lightgray', alpha=0.8) 
axs[0].legend(loc='upper right')
# --- Tick-shape subplot ---
axs[1].plot(x_t, bic_t, marker='o', color='blue', label='BIC', markersize=6)
axs[1].axvline(x_t[min_t_idx], color='red', linestyle='--', label='Global Min')
slope_labeled1, slope_labeled2 = False, False  # flags to avoid duplicate labels
if tick_adjust_idx is not None:
    axs[1].axvline(x_t[tick_adjust_idx], color='green', linestyle=':', label='Tick-shape Adj.')

for i in range(len(tangents_t)):
    if i == min_t_idx:
        continue
    mid_x = (x_t[i] + x_t[min_t_idx]) / 2
    mid_y = (bic_t[i] + bic_t[min_t_idx]) / 2
    slope_text = f"{tangents_t[i]:.0f}"
    if i != min_t_idx:
        ratio = tangents_t[i] / max_post_slope if max_post_slope else 0
        ratio_text = f"{100*ratio:.0f}%"
        color = "green" if ratio < 0.2 else "gray"
        # Set label only once for each slope group
        if ratio < 0.2:
            label = 'Slope (tangent<20%)' if not slope_labeled2 else None
            slope_labeled2 = True
        else:
            label = 'Slope (tangent≥20%)' if not slope_labeled1 else None
            slope_labeled1 = True
            
        axs[1].text(mid_x, mid_y + 40, f"{slope_text}\n({ratio_text})", fontsize=8,
                    ha='center', color=color, zorder=1)
        axs[1].plot([x_t[i], x_t[min_t_idx]], [bic_t[i], bic_t[min_t_idx]], 
                    'gray', alpha=0.5,label=label, color=color,zorder =1)




axs[1].set_title("Tick-shape pattern (Post-minimum overfitting)")
axs[1].set_xlabel("Number of Components")
axs[1].grid(True, color='lightgray', alpha=0.8) 
axs[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('LTick_eg.pdf', dpi=300)  # Save with desired filename and resolution
plt.show()
