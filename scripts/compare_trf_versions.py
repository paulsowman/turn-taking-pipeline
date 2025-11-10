#!/usr/bin/env python3
"""
Quick script to compare TRF kernels between new and safe versions.
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# Paths
new_model = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline/outputs/trf_multipredictor/sub-01/runs-1_3_5/mag/trf_models.pkl")
safe_model = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline/outputs/trf_multipredictor/sub-01/runs-1_3_5_safe/mag/trf_models.pkl")

print("Loading models...")
with open(new_model, 'rb') as f:
    new_models = pickle.load(f)

with open(safe_model, 'rb') as f:
    safe_models = pickle.load(f)

print("\nComparing envelope_only models:")
new_env = new_models['envelope_only'].h.x  # (sensors, times)
safe_env = safe_models['envelope_only'].h.x

# Flatten for correlation
new_env_flat = new_env.flatten()
safe_env_flat = safe_env.flatten()

corr_env, p_env = pearsonr(new_env_flat, safe_env_flat)
print(f"  Envelope kernel correlation: {corr_env:.4f} (p={p_env:.2e})")
print(f"  New shape: {new_env.shape}, Safe shape: {safe_env.shape}")
print(f"  New range: [{new_env.min():.6f}, {new_env.max():.6f}]")
print(f"  Safe range: [{safe_env.min():.6f}, {safe_env.max():.6f}]")

print("\nComparing f0_only models:")
new_f0 = new_models['f0_only'].h.x
safe_f0 = safe_models['f0_only'].h.x

new_f0_flat = new_f0.flatten()
safe_f0_flat = safe_f0.flatten()

corr_f0, p_f0 = pearsonr(new_f0_flat, safe_f0_flat)
print(f"  F0 kernel correlation: {corr_f0:.4f} (p={p_f0:.2e})")
print(f"  New shape: {new_f0.shape}, Safe shape: {safe_f0.shape}")
print(f"  New range: [{new_f0.min():.6f}, {new_f0.max():.6f}]")
print(f"  Safe range: [{safe_f0.min():.6f}, {safe_f0.max():.6f}]")

print("\nComparing full models:")
# Full model is a tuple of (envelope_kernel, f0_kernel)
new_full_env = new_models['full'].h[0].x
new_full_f0 = new_models['full'].h[1].x
safe_full_env = safe_models['full'].h[0].x
safe_full_f0 = safe_models['full'].h[1].x

# Compare envelope component of full model
new_full_env_flat = new_full_env.flatten()
safe_full_env_flat = safe_full_env.flatten()
corr_full_env, p_full_env = pearsonr(new_full_env_flat, safe_full_env_flat)
print(f"  Full model envelope component correlation: {corr_full_env:.4f} (p={p_full_env:.2e})")

# Compare f0 component of full model
new_full_f0_flat = new_full_f0.flatten()
safe_full_f0_flat = safe_full_f0.flatten()
corr_full_f0, p_full_f0 = pearsonr(new_full_f0_flat, safe_full_f0_flat)
print(f"  Full model F0 component correlation: {corr_full_f0:.4f} (p={p_full_f0:.2e})")

# Check if there are systematic differences
print("\nDifference statistics:")
diff_env = new_env - safe_env
print(f"  Envelope diff: mean={diff_env.mean():.6f}, std={diff_env.std():.6f}, max_abs={np.abs(diff_env).max():.6f}")

diff_f0 = new_f0 - safe_f0
print(f"  F0 diff: mean={diff_f0.mean():.6f}, std={diff_f0.std():.6f}, max_abs={np.abs(diff_f0).max():.6f}")

diff_full = new_full - safe_full
print(f"  Full diff: mean={diff_full.mean():.6f}, std={diff_full.std():.6f}, max_abs={np.abs(diff_full).max():.6f}")
