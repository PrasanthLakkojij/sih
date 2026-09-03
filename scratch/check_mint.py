import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from phase6_outage_sim import get_zupt_v1_mask, get_col, latlon_to_enu
from phase8_ml_correction import segment_features, FEAT_NAMES

# Let's inspect the cached yC and see if we should recompute with ZUPT v1 and MIN_T >= 2.0s
meta_all = pd.read_parquet('data/phase8_cache/meta_all.parquet')
yC_all = np.load('data/phase8_cache/yC_all.npy')
X_all = np.load('data/phase8_cache/X_all.npy')

# Filter MIN_T >= 2.0s
valid = meta_all['dur_s'].values >= 2.0
print(f"Total: {len(meta_all)}, valid (t >= 2.0s): {valid.sum()}")
print(f"yC_all on valid: mean={yC_all[valid].mean():.2f}m, std={yC_all[valid].std():.2f}m")
