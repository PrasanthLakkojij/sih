import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from phase6_outage_sim import sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask, latlon_to_enu, get_col
from phase4_orientation import transform_to_vehicle_frame
from phase8_ml_correction import segment_features

# Let's compute yC_zupt for S-Vw4 test
# We can recompute yC_zupt on the cached valid segments
# Let's see how much difference yC_open vs yC_zupt makes
print("Testing yC_open vs yC_zupt concept...")
