"""Full-data Snap ML GFP + XGBoost with the FROZEN settings (3-day windows, cycles on),
streaming transactions to Snap ML in time order so memory stays bounded.
Features are causal: each transaction only sees transactions up to that moment.
Every view (silo and combined) uses the same streaming method."""
import sys, time
sys.path.insert(0, '/home/ibm-user/shared')
from mulenet.gfp import GFP_PARAMS, run_gfp_xgb

# 2) full run
OUT = '/home/ibm-user/shared/results/scores'
t0 = time.time()
for setting, scope in [('combined', 'all'), ('silo', 'all'), ('combined', 'own')]:
    run_gfp_xgb(setting, sample=False, out_dir=OUT, label_scope=scope,
                params=GFP_PARAMS, suffix='_stream', stream=True)
    print(f'--- {setting}/{scope} done at {(time.time() - t0) / 60:.0f} min', flush=True)
print('ALL DONE', flush=True)
