"""Check archived hyperparameter CSV consistency without running training."""
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
p = root / 'results' / 'hyperparameters'
subset = pd.read_csv(p / 'hyperparameter_experiment.csv')
full = pd.read_csv(p / 'hyperparameter_plots' / '00_summary' / 'all_results_ranked.csv')
keys = ['batch_size', 'hidden_size', 'learning_rate']
merged = subset.merge(full, on=keys, suffixes=('_150','_240'), validate='one_to_one', indicator=True)
assert len(subset) == 150 and len(full) == 240
assert len(merged) == 150 and (merged['_merge'] == 'both').all()
for field in ['initial_train_acc', 'initial_test_acc', 'final_train_acc',
              'final_test_acc', 'best_test_acc', 'best_epoch']:
    assert (merged[field+'_150'] == merged[field+'_240']).all(), field
n_different_times = int((merged['time_seconds_150'] != merged['time_seconds_240']).sum())
print('PASS: 150 configurations are a subset of the 240-grid results.')
print('PASS: all shared accuracy and best-epoch values match exactly.')
print(f'Note: time_seconds differs for {n_different_times} overlapping rows; both originals are preserved.')
print('Historical plot PNGs:',len(list((p/'hyperparameter_plots').rglob('*.png'))))
