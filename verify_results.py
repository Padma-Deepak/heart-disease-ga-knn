"""Audit saved experiment outputs and the dataset; no model training is performed."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score

root = Path(__file__).resolve().parent
r = json.loads((root/'results.json').read_text())
predictions = json.loads((root/'predictions.json').read_text())
data_path = root/'heart_statlog_cleveland_hungary_final.csv'
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == r['dataset_sha256']
assert hashlib.sha256((root/'run_comparison.py').read_bytes()).hexdigest() == r['code_sha256']
raw = pd.read_csv(data_path)
data = raw.drop_duplicates().reset_index(drop=True)
assert len(raw) == 1190 and len(data) == 918
assert len(predictions) == 40 and len(r['runs']) == 10
assert not data.drop(columns='target').duplicated().any(), 'Conflicting or repeated feature rows need a group-aware split.'
for run in r['runs']:
    train = set(run['train_row_indices_deduplicated'])
    test = set(run['test_row_indices_deduplicated'])
    assert len(train) == 734 and len(test) == 184
    assert not train.intersection(test) and train.union(test) == set(range(918))
    assert np.all(np.diff(run['ga_convergence']) >= -1e-12)
    assert run['ga_unique_candidates'] <= 153
    for name in r['summary']:
        record = run[name]
        item = next(x for x in predictions if x['seed'] == run['seed'] and x['model'] == name)
        assert item['test_row_indices_deduplicated'] == run['test_row_indices_deduplicated']
        truth = data.target.iloc[item['test_row_indices_deduplicated']].tolist()
        assert item['truth'] == truth
        prob = np.asarray(item['positive_class_probabilities'])
        assert len(prob) == len(truth) and np.all((prob >= 0) & (prob <= 1))
        assert np.array_equal((prob > .5).astype(int), item['predictions'])
        metrics = {
            'accuracy': accuracy_score(truth, item['predictions']),
            'precision': precision_score(truth, item['predictions']),
            'recall': recall_score(truth, item['predictions']),
            'f1': f1_score(truth, item['predictions']),
            'roc_auc': roc_auc_score(truth, prob),
            'balanced_accuracy': balanced_accuracy_score(truth, item['predictions']),
        }
        cm = confusion_matrix(truth, item['predictions'])
        metrics['specificity'] = cm[0,0] / cm[0].sum()
        assert cm.tolist() == record['confusion_matrix']
        for metric, value in metrics.items():
            assert np.isclose(value, record[metric]), (run['seed'], name, metric)
        assert np.isclose(record['total_development_seconds'], record['search_seconds'] + record['fit_seconds'])
for name, summary in r['summary'].items():
    for metric, statistics in summary.items():
        values = [run[name][metric] for run in r['runs']]
        assert np.isclose(np.mean(values), statistics['mean'])
        assert np.isclose(np.std(values, ddof=1), statistics['sd'])
delta = np.array([run['GA_KNN']['accuracy']-run['KNN']['accuracy'] for run in r['runs']])
assert (delta > 0).sum() == r['ga_vs_knn']['wins']
assert (delta < 0).sum() == r['ga_vs_knn']['losses']
assert np.isclose(delta.mean()*100, r['ga_vs_knn']['mean_accuracy_change_percentage_points'])
print('PASS: dataset/code hashes, all 10 split partitions, all 40 prediction sets, probabilities, metrics, confusion matrices, timings, convergence and summary statistics.')
