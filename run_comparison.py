"""Paper-inspired KNN versus GA-KNN experiment. See results.json for protocol.
Run: python run_comparison.py --data path/to/heart_statlog_cleveland_hungary_final.csv
Requires numpy, pandas, scipy, scikit-learn, threadpoolctl.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from threadpoolctl import threadpool_limits

METRICS = ['euclidean', 'manhattan', 'chebyshev']
SPACE = [(k, m) for k in range(1, 52) for m in range(3)]

def model(chromosome):
    k, m = chromosome
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(k), metric=METRICS[int(m)], algorithm='brute', n_jobs=1))

def genetic_search(score, seed):
    rng = np.random.default_rng(seed)
    population = [(int(rng.integers(1, 52)), int(rng.integers(3))) for _ in range(50)]
    population[0] = (5, 0)  # Include the baseline as a candidate, without using test data.
    history = []
    for _ in range(100):
        fitness = np.array([score(c) for c in population])
        best = population[int(np.argmax(fitness))]
        history.append(float(fitness.max()))
        new = [best]
        def parent():
            ix = rng.integers(0, len(population), size=3)
            return population[int(ix[np.argmax(fitness[ix])])]
        while len(new) < 50:
            a, b = parent(), parent()
            c = [a[0], b[1]] if rng.random() < .8 else list(a)
            if rng.random() < .1:
                gene = int(rng.integers(2))
                c[gene] = int(rng.integers(1, 52)) if gene == 0 else int(rng.integers(3))
            new.append(tuple(c))
        population = new
    return max(population, key=score), history

def evaluate(chromosome, Xtr, Xte, ytr, yte):
    estimator = model(chromosome)
    start = time.perf_counter()
    estimator.fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = estimator.predict(Xte)
    predict_seconds = time.perf_counter() - start
    prob = estimator.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    return {'accuracy': accuracy_score(yte, pred), 'precision': precision_score(yte, pred, zero_division=0),
            'recall': recall_score(yte, pred), 'f1': f1_score(yte, pred), 'roc_auc': roc_auc_score(yte, prob),
            'specificity': tn / (tn + fp), 'balanced_accuracy': balanced_accuracy_score(yte, pred),
            'confusion_matrix': confusion_matrix(yte, pred).tolist(), 'fit_seconds': fit_seconds,
            'predict_seconds': predict_seconds,
            'k': int(chromosome[0]), 'distance': METRICS[chromosome[1]]}, pred.tolist(), prob.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(Path(__file__).resolve().parent / 'heart_statlog_cleveland_hungary_final.csv'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    path = Path(args.data)
    raw = pd.read_csv(path)
    if raw.isna().any().any() or set(raw.target.unique()) != {0, 1}:
        raise ValueError('Expected the complete numeric binary-classification dataset supplied with this project.')
    if len(raw.columns) != 12 or not all(pd.api.types.is_numeric_dtype(t) for t in raw.dtypes):
        raise ValueError('Expected 11 numeric input columns and target.')
    data = raw.drop_duplicates().reset_index(drop=True)
    X, y = data.drop(columns='target'), data.target
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, predictions = [], []
    for seed in range(10):
        train, test = train_test_split(np.arange(len(data)), test_size=.2, stratify=y, random_state=seed)
        Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
        cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr, ytr))
        cache = {}
        def score(c):
            c = tuple(c)
            if c not in cache:
                cache[c] = float(cross_val_score(model(c), Xtr, ytr, cv=cv, scoring='accuracy', n_jobs=1).mean())
            return cache[c]
        start = time.perf_counter()
        chosen, history = genetic_search(score, seed)
        ga_time = time.perf_counter() - start
        ga_evals = len(cache)
        # Independent caches keep the control search times comparable.
        ga_cv_accuracy = score(chosen)
        cache = {}
        rng = np.random.default_rng(seed)
        random_candidates = [SPACE[i] for i in rng.choice(len(SPACE), size=50, replace=False)]
        start = time.perf_counter()
        random_best = max(random_candidates, key=score)
        random_time = time.perf_counter() - start
        random_evals = len(cache)
        random_cv_accuracy = score(random_best)
        cache = {}
        start = time.perf_counter()
        grid_best = max(SPACE, key=score)
        grid_time = time.perf_counter() - start
        row = {'seed': seed, 'train_size': len(train), 'test_size': len(test),
               'ga_search_seconds': ga_time, 'ga_unique_candidates': ga_evals,
               'random_search_seconds': random_time, 'grid_search_seconds': grid_time,
               'random_unique_candidates': random_evals, 'grid_unique_candidates': len(cache),
               'ga_cv_accuracy': ga_cv_accuracy, 'random_cv_accuracy': random_cv_accuracy,
               'grid_cv_accuracy': score(grid_best),
               'ga_matched_grid_cv': bool(np.isclose(ga_cv_accuracy, score(grid_best))),
               'train_row_indices_deduplicated': train.tolist(),
               'test_row_indices_deduplicated': test.tolist(),
               'ga_convergence': history}
        for name, c in [('KNN', (5, 0)), ('GA_KNN', chosen), ('RandomSearch_KNN', random_best), ('GridSearch_KNN', grid_best)]:
            row[name], pred, prob = evaluate(c, Xtr, Xte, ytr, yte)
            search_seconds = {'KNN': 0., 'GA_KNN': ga_time, 'RandomSearch_KNN': random_time, 'GridSearch_KNN': grid_time}[name]
            row[name]['search_seconds'] = search_seconds
            row[name]['total_development_seconds'] = search_seconds + row[name]['fit_seconds']
            predictions.append({'seed': seed, 'model': name, 'test_row_indices_deduplicated': test.tolist(),
                                'truth': yte.tolist(), 'predictions': pred, 'positive_class_probabilities': prob})
        records.append(row)
        print(json.dumps({'seed': seed, 'knn': row['KNN']['accuracy'], 'ga_knn': row['GA_KNN']['accuracy'],
                          'ga_k': chosen[0], 'ga_distance': METRICS[chosen[1]], 'search_seconds': ga_time}), flush=True)
    names = ['KNN', 'GA_KNN', 'RandomSearch_KNN', 'GridSearch_KNN']
    summary = {name: {metric: {'mean': float(np.mean([r[name][metric] for r in records])),
                              'sd': float(np.std([r[name][metric] for r in records], ddof=1))}
                      for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity', 'balanced_accuracy', 'fit_seconds', 'predict_seconds', 'search_seconds', 'total_development_seconds']} for name in names}
    delta = np.array([r['GA_KNN']['accuracy'] - r['KNN']['accuracy'] for r in records])
    result = {'paper': 'https://doi.org/10.1016/j.mex.2025.103510',
              'dataset_url': 'https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final',
              'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'raw_rows': len(raw), 'duplicate_rows_removed': len(raw)-len(data), 'unique_rows': len(data),
              'features': X.columns.tolist(), 'missing_values': int(data.isna().sum().sum()),
              'zero_cholesterol_rows_retained': int((data.cholesterol == 0).sum()),
              'zero_resting_bp_rows_retained': int((data['resting bp s'] == 0).sum()),
              'class_counts': {str(k): int(v) for k,v in y.value_counts().items()},
              'protocol': 'Paper-inspired implementation, not exact numerical replication. Exact duplicate rows removed before splitting. Ten predefined stratified 80/20 holdouts, seeds 0-9; inner stratified shuffled 5-fold CV selects parameters using accuracy. StandardScaler fitted separately inside each CV fold; existing numeric category codes and zero measurements retained. Uniform KNN voting. Baseline k=5, Euclidean. Search k=1..51; Euclidean, Manhattan, Chebyshev distances. GA population 50, generations 100, tournament size 3, crossover probability .8, chromosome mutation probability .1, single elite, baseline included in initial population. Duplicate parameter evaluations cached. Search ranges, preprocessing and operator details are explicit implementation choices. Random search 50 distinct candidates; grid search all 153. Repeated holdouts overlap, so their standard deviation is descriptive and not an independent-sample significance test.',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(), 'machine': platform.machine(), 'logical_cpu_count': __import__('os').cpu_count(), 'thread_limit': 1},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'summary': summary, 'ga_vs_knn': {'mean_accuracy_change_percentage_points': float(delta.mean()*100),
              'wins': int((delta > 0).sum()), 'ties': int((delta == 0).sum()), 'losses': int((delta < 0).sum())},
              'runs': records}
    (output/'results.json').write_text(json.dumps(result, indent=2))
    (output/'predictions.json').write_text(json.dumps(predictions))
    print(json.dumps({'summary': summary, 'ga_vs_knn': result['ga_vs_knn']}, indent=2), flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
