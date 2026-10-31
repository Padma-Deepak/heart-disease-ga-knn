"""Exploratory fuzzy extension; preserves the original experiment and report.

Run from any directory: python run_fuzzy_comparison.py
Requires the parent folder's supplied dataset, results.json and requirements.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from threadpoolctl import threadpool_limits
from fuzzy_knn import FuzzyKNNClassifier, training_memberships, nearest_geometry, combine_memberships

ROOT = Path(__file__).resolve().parent
METRICS = ['euclidean', 'manhattan', 'chebyshev']
FUZZINESS = [1.5, 2., 2.5, 3., 4.]
BOUNDS = [51, 3, 5]
BASELINE = (5, 0, 1)

def genetic_search(score, seed):
    rng = np.random.default_rng(seed)
    def random_chromosome():
        return (int(rng.integers(1, 52)), int(rng.integers(3)), int(rng.integers(5)))
    pop = [random_chromosome() for _ in range(50)]
    pop[0] = BASELINE
    history = []
    for generation in range(100):
        fitness = np.array([score(c) for c in pop])
        history.append(float(fitness.max()))
        new = [pop[int(np.argmax(fitness))]]
        def parent():
            ix = rng.integers(0, 50, 3)
            return pop[int(ix[np.argmax(fitness[ix])])]
        while len(new) < 50:
            a, b = parent(), parent()
            if rng.random() < .8:
                cut = int(rng.integers(1, 3))
                child = list(a[:cut]+b[cut:])
            else:
                child = list(a)
            if rng.random() < .1:
                j = int(rng.integers(3))
                child[j] = int(rng.integers(1, 52)) if j == 0 else int(rng.integers(BOUNDS[j]))
            new.append(tuple(child))
        pop = new
    return max(pop, key=score), history

def parameters(c):
    return {'n_neighbors': int(c[0]), 'metric': METRICS[c[1]], 'fuzziness': FUZZINESS[c[2]], 'init_k': 5}

def pipeline(c):
    return make_pipeline(StandardScaler(), FuzzyKNNClassifier(**parameters(c)))

def measure(truth, pred, scores):
    return {'accuracy': accuracy_score(truth,pred), 'precision': precision_score(truth,pred),
            'recall': recall_score(truth,pred), 'f1': f1_score(truth,pred),
            'roc_auc': roc_auc_score(truth,scores), 'confusion_matrix': confusion_matrix(truth,pred).tolist()}

def main():
    data_path = ROOT.parent/'heart_statlog_cleveland_hungary_final.csv'
    original = json.loads((ROOT.parent/'results.json').read_text())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == original['dataset_sha256']
    data = pd.read_csv(data_path).drop_duplicates().reset_index(drop=True)
    X = data.drop(columns='target').to_numpy(dtype=float)
    y = data.target.to_numpy(dtype=int)
    runs, predictions = [], []
    for old in original['runs']:
        seed = old['seed']
        train, test = old['train_row_indices_deduplicated'], old['test_row_indices_deduplicated']
        assert not set(train).intersection(test)
        Xtr, Xte, ytr, yte = X[train], X[test], y[train], y[test]
        cv = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr,ytr))
        start = time.perf_counter()
        folds = []
        for inner_train, validation in cv:
            scaler = StandardScaler().fit(Xtr[inner_train])
            A, B = scaler.transform(Xtr[inner_train]), scaler.transform(Xtr[validation])
            geometry = []
            for metric in METRICS:
                U = training_memberships(A, ytr[inner_train], metric, init_k=5)
                dist, idx = nearest_geometry(A, B, metric, max_k=51)
                geometry.append((dist,idx,U))
            folds.append((geometry, ytr[validation]))
        preparation_seconds = time.perf_counter()-start
        cache = {}
        def score(c):
            if c not in cache:
                values = []
                for geometry, labels in folds:
                    distances, indices, memberships = geometry[c[1]]
                    u = combine_memberships(distances, indices, memberships, c[0], FUZZINESS[c[2]])
                    values.append(float(np.mean(np.argmax(u, axis=1) == labels)))
                cache[c] = float(np.mean(values))
            return cache[c]
        # Independently confirm the accelerated fitness equals fresh pipeline CV.
        for c in [BASELINE, (13,1,4)]:
            direct = cross_val_score(pipeline(c), Xtr, ytr, cv=cv, scoring='accuracy', n_jobs=1).mean()
            assert np.isclose(direct, score(c)), (seed,c,direct,score(c))
        start = time.perf_counter()
        chosen, history = genetic_search(score,seed)
        search_seconds = time.perf_counter()-start
        ga_evals = len(cache)
        # Exhaustive training-only reference, evaluated AFTER freezing the GA choice.
        full_space = [(k,mi,fi) for k in range(1,52) for mi in range(3) for fi in range(5)]
        grid_best = max(full_space,key=score)
        row = {'seed':seed,'train_size':len(train),'test_size':len(test),
               'train_indices':train,'test_indices':test,'geometry_preparation_seconds':preparation_seconds,
               'ga_search_seconds_excluding_preparation':search_seconds, 'ga_unique_candidates':ga_evals,
               'ga_cv_accuracy':score(chosen),'grid_cv_accuracy':score(grid_best),
               'ga_matched_grid_cv':bool(np.isclose(score(chosen),score(grid_best))),
               'ga_convergence':history}
        for name,c in [('Fuzzy_KNN',BASELINE),('GA_Fuzzy_KNN',chosen),('Grid_Fuzzy_KNN',grid_best)]:
            estimator = pipeline(c).fit(Xtr,ytr)
            u = estimator.predict_proba(Xte)
            pred = np.argmax(u,axis=1)
            assert np.allclose(u.sum(axis=1),1) and np.all((u>=-1e-12)&(u<=1+1e-12))
            U = estimator[-1].memberships_
            assert np.allclose(U.sum(axis=1),1)
            assert np.all(U[np.arange(len(ytr)),ytr] >= .51-1e-12)
            row[name] = {**measure(yte,pred,u[:,1]),**parameters(c)}
            predictions.append({'seed':seed,'model':name,'test_indices':test,'truth':yte.tolist(),
                                'predictions':pred.tolist(),'memberships':u.tolist()})
        for name in ['KNN','GA_KNN']:
            row[name] = old[name]
        runs.append(row)
        print(json.dumps({'seed':seed,'KNN':row['KNN']['accuracy'],'GA_KNN':row['GA_KNN']['accuracy'],
                          'Fuzzy_KNN':row['Fuzzy_KNN']['accuracy'],'GA_Fuzzy_KNN':row['GA_Fuzzy_KNN']['accuracy'],
                          'chosen':parameters(chosen),'matched_grid_cv':row['ga_matched_grid_cv']}),flush=True)
    names = ['KNN','GA_KNN','Fuzzy_KNN','GA_Fuzzy_KNN','Grid_Fuzzy_KNN']
    summary = {name:{metric:{'mean':float(np.mean([r[name][metric] for r in runs])),
                            'sd':float(np.std([r[name][metric] for r in runs],ddof=1))}
                     for metric in ['accuracy','precision','recall','f1','roc_auc']} for name in names}
    comparisons={}
    for baseline in ['KNN','GA_KNN','Fuzzy_KNN']:
        delta=np.array([r['GA_Fuzzy_KNN']['accuracy']-r[baseline]['accuracy'] for r in runs])
        comparisons[baseline]={'mean_accuracy_change_percentage_points':float(delta.mean()*100),
                               'wins':int((delta>0).sum()),'ties':int((delta==0).sum()),'losses':int((delta<0).sum())}
    result={'protocol':'Exploratory extension using the exact original ten outer splits, already inspected in earlier work. StandardScaler and training fuzzy memberships are refitted inside every inner-CV training fold. Initialization k=5 excluding self, U=.51*onehot(y)+.49*neighbour frequencies. Query weighting d**(-2/(m-1)), normalized in log space, zero-distance memberships averaged. k=1..51, three distances, fuzziness in [1.5,2,2.5,3,4]; fixed fuzzy baseline k5/Euclidean/m2. GA pop50/gen100/tournament3/crossover.8/mutation.1/one elite. Grid control spans all 765 candidates and never changes GA choice. Training-fold distance geometry is reused to accelerate scoring; verified against independent pipeline CV for two predetermined configurations on every split. Hard class ties favour 0. Same numeric categories and retained zeros as the original experiment. Not a 2025 paper replication and not a fresh confirmatory test.',
            'fuzzy_reference':'https://doi.org/10.1109/TSMC.1985.6313426',
            'dataset_sha256':original['dataset_sha256'],'original_results_sha256':hashlib.sha256((ROOT.parent/'results.json').read_bytes()).hexdigest(),
            'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['fuzzy_knn.py','run_fuzzy_comparison.py']},
            'summary':summary,'ga_fuzzy_comparisons':comparisons,'runs':runs}
    (ROOT/'fuzzy_results.json').write_text(json.dumps(result,indent=2))
    (ROOT/'fuzzy_predictions.json').write_text(json.dumps(predictions))
    print(json.dumps({'summary':summary,'comparisons':comparisons},indent=2),flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
