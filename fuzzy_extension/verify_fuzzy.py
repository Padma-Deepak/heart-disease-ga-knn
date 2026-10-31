"""Hand-calculated fuzzy-classifier checks plus independent saved-output audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from fuzzy_knn import FuzzyKNNClassifier

root = Path(__file__).resolve().parent
# For x=2, the two other nearest training points are x=1/class0 and x=3/class1.
# Hence class memberships are [0.51+0.49/2, 0.49/2] = [0.755, 0.245].
X = np.arange(6).reshape(-1,1)
y = np.array([0,0,0,1,1,1])
model = FuzzyKNNClassifier(n_neighbors=2,init_k=2,fuzziness=2).fit(X,y)
assert np.allclose(model.memberships_[2],[.755,.245])
assert np.allclose(model.memberships_[3],[.245,.755])
# Query at midpoint sees the two boundary samples at equal distance.
assert np.allclose(model.predict_proba([[2.5]]),[[.5,.5]])
assert model.predict([[2.5]])[0] == 0
# At an exact training match, no division by zero and the stored membership returns.
assert np.allclose(model.predict_proba([[2.]]),[[.755,.245]])
# Query 2.25 sees x=2 and x=3 with distances .25 and .75: squared inverse weights 9:1.
assert np.allclose(model.predict_proba([[2.25]]), [[(.755*9+.245)/10, (.245*9+.755)/10]])
print('PASS: hand-calculated training memberships, midpoint, zero-distance and unequal-distance cases.')

if (root/'fuzzy_results.json').exists():
    r=json.loads((root/'fuzzy_results.json').read_text())
    p=json.loads((root/'fuzzy_predictions.json').read_text())
    old_path=root.parent/'results.json'
    assert hashlib.sha256(old_path.read_bytes()).hexdigest()==r['original_results_sha256']
    old=json.loads(old_path.read_text())
    path=root.parent/'heart_statlog_cleveland_hungary_final.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['dataset_sha256']
    data=pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    for name,expected in r['source_hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
    for run in r['runs']:
        original=old['runs'][run['seed']]
        assert run['train_indices']==original['train_row_indices_deduplicated']
        assert run['test_indices']==original['test_row_indices_deduplicated']
        assert not set(run['train_indices']).intersection(run['test_indices'])
        assert np.all(np.diff(run['ga_convergence'])>=-1e-12)
        assert 1<=run['GA_Fuzzy_KNN']['n_neighbors']<=51
        for name in ['Fuzzy_KNN','GA_Fuzzy_KNN','Grid_Fuzzy_KNN']:
            item=next(x for x in p if x['seed']==run['seed'] and x['model']==name)
            truth=data.target.iloc[item['test_indices']].to_numpy()
            U=np.array(item['memberships'])
            assert np.array_equal(truth,item['truth'])
            assert np.allclose(U.sum(axis=1),1) and np.all((U>=-1e-12)&(U<=1+1e-12))
            assert np.array_equal(np.argmax(U,axis=1),item['predictions'])
            pred=item['predictions']
            checks={'accuracy':accuracy_score(truth,pred),'precision':precision_score(truth,pred),
                    'recall':recall_score(truth,pred),'f1':f1_score(truth,pred),'roc_auc':roc_auc_score(truth,U[:,1])}
            for metric,value in checks.items(): assert np.isclose(value,run[name][metric])
            assert confusion_matrix(truth,pred).tolist()==run[name]['confusion_matrix']
    for name,summary in r['summary'].items():
        for metric,stats in summary.items():
            values=[run[name][metric] for run in r['runs']]
            assert np.isclose(np.mean(values),stats['mean'])
            assert np.isclose(np.std(values,ddof=1),stats['sd'])
    assert len(p)==30 and len(r['runs'])==10
    print('PASS: all ten identical outer partitions; all 30 fuzzy prediction sets, membership sums, metrics and source fingerprints; summary means and SDs.')
