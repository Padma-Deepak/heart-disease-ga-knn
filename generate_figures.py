"""Generate report figures from saved results, without retraining or selecting models."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
r = json.loads((ROOT / 'results.json').read_text())
p = json.loads((ROOT / 'predictions.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
COLORS = ['#425d7b', '#158078', '#bd8748', '#858295']
NAMES = ['KNN', 'GA_KNN', 'RandomSearch_KNN', 'GridSearch_KNN']
LABELS = ['Standard KNN', 'GA-KNN', 'Random search', 'Grid search']

def save(fig, name):
    fig.savefig(FIG / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def architecture(name, title, boxes):
    fig, ax = plt.subplots(figsize=(10, 1.55 * len(boxes)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(boxes) * 1.55)
    ax.axis('off')
    ax.set_title(title, fontweight='bold', pad=15)
    for i, text in enumerate(boxes):
        y = (len(boxes)-i-1)*1.55 + .25
        ax.add_patch(FancyBboxPatch((.7, y), 8.6, .95, boxstyle='round,pad=.08', facecolor='#eef4f6', edgecolor='#158078'))
        ax.text(5, y+.475, text, ha='center', va='center', fontsize=11)
        if i < len(boxes)-1:
            ax.annotate('', xy=(5,y-.38), xytext=(5,y-.09), arrowprops={'arrowstyle':'->', 'color':'#425d7b', 'lw':1.7})
    save(fig, name)

architecture('overall_architecture.png', 'Overall experimental architecture', [
    'Public dataset: 1,190 rows and 11 input features',
    'Remove 272 exact duplicate rows → 918 unique rows',
    'Ten stratified 80/20 splits, seeds 0–9: 734 training / 184 test',
    'TRAINING ONLY: fixed KNN / GA search / random search / grid search\nSearch candidates use the same inner 5-fold CV and fold-specific scaling',
    'Refit each selected pipeline on all 734 training records',
    'Evaluate once on the shared 184-record test partition\nSave metrics, probabilities, predictions, timings and split indices'])
architecture('nonhybrid_architecture.png', 'Non-hybrid system: standard KNN', [
    '734 training records: fit StandardScaler',
    'Store standardized training records and labels\nFixed KNN: k = 5, Euclidean distance, uniform vote',
    '184 held-out records: transform using the training scaler',
    'Find five nearest training neighbours and predict each class\nSave class predictions and positive-class vote proportions'])
architecture('hybrid_architecture.png', 'Hybrid system: genetic algorithm + KNN', [
    'Training data only: initialize 50 chromosomes (k, distance)',
    'Score candidates using inner 5-fold cross-validation\nFit scaling separately within each fold; cache repeated candidates',
    'Tournament selection → crossover → mutation → one elite\nRepeat for 100 generations',
    'Choose the chromosome with highest inner-CV accuracy',
    'Refit StandardScaler + selected KNN on the full training partition',
    'Predict the same untouched test partition used by standard KNN'])

fig, ax = plt.subplots(figsize=(10, 4.8))
metrics = ['accuracy','precision','recall','f1','roc_auc']
positions = np.arange(len(metrics))
for i, name in enumerate(NAMES[:2]):
    means = [r['summary'][name][m]['mean']*100 for m in metrics]
    sds = [r['summary'][name][m]['sd']*100 for m in metrics]
    ax.bar(positions + (i-.5)*.34, means, .34, yerr=sds, capsize=3, color=COLORS[i], label=LABELS[i])
ax.set(xticks=positions, xticklabels=['Accuracy','Precision','Recall','F1 × 100','AUC × 100'], ylabel='Score (%)', ylim=(0, 100), title='Mean test performance across ten splits')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=.2)
fig.text(.13,-.02,'Error bars: sample SD across overlapping splits; these are not confidence intervals.',fontsize=9)
save(fig,'performance_metrics.png')

fig, ax = plt.subplots(figsize=(9,4.8))
for name,color,label in zip(NAMES[:2],COLORS,LABELS):
    ax.plot(range(10),[x[name]['accuracy']*100 for x in r['runs']], marker='o', lw=2, color=color,label=label)
ax.set(xlabel='Predefined split seed',ylabel='Test accuracy (%)',xticks=range(10),ylim=(78,94),title='GA-KNN versus standard KNN on identical test partitions')
ax.legend();ax.grid(alpha=.2)
save(fig,'accuracy_by_split.png')

fig, axes = plt.subplots(1,2,figsize=(9,4.5))
for ax,name,label in zip(axes,NAMES[:2],LABELS):
    cm=np.array(r['runs'][0][name]['confusion_matrix'])
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=102)
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,color='white' if cm[i,j]>55 else '#233')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No disease','Disease'],yticklabels=['No disease','Disease'],xlabel='Predicted class',ylabel='Actual class',title=label)
fig.suptitle('Confusion matrices: seed 0 (184 held-out records)',fontweight='bold')
fig.tight_layout()
save(fig,'confusion_matrices_seed0.png')

fig,ax=plt.subplots(figsize=(6.8,5.5))
for name,color,label in zip(NAMES[:2],COLORS,LABELS):
    item=next(x for x in p if x['seed']==0 and x['model']==name)
    fpr,tpr,_=roc_curve(item['truth'],item['positive_class_probabilities'])
    ax.plot(fpr,tpr,color=color,lw=2,label=f"{label}: AUC {r['runs'][0][name]['roc_auc']:.4f}")
ax.plot([0,1],[0,1],'--',color='#999')
ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC curves: illustrative split, seed 0',xlim=(0,1),ylim=(0,1.02))
ax.legend(loc='lower right');ax.grid(alpha=.2)
save(fig,'roc_seed0.png')

fig,ax=plt.subplots(figsize=(9,4.5))
for run in r['runs']:
    ax.plot(range(1,101),np.array(run['ga_convergence'])*100,alpha=.5,lw=1,label=f"Seed {run['seed']}")
ax.set(xlabel='GA generation',ylabel='Best inner-CV accuracy (%)',title='GA convergence on training data only')
ax.legend(ncol=5,fontsize=8,loc='lower right');ax.grid(alpha=.2)
save(fig,'ga_convergence.png')

fig,ax=plt.subplots(figsize=(9,4.8))
means=[r['summary'][name]['total_development_seconds']['mean'] for name in NAMES]
bars=ax.bar(LABELS,means,color=COLORS)
for bar,mean in zip(bars,means): ax.text(bar.get_x()+bar.get_width()/2,mean+max(means)*.025,f'{mean:.4f} s',ha='center')
ax.set(ylabel='Mean development time (seconds)',title='Parameter search plus final fit; single-thread execution',ylim=(0,max(means)*1.2))
ax.grid(axis='y',alpha=.2)
save(fig,'development_time.png')
print('Generated 9 report figures.')
