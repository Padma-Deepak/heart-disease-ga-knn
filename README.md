# Heart Disease Classification: KNN and Genetic Algorithm

A reproducible mini project comparing **standard K-Nearest Neighbours (KNN)** with **Genetic Algorithm–optimized KNN (GA-KNN)** on a public heart-disease dataset.

The repository includes the experiment code, dataset, saved predictions, verification scripts, figures, a project report, and an optional **Fuzzy KNN / GA-Fuzzy KNN** extension.

**Main finding:** GA-KNN improves mean accuracy from **85.22% to 86.79%** across ten predefined splits. Grid search matches it. The experiment supports tuning the fixed KNN baseline; it does not establish an advantage unique to GA.

## Results

Means across ten stratified holdouts:

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Standard KNN | 85.22% | 0.8688 | 0.9024 |
| **GA-KNN** | **86.79%** | **0.8826** | 0.9223 |
| Random-search KNN | 86.63% | 0.8809 | **0.9235** |
| Grid-search KNN | **86.79%** | **0.8826** | 0.9223 |

GA-KNN has higher accuracy than standard KNN on **9 of 10 splits**, for a mean gain of **1.58 percentage points**.

![Mean performance](figures/performance_metrics.png)

![Accuracy across splits](figures/accuracy_by_split.png)

These are independently measured results. The selected journal paper reports 95.38% GA-KNN accuracy, which this implementation **does not reproduce**.

## Quick start

Use **Python 3.12**. Clone this repository, enter its folder, and create an isolated environment.

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Check the supplied saved results before changing them.
.venv/bin/python verify_results.py
.venv/bin/python fuzzy_extension/verify_fuzzy.py

# Repeat the main experiment and generate figures.
.venv/bin/python run_comparison.py
.venv/bin/python verify_results.py
.venv/bin/python generate_figures.py

# Optional: repeat the fuzzy extension against that new main run.
.venv/bin/python fuzzy_extension/run_fuzzy_comparison.py
.venv/bin/python fuzzy_extension/verify_fuzzy.py
```

If your Python 3.12 executable is named `python3`, use `python3 -m venv .venv` for the first command.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_comparison.py
.\.venv\Scripts\python.exe verify_results.py
.\.venv\Scripts\python.exe generate_figures.py
.\.venv\Scripts\python.exe fuzzy_extension/run_fuzzy_comparison.py
.\.venv\Scripts\python.exe fuzzy_extension/verify_fuzzy.py
```

The commands use the environment's Python explicitly, so activation is not required. If you see `ModuleNotFoundError` for Pandas or Matplotlib, check that you used `.venv/bin/python` (or the Windows equivalent), rather than macOS's `/usr/bin/python3`.

The dataset is included. Training requires no Kaggle login or network download.

## How the experiment works

1. Read the dataset and remove exact duplicate rows.
2. Create ten stratified 80/20 train/test splits using seeds 0–9.
3. Keep each test partition out of parameter selection.
4. Evaluate candidate settings using stratified five-fold cross-validation inside the training partition.
5. Refit the selected pipeline on all training records.
6. Evaluate standard KNN, GA-KNN, random search, and grid search on the same test records.
7. Save metrics, predictions, vote scores, selected settings, split indices, timings, and file fingerprints.

![Architecture](figures/overall_architecture.png)

### Dataset

Source: [Heart Disease Dataset (Comprehensive)](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final).

| Property | Value |
|---|---:|
| Downloaded records | 1,190 |
| Exact duplicates removed | 272 |
| Records used | 918 |
| Input features | 11 |
| Target | 0: disease absent; 1: disease present |
| Training / test records per split | 734 / 184 |

The inputs include age, encoded sex, chest-pain type, resting blood pressure, cholesterol, fasting-sugar indicator, resting ECG, maximum heart rate, exercise angina, oldpeak, and ST slope.

The experiment retains the original numeric category codes and zero-valued measurements. These are documented representation limitations. Scaling is fitted within each training fold through a scikit-learn pipeline.

### Genetic Algorithm

Each chromosome is `(neighbour_count, distance_index)`.

| Setting | Value |
|---|---|
| Neighbour count | 1–51 |
| Distances | Euclidean, Manhattan, Chebyshev |
| Distinct configurations | 153 |
| Population | 50, including the fixed baseline |
| Generations | 100 |
| Fitness | Mean inner five-fold accuracy |
| Selection | Tournament of 3 |
| Crossover probability | 0.8 |
| Mutation probability | 0.1 per child |
| Elitism | Preserve one best candidate |
| Repeated candidates | Cached within each search |

Standard KNN uses `k=5` and Euclidean distance. KNN comes from scikit-learn; the GA is implemented explicitly in this repository. GA selects KNN settings rather than changing the patient records or selecting features.

Random search evaluates 50 distinct configurations. Grid search evaluates all 153. Each method receives an independent cache; these are not equal-budget optimizer comparisons.

## Optional fuzzy extension

The extension implements fuzzy training memberships and distance-weighted membership aggregation, based on [Keller, Gray and Givens (1985)](https://doi.org/10.1109/TSMC.1985.6313426). GA additionally selects a fuzziness setting.

| Model | Mean accuracy |
|---|---:|
| Fixed Fuzzy KNN | 84.95% |
| GA-Fuzzy KNN | 85.76% |

The fuzzy hybrid improves on standard KNN but falls below the existing GA-KNN. It is **our extension**, not a fuzzy method from the selected 2025 paper.

See [the detailed fuzzy trial](fuzzy_extension/trial_results.txt).

## Repository guide

```text
.
├── run_comparison.py              # Main KNN / GA / random / grid experiment
├── verify_results.py              # Audit main saved results
├── generate_figures.py            # Generate the report's nine figures
├── requirements.txt              # Pinned environment dependencies
├── heart_statlog_cleveland_hungary_final.csv
├── results.json                  # Main settings, metrics, splits and timings
├── predictions.json              # Main test labels, predictions and scores
├── figures/                      # Charts and architecture diagrams
├── docs/
│   └── Mini_Project_Report.md     # Full assignment-format report
└── fuzzy_extension/
    ├── fuzzy_knn.py               # Fuzzy classifier
    ├── run_fuzzy_comparison.py    # Fuzzy experiments
    ├── verify_fuzzy.py            # Mathematical checks and output audit
    ├── fuzzy_results.json
    ├── fuzzy_predictions.json
    └── trial_results.txt
```

In [run_comparison.py](run_comparison.py), start with:

- `model()`: scaling and KNN configuration.
- `genetic_search()`: evolutionary search.
- `score()` inside `main()`: training-only cross-validation fitness.
- `evaluate()`: final fit, predictions, metrics, and timings.

## Reproducibility and verification

The verification scripts check source/data fingerprints, train/test boundaries, prediction alignment, confusion matrices, metrics, and summary statistics. Fuzzy checks also include hand-calculated membership examples.

Rerunning the main experiment overwrites `results.json` and `predictions.json`. Rerun the fuzzy experiment afterward before verifying the fuzzy outputs: they record the fingerprint of the main result they used. A main rerun changes that fingerprint even if predictive metrics remain the same, because timings are recorded.

To write a main run elsewhere:

```bash
.venv/bin/python run_comparison.py --output rerun
```

The figure and verification scripts read files beside themselves. The supplied report records the delivered run; its tables are not automatically rewritten after new runs.

GitHub Actions checks the saved outputs and repeats both experiments on pushes and pull requests.

## Interpretation and limitations

- This is a **paper-inspired implementation**, not the authors' original code or an exact numerical replication.
- Repeated holdouts overlap. Their standard deviations and win counts are descriptive, not independent-sample significance tests.
- No external clinical validation, patient-identity audit, calibrated medical probabilities, or deployment evaluation is claimed.
- The code classifies recorded disease labels; it does not establish when someone will develop a future cardiac event.
- Grid search matches the main hybrid's accuracy, so an advantage unique to GA is not demonstrated.

## References and attribution

**Selected paper:** S. H. Murad, N. B. Tayfor, N. H. Mahmood and L. Arman, “Hybrid genetic algorithms-driven optimization of machine learning models for heart disease prediction,” *MethodsX*, vol. 15, article 103510, 2025. [DOI](https://doi.org/10.1016/j.mex.2025.103510).

**Dataset:** [Heart Disease Dataset (Comprehensive), Kaggle](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final), publicly listed under **CC0**.

**Fuzzy method:** J. M. Keller, M. R. Gray and J. A. Givens, “A fuzzy K-nearest neighbor algorithm,” *IEEE Transactions on Systems, Man, and Cybernetics*, 15(4), 580–585, 1985. [DOI](https://doi.org/10.1109/TSMC.1985.6313426).

**Software:** [scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Matplotlib](https://matplotlib.org/).

Original project code and documentation are distributed under the [MIT License](LICENSE). Dataset rights remain separate as described above; the cited papers are not relicensed by this project.
