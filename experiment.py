import gc
gc.set_threshold(0)

import os
default_n_threads = 4
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler, ADASYN as ADASYN_IMB
from ssmote import Mixup
from ssmote import SMOTE, BorderlineSMOTE
from ssmote import SimplicialSMOTE, BorderlineSimplicialSMOTE

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GB

from imblearn.pipeline import Pipeline

from sklearn.model_selection import ParameterGrid, GridSearchCV, RepeatedStratifiedKFold

from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import make_scorer

from tqdm import tqdm
from datetime import datetime

import pickle

# set print options
np.set_printoptions(precision=4, linewidth=150)

# get date and time
now = datetime.now()
date = now.strftime("%Y-%m-%d")
time = now.strftime("%H-%M-%S")

scaler = StandardScaler()

# imbalanced-learn (RO, ADASYN)
sampler_random = RandomOverSampler()
sampler_adasyn_imb = ADASYN_IMB()

# smote-variants (MWMOTE, DBSMOTE, LVQ-SMOTE)
from smote_variants.classifiers import OversamplingClassifier

# simplicial-smote (Mixup, SMOTE, B-SMOTE, σSMOTE, B-σSMOTE)
sampler_mixup = Mixup()
sampler_smote = SMOTE()
sampler_ssmote = SimplicialSMOTE()
sampler_b_smote = BorderlineSMOTE()
sampler_b_ssmote = BorderlineSimplicialSMOTE()

# classifiers
classifiers = {
    "knn": KNN(),
    "gb": GB(max_depth=2, n_estimators=20)
}

classifiers_var = {
    "knn": ("sklearn.neighbors", "KNeighborsClassifier", {}),
    "gb": ("sklearn.ensemble", "GradientBoostingClassifier", { "max_depth": 2, "n_estimators": 20 })
}

clf_name = "knn"
clf = classifiers[clf_name]

# methods
methods = {
    "imbalanced": Pipeline([("scaler", scaler), ("clf", clf)]),
    "random": Pipeline([("scaler", scaler), ("sampler", sampler_random), ("clf", clf)]),
    "mixup": Pipeline([("scaler", scaler), ("sampler", sampler_mixup), ("clf", clf)]),
    "safelevel_var": ("smote_variants", "Safe_Level_SMOTE", {}),
    "adasyn_imb": Pipeline([("scaler", scaler), ("sampler", sampler_adasyn_imb), ("clf", clf)]),
    "mwmote_var": ("smote_variants", "MWMOTE", {}),
    "dbsmote_var": ("smote_variants", "DBSMOTE", {}),
    "lvq_var": ("smote_variants", "LVQ_SMOTE", {}),
    "smote": Pipeline([("scaler", scaler), ("sampler", sampler_smote), ("clf", clf)]),
    "b_smote": Pipeline([("scaler", scaler), ("sampler", sampler_b_smote), ("clf", clf)]),
    "ssmote": Pipeline([("scaler", scaler), ("sampler", sampler_ssmote), ("clf", clf)]),
    "b_ssmote": Pipeline([("scaler", scaler), ("sampler", sampler_b_ssmote), ("clf", clf)])
}

# metrics
scoring = {
    "f1": make_scorer(f1_score),
    "matthews_corrcoef": make_scorer(matthews_corrcoef)
}

# datasets
PATH = os.path.dirname(os.path.realpath(__file__))
f = h5py.File(PATH + "/data/datasets.hdf5", "r")
datasets = list(f["/"].attrs["datasets"])

# random state
random_state = 0
score_refit = "f1"

# outer/inner CVs parameters
n_train_splits, n_train_repeats = 4, 25
n_test_splits, n_test_repeats = 5, 5

# hyperparameters
ks = [3, 5, 7, 9, 11]
ps = [list(range(3, k+1)) for k in ks]
params_max = [{"sampler__k": [k], "sampler__p": ps[i]} for i, k in enumerate(ks)]

n_datasets = len(datasets)
n_methods = len(methods)
n_metrics = len(scoring)
n_grid_params = len(list(ParameterGrid(params_max)))

j_idx = {8: 0, 9: 1, 10: 2, 11: 3}
train_val_scores = np.zeros((n_datasets, 4, n_test_splits * n_test_repeats, n_train_splits * n_train_repeats, n_metrics, n_grid_params))
best_params = [[[{} for k in range(n_test_splits * n_test_repeats)] for j in range(n_methods)] for i in range(n_datasets)]
scores = np.zeros((n_datasets, n_methods, n_test_splits * n_test_repeats, n_metrics))

# for each dataset
for i, dataset in enumerate(pbar:=tqdm(datasets)):

    X, y = f["/" + dataset + "/x"][:], f["/" + dataset + "/y"][:]
    y[y==-1] = 0 # fix

    pbar.set_description("{:16} 3-{}".format(dataset, ks[-1]))

    # outer/inner CVs
    cv_outer = RepeatedStratifiedKFold(n_splits=n_test_splits, n_repeats=n_test_repeats, random_state=random_state)
    cv_inner = RepeatedStratifiedKFold(n_splits=n_train_splits, n_repeats=n_train_repeats, random_state=random_state)

    for t_idx, (train_val_idx, test_idx) in enumerate(cv_outer.split(X, y)):

        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # for each method
        for j, m in enumerate(methods):
            
            # pipeline, for smote-variants
            if m.split("_")[-1]=="var":
                sampler_ = methods[m]
                if clf_name!="knn": # classifer randomness
                    clf_boilerplate = classifiers_var[clf_name]
                    classifier_ = (clf_boilerplate[0], clf_boilerplate[1], {**clf_boilerplate[2], "random_state": random_state})
                else:
                    classifier_ = classifiers_var[clf_name]
                pipeline = Pipeline([("scaler", scaler), ("sampler_clf", OversamplingClassifier(sampler_, classifier_))])
            
            # pipeline, for imblearn/ssmote
            else:
                pipeline = methods[m]

            # pipeline parameters, for smote-variants
            if m.split("_")[-1]=="var":
                sampler_boilerplate = methods[m]

                if m in ["adasyn_var", "safelevel_var", "mwmote_var", "lvq_var"]: # grid-search over k, sampler randomness
                    sampler_parameters = []
                    for k in ks:
                        sampler_parameters.append((sampler_boilerplate[0], sampler_boilerplate[1], {"n_neighbors": k, "random_state": random_state}))
                    params = {"sampler_clf__oversampler": sampler_parameters}
                
                elif m in ["dbsmote_var"]:
                    params = {"sampler_clf__oversampler": [(sampler_boilerplate[0], sampler_boilerplate[1], {"random_state": random_state})]}

            # pipeline params, for imblearn/ssmote
            else:
                if clf_name!="knn": # classifier randomness
                    pipeline.set_params(**{"clf__random_state": random_state})
                if m!="imbalanced": # sampler randomness
                    pipeline.set_params(**{"sampler__random_state": random_state})

                if m in ["smote", "b_smote"]:
                    params = [{"sampler__k": ks}]
                elif m in ["ssmote", "b_ssmote"]:
                    params = [{"sampler__k": [k], "sampler__p": ps[i]} for i, k in enumerate(ks)]
                elif m=="adasyn_imb":
                    params = [{"sampler__n_neighbors": ks}]
                else:
                    params = {} # imbalanced, random, mixup

            # fit on train+val
            pipeline_best = GridSearchCV(pipeline, params, scoring=scoring, refit=score_refit, cv=cv_inner, pre_dispatch=92*2, n_jobs=92)
            pipeline_best.fit(X_train_val, y_train_val)

            # save train+val scores and best_params for this methods only
            if m in ["smote", "b_smote", "ssmote", "b_ssmote"]:
            
                # update train+val scores of shape = n_datasets, n_methods, n_test_splits_repeats, n_train_splits_repeats, n_metrics, n_grid_params
                for tv_idx in range(n_train_splits * n_train_repeats):
                    for s_idx, key in enumerate(scoring):
                        test_score_tts = pipeline_best.cv_results_["split{}_test_{}".format(tv_idx, key)]
                        train_val_scores[i,j_idx[j],t_idx,tv_idx,s_idx,:] = np.pad(test_score_tts, (0, n_grid_params-len(test_score_tts)))

                # update best_params of shape = n_datasets, n_methods, n_test_splits_repeats
                best_params[i][j][t_idx] = pipeline_best.best_params_
            
            # eval on test, update test scores of shape
            # n_datasets, n_methods, n_test_splits_repeats, n_metrics
            for m_idx, key in enumerate(scoring):
                scores[i,j,t_idx,m_idx] = scoring[key](pipeline_best, X_test, y_test)

    # save results after completing each dataset
    np.save(open(PATH + "/data/cvn_test_{}_{}_{}_{}_{}-{}-{}-{}.npy".format(clf_name, score_refit, date, time, n_train_splits, n_train_repeats, n_test_splits, n_test_repeats), "wb"), scores)
    pickle.dump(train_val_scores, open(PATH + "/data/cvn_train_val_{}_{}_{}_{}_{}-{}-{}-{}.pkl".format(clf_name, score_refit, date, time, n_train_splits, n_train_repeats, n_test_splits, n_test_repeats), "wb"))
    pickle.dump(best_params, open(PATH + "/data/cvn_best_params_{}_{}_{}_{}_{}-{}-{}-{}.pkl".format(clf_name, score_refit, date, time, n_train_splits, n_train_repeats, n_test_splits, n_test_repeats), "wb"))

print("cvn_{}_{}_{}_{}_{}-{}-{}-{}.npy".format(clf_name, score_refit, date, time, n_train_splits, n_train_repeats, n_test_splits, n_test_repeats))
print("=======")

for i, dataset in enumerate(datasets):
    print(dataset.upper())
    print("-------")
    print("F1     ", scores[i,:,:,0].mean(axis=1))
    print("MAT COR", scores[i,:,:,1].mean(axis=1))
    print("\n")

print("-------")
print("F1 MEAN", scores[:,:,:,0].mean(axis=(0,2)))
print("MC MEAN", scores[:,:,:,1].mean(axis=(0,2)))

results_f1 = scores[:,:,:,0].mean(axis=2)
results_mc = scores[:,:,:,1].mean(axis=2)

print("-------")
print("F1 RANK", np.mean(results_f1.shape[1] - results_f1.argsort().argsort(), axis=0))
print("MC RANK", np.mean(results_mc.shape[1] - results_mc.argsort().argsort(), axis=0))