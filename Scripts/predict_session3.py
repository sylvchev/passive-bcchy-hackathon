# Stub for all steps from loading data to classification

from copy import deepcopy
from itertools import product
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from pyriemann.classification import FgMDM
from pyriemann.spatialfilters import SPoC
from pyriemann.spatialfilters import CSP as Pyr_CSP
from pyriemann.tangentspace import TangentSpace
from fc_helpers import EnsureSPD


mne.set_log_level("ERROR")

# Prelooading files
data_dir = "../Database/"
data_fc = "MNE_FC_AllSession_4Estim_2freqs.npz"
database = np.load(data_dir + data_fc, allow_pickle=True)
db = database["arr_0"].item()
data_raw = "MNE_Raw_3sessions.npy"
with open(data_dir + data_raw, "rb") as f:
    dbraw = np.load(f, allow_pickle=True).item()

level_values = [0, 1, 2]
diff = ["MATBeasy", "MATBmed", "MATBdiff"]
freqbands = {"gamma": [30, 45], "defaultBand": [8, 35]}
spectral_met = ["coh", "imcoh", "plv", "cov"]
n_subjects, n_sessions = 15, 3


class InputFeat(TransformerMixin, BaseEstimator):
    """Getting connectivity features from dict input"""

    def __init__(self, feat="raw"):
        self.feat = feat

    def fit(self, idx, y=None):
        pass

    def transform(self, idx):
        if idx.max() < lsess["S1"]:
            sess = "S1"
            lidx = np.array(idx)
        elif idx.min() >= lsess["S1"] and idx.max() < lsess["S1"] + lsess["S2"]:
            sess = "S2"
            lidx = idx - lsess["S1"]
        elif idx.min() >= lsess["S2"]:
            sess = "S3"
            lidx = idx - (lsess["S1"] + lsess["S2"])
        elif idx.min() < lsess["S1"] and idx.max() < lsess["S1"] + lsess["S2"]:
            # session 1 and 2
            lidx = np.array(idx)
            X1 = np.concatenate(Xsess["S1"][self.feat], axis=0)
            X2 = np.concatenate(Xsess["S2"][self.feat], axis=0)
            return np.concatenate([X1, X2], axis=0)[lidx]
        else:
            raise ValueError("Choose S1, S2 or S3 indexes")
        return np.concatenate(Xsess[sess][self.feat], axis=0)[lidx]

    def fit_transform(self, idx, y=None):
        return self.transform(idx)


# Pipelines
# ---------
pipelines = {}

# CSP-based
# =========
pipelines["CSP-SVM"] = make_pipeline(
    InputFeat("cov-defaultBand"),
    EnsureSPD(),
    Pyr_CSP(nfilter=6),
    SVC(kernel="rbf", C=0.1, probability=True),
)

# SPoC-based
# ==========
pipelines["SPoC-EN"] = make_pipeline(
    InputFeat("cov-defaultBand"),
    EnsureSPD(),
    SPoC(nfilter=4, metric="euclid", log=True),
    LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.1,
        C=0.1,
        multi_class="multinomial",
    ),
)

# FgMDM on Cov and Coh
# ====================
for fc in ["cov", "coh"]:
    for f in freqbands:
        pipelines["fgMDM-" + fc + "-" + f] = make_pipeline(
            InputFeat(fc + "-" + f),
            EnsureSPD(),
            FgMDM(metric="logeuclid", tsupdate=True),
        )

# ElasticNet on ImCoh and PLV
# ===========================
for fc in ["imcoh", "plv"]:
    for f in freqbands:
        pipelines["EN-" + fc + "-" + f] = make_pipeline(
            InputFeat(fc + "-" + f),
            EnsureSPD(),
            TangentSpace(metric="riemann"),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.1,
                C=0.1,
                multi_class="multinomial",
            ),
        )

# Ensemble
estimators = [(k, v) for k, v in pipelines.items()]
final_estimator = RidgeClassifier(class_weight="balanced")
cvkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scl = StackingClassifier(
    estimators=estimators,
    cv=cvkf,
    final_estimator=final_estimator,
    stack_method="predict_proba",
)
ens_ppl = {"ensemble": scl}
# pipelines["ensemble"] = scl

all_res = []
for subject in tqdm(range(1, n_subjects + 1), desc="subject"):
    sub = f"P{subject:02}"
    Xsess, ysess, lsess = {}, {}, {}
    for session in range(1, n_sessions + 1):
        sess = f"S{session}"
        Xsess[sess] = {
            f"{m}-{b}": [] for m, b in product(spectral_met, freqbands.keys())
        }
        Xsess[sess]["raw"] = []
        ysess[sess] = []
        if session == 3:
            trials = {}
            trials["raw"] = dbraw[sub][sess].get_data()
            for band, (fmin, fmax) in freqbands.items():
                for sm in spectral_met:
                    sname = f"{sub}-{sess}-{sm}-1-0.5-{fmin}-{fmax}"
                    trials[f"{sm}-{band}"] = db[sname]
            for sig in trials.keys():
                Xsess[sess][sig].append(trials[sig])
        else:
            for level, lv in zip(diff, level_values):
                # load data in a dict
                trials = {}
                trials["raw"] = dbraw[sub][sess][level].get_data()
                for band, (fmin, fmax) in freqbands.items():
                    for sm in spectral_met:
                        sname = f"{sub}-{sess}-{level}-{sm}-1-0.5-{fmin}-{fmax}"
                        trials[f"{sm}-{band}"] = db[sname]

                for sig in trials.keys():
                    Xsess[sess][sig].append(trials[sig])
                ysess[sess] += [lv] * len(trials["raw"])
        lsess[sess] = len(trials["raw"]) if session == 3 else len(ysess[sess])
    idx = np.arange(np.array([i for i in lsess.values()]).sum())

    # Train and evaluate
    for ppn, ppl in ens_ppl.items():
        X_idx_S1, X_idx_S2 = (
            idx[: lsess["S1"]],
            idx[lsess["S1"] : lsess["S1"] + lsess["S2"]],
        )
        X_idx_S12 = idx[: lsess["S1"] + lsess["S2"]]
        ysess12 = np.concatenate([ysess["S1"], ysess["S2"]])
        X_idx_S3 = idx[lsess["S1"] + lsess["S2"] :]
        ppl1, ppl2, ppl12 = deepcopy(ppl), deepcopy(ppl), deepcopy(ppl)

        # Train on sess 1 and 2, test on S3
        ppl12.fit(X_idx_S12, ysess12)
        yp12 = ppl12.predict(X_idx_S3)

        # save submission
        submission = pd.DataFrame({"epochID": np.arange(len(yp12)), "prediction": yp12})
        submission.to_csv(f"../Database/submission-{sub}.csv", header=True, index=False)

        # save distribution
        ce, cm, cd = (yp12 == 0).sum(), (yp12 == 1).sum(), (yp12 == 2).sum()
        res = {
            "subject": sub,
            "pipeline": ppn,
            "pred_easy": ce,
            "pred_med": cm,
            "pred_diff": cd,
        }
        all_res.append(res)
        for est_n, est_p in ppl2.named_estimators_.items():
            ype12 = est_p.predict(X_idx_S3)
            ce, cm, cd = (ype12 == 0).sum(), (ype12 == 1).sum(), (ype12 == 2).sum()
            res = {
                "subject": sub,
                "pred_easy": ce,
                "pred_med": cm,
                "pred_diff": cd,
                "pipeline": est_n,
            }
            all_res.append(res)

all_res = pd.DataFrame(all_res)
all_res.to_csv(data_dir + "classif_subsmission_session3.csv")
