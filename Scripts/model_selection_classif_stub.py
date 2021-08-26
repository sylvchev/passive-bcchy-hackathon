# Stub for all steps from loading data to classification

import configparser
from copy import deepcopy
from itertools import product
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP as MNE_CSP
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.spatialfilters import SPoC
from pyriemann.spatialfilters import CSP as Pyr_CSP
from pyriemann.tangentspace import TangentSpace
from fc_helpers import EnsureSPD


mne.set_log_level("ERROR")

# Prelooading files
data_dir = "../Database/"
data_fc = "MNE_FC_AllEstim_6Freqs.npz"
database = np.load(data_dir + data_fc, allow_pickle=True)
db = database["arr_0"].item()
data_raw = "MNE_Raw_SingleTrial_AllSubjects_AllLevels.npy"
with open(data_dir + data_raw, "rb") as f:
    dbraw = np.load(f, allow_pickle=True).item()
# config = configparser.ConfigParser()
# config.read(data_dir + "drop_epoch.ini")

level_values = [0, 1, 2]
diff = ["MATBeasy", "MATBmed", "MATBdiff"]
# freqbands = {
#     "delta": [2, 4],
#     "theta": [4, 8],
#     "alpha": [8, 12],
#     "beta": [15, 30],
#     "gamma": [30, 45],
#     "defaultBand": [8, 35],
# }
freqbands = {"gamma": [30, 45], "defaultBand": [8, 35]}
spectral_met = ["coh", "imcoh", "plv", "cov"]
n_subjects, n_sessions = 15, 2


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
        elif idx.min() >= lsess["S1"]:
            sess = "S2"
            lidx = idx - lsess["S1"]
        else:
            raise ValueError("Choose S1 or S2 indexes")
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
    sub = "P{0:02d}".format(subject)
    Xsess, ysess, lsess = {}, {}, {}
    for session in range(1, n_sessions + 1):
        sess = f"S{session}"
        Xsess[sess] = {
            f"{m}-{b}": [] for m, b in product(spectral_met, freqbands.keys())
        }
        Xsess[sess + "full"] = {
            f"{m}-{b}": [] for m, b in product(spectral_met, freqbands.keys())
        }
        Xsess[sess]["raw"], Xsess[sess + "full"]["raw"] = [], []
        ysess[sess], ysess[sess + "full"] = [], []
        for level, lv in zip(diff, level_values):
            # load data in a dict
            trials = {}
            trials["raw"] = dbraw[sub][sess][level].get_data()
            for band, (fmin, fmax) in freqbands.items():
                for sm in spectral_met:
                    sname = f"{sub}-{sess}-{level}-{sm}-1-0.5-{fmin}-{fmax}"
                    trials[f"{sm}-{band}"] = db[sname]

            # # Not used: load index of good trials as a list
            # good_idx = config[f"Subj{sub}_Sess_{sess}_level_{level}"]["listoftrials"]
            # good_idx = [int(i) for i in good_idx.split("[")[1].split("]")[0].split(",")]
            # print("Dropped {} epochs".format(trials["raw"].shape[0] - len(good_idx)))
            # use all trials:
            good_idx = range(trials["raw"].shape[0])

            # Use only good trials
            for sig in trials.keys():
                Xsess[sess][sig].append(trials[sig])
            ysess[sess] += [lv] * len(trials["raw"])
        lsess[sess] = len(ysess[sess])
    idx = np.arange(np.array([i for i in lsess.values()]).sum())

    # Train and evaluate
    for ppn, ppl in tqdm(ens_ppl.items(), total=len(ens_ppl), desc="pipelines"):
        X_idx_S1, X_idx_S2 = idx[: lsess["S1"]], idx[lsess["S1"] :]
        ppl1, ppl2 = deepcopy(ppl), deepcopy(ppl)

        ppl1.fit(X_idx_S1, ysess["S1"])
        y2, yp2 = ysess["S2"], ppl1.predict(X_idx_S2)
        sc_s2 = balanced_accuracy_score(y2, yp2)
        res = {"subject": sub, "test_session": "S2", "score": sc_s2, "pipeline": ppn}
        all_res.append(res)
        for est_n, est_p in ppl1.named_estimators_.items():
            ype2 = est_p.predict(X_idx_S2)
            sce_s2 = balanced_accuracy_score(y2, ype2)
            res = {
                "subject": sub,
                "test_session": "S2",
                "score": sce_s2,
                "pipeline": est_n,
            }
            all_res.append(res)

        ppl2.fit(X_idx_S2, ysess["S2"])
        y1, yp1 = ysess["S1"], ppl2.predict(X_idx_S1)
        sc_s1 = balanced_accuracy_score(y1, yp1)
        res = {"subject": sub, "test_session": "S1", "score": sc_s1, "pipeline": ppn}
        all_res.append(res)
        for est_n, est_p in ppl2.named_estimators_.items():
            ype1 = est_p.predict(X_idx_S1)
            sce_s1 = balanced_accuracy_score(y1, ype1)
            res = {
                "subject": sub,
                "test_session": "S1",
                "score": sce_s1,
                "pipeline": est_n,
            }
            all_res.append(res)

all_res = pd.DataFrame(all_res)
all_res.to_csv(data_dir + "classif_model_selection.csv")


#%% plot results
sns.catplot(
    data=all_res,
    x="subject",
    y="score",
    palette="plasma",
    hue="pipeline",
    kind="point",
    dodge=True,
)
plt.savefig("../Figures/Plot_Results_selection_classif_stub.png")
