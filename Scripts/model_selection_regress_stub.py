# Stub for all steps from loading data to classification

import configparser
from itertools import product
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP as MNE_CSP
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.spatialfilters import SPoC
from fc_helpers import EnsureSPD


mne.set_log_level("ERROR")

# Prelooading files
data_dir = "../Database/"
data_fc = "MNE_FC_SingleTrial_AllSubjects_AllLevels.npz"
database = np.load(data_dir + data_fc, allow_pickle=True)
db = database["arr_0"].item()
data_raw = "MNE_Raw_SingleTrial_AllSubjects_AllLevels.npy"
with open(data_dir + data_raw, "rb") as f:
    dbraw = np.load(f, allow_pickle=True).item()
config = configparser.ConfigParser()
config.read(data_dir + "drop_epoch.ini")

level_values = [-1, 0, 1]
diff = ["MATBeasy", "MATBmed", "MATBdiff"]
# freqbands = {
#     "delta": [2, 4],
#     "theta": [4, 8],
#     "alpha": [8, 12],
#     "beta": [15, 30],
#     "gamma": [30, 45],
#     "defaultBand": [8, 35],
# }
freqbands = {"defaultBand": [8, 35]}
spectral_met = ["coh", "plv", "cov"]
n_subjects, n_sessions = 2, 2
n_jobs = -1

# db['P01-S1-MATBeasy-coh-1-0.5-8-35']
# name : sub, sess, level, method, delta, ratio, fmin, fmax

# pour tous les sujets
# liste d'epochs à supprimer, drop_list
# load les epochs
# train/test un classifieur
# sauver les résultats


class InputFeat(TransformerMixin, BaseEstimator):
    """Getting connectivity features from dict input"""

    def __init__(self, feat="raw"):
        self.feat = feat

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return np.concatenate(X[self.feat], axis=0)

    def fit_transform(self, X, y=None):
        return self.transform(X)


def standardize(level_values):
    """Standardize values,
    to imitate SPoC inner processing of target continuous values"""
    level_stdvalues = level_values / np.std(level_values)
    return level_stdvalues


def reg_to_classif(y_reg, level_values):
    """Transform regression values to classes"""
    level_stdvalues = standardize(level_values)
    level_thresholds = (level_stdvalues[1:] + level_stdvalues[:-1]) / 2

    y_classif = np.empty((len(y_reg)))
    for i, y in enumerate(y_reg):
        if y < level_thresholds[0]:
            y_classif[i] = level_values[0]
        elif y > level_thresholds[1]:
            y_classif[i] = level_values[2]
        else:
            y_classif[i] = level_values[1]

    return y_classif


pipelines = {}
pipelines["CSP-LR-raw"] = make_pipeline(
    InputFeat("raw"), MNE_CSP(n_components=8), LinearRegression()
)
pipelines["TS-LR-coh"] = make_pipeline(
    InputFeat("coh-defaultBand"),
    EnsureSPD(),
    TangentSpace(metric="riemann", tsupdate=True),
    LinearRegression(),
)
pipelines["TS-LR-cov"] = make_pipeline(
    InputFeat("cov-defaultBand"),
    EnsureSPD(),
    TangentSpace(metric="riemann", tsupdate=True),
    LinearRegression(),
)
pipelines["SPoC-LR"] = make_pipeline(
    InputFeat("cov-defaultBand"),
    EnsureSPD(),
    SPoC(nfilter=6, metric="euclid", log=True),
    LinearRegression(),
)

all_res = []
for subject in range(1, n_subjects + 1):
    sub = "P{0:02d}".format(subject)
    print(f"Subject {sub}")
    Xsess, ysess = {}, {}
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

            # load index of good trials as a list
            good_idx = config[f"Subj{sub}_Sess_{sess}_level_{level}"]["listoftrials"]
            good_idx = [int(i) for i in good_idx.split("[")[1].split("]")[0].split(",")]
            # uncomment to use all trials:
            # good_idx = range(trials["raw"].shape[0])
            print("Dropped {} epochs".format(trials["raw"].shape[0] - len(good_idx)))

            # Use only good trials
            for sig in trials.keys():
                Xsess[sess + "full"][sig].append(trials[sig])
                Xsess[sess][sig].append(trials[sig][good_idx, :, :])
            ysess[sess] += [lv] * len(good_idx)
            ysess[sess + "full"] += [lv] * len(trials["raw"])

    # Train and evaluate
    for ppn, ppl in pipelines.items():
        ppl.fit(Xsess["S1"], ysess["S1"])
        mse1 = mean_squared_error(ysess["S1"], ppl.predict(Xsess["S1"]))
        y2, yr2 = ysess["S2full"], ppl.predict(Xsess["S2full"])
        yp2 = reg_to_classif(yr2, level_values)
        sc_s2 = balanced_accuracy_score(y2, yp2)
        res = {
            "subject": sub,
            "test_session": "S2",
            "trainingMSE": mse1,
            "score": sc_s2,
            "pipeline": ppn,
        }
        all_res.append(res)

        ppl.fit(Xsess["S2"], ysess["S2"])
        mse2 = mean_squared_error(ysess["S2"], ppl.predict(Xsess["S2"]))
        y1, yr1 = ysess["S1full"], ppl.predict(Xsess["S1full"])
        yp1 = reg_to_classif(yr1, level_values)
        sc_s1 = balanced_accuracy_score(y1, yp1)
        res = {
            "subject": sub,
            "test_session": "S1",
            "trainingMSE": mse2,
            "score": sc_s1,
            "pipeline": ppn,
        }
        all_res.append(res)

all_res = pd.DataFrame(all_res)
all_res.to_csv(data_dir + "regress_model_selection.csv")

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
plt.savefig("../Figures/Plot_Results_selection_regression_stub.png")
