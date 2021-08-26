# Regression with a Whitened SPoC, cross-session (and intra-subject)

# Remarques générales:
# - SotA on mental workload
# theta power [4, 8] in frontal sensors
# and alpha power [8, 12] in parietal sensors
# - fondamentalement, le challenge est un problème de régression;
# mais les labels du dataset ont été seuillé en 3 niveaux => bruit d'annotations!

# Inconvénients de Whitened SPoC:
# - Whitening avec dim_red (non-supervisée): on ne projette pas sur le même
# espace source entre la session 1 et la session 2

#TODO 
# - add marks to show classif results on plots


###############################################################################

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.spatialfilters import SPoC
from sklearn.metrics import mean_squared_error, accuracy_score
from matplotlib import pyplot as plt


###############################################################################

data_dir="../Database/"
data_file="MNE_Raw_SingleTrial_AllSubjects.npy"
level_names = ["MATBeasy", "MATBmed", "MATBdiff"]
level_values = [-1, 0, 1]

with open(data_dir + data_file, 'rb') as f:
    Database = np.load(f, allow_pickle=True).item()

n_subjects=len(Database)
n_sessions=len(Database["P01"])
n_chan=len(Database["P01"]["S1"][level_names[0]].ch_names)


##################################

# True, if you want to reduce on frontal theta and parietal alpha
if True:
    ch_frontal = [
        "F7", "F5", "F3", "F1", "F2", "F4", "F6", "AF3", "AFz", "AF4"
    ]
    ch_parietal = [
        "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"
    ]
    n_chan = len(ch_frontal) + len(ch_parietal)


###############################################################################


def standardize(level_values):
    """Standardize values,
    to imitate SPoC inner processing of target continuous values"""
    level_stdvalues = level_values / np.std(level_values)
    return level_stdvalues

def extract_covs(sub, sess, level_values, level_names):
    """ Extract covariance matrices
    after a band-pass filtering (bilateral IIR filtering)
    """
    level_stdvalues = standardize(level_values)

    X, y, y_std = np.empty((0, n_chan, n_chan)), np.empty((0)), np.empty((0))

    for level_value, level_stdvalue, level_name \
    in zip(level_values, level_stdvalues, level_names):
        epochs=Database[sub][sess][level_name]

        if ch_frontal and ch_parietal:
            epochs_1 = epochs.copy().pick_channels(ch_frontal).filter(l_freq=4, h_freq=8, method="iir").get_data()
            epochs_2 = epochs.copy().pick_channels(ch_parietal).filter(l_freq=8, h_freq=12, method="iir").get_data()
            epochs_ = np.hstack((epochs_1, epochs_2))
        else:
            epochs_ = epochs.get_data()

        covs = Covariances(estimator='lwf').transform(epochs_)
        X = np.concatenate((X, covs))
        y = np.concatenate((y, level_value * np.ones(covs.shape[0])))
        y_std = np.concatenate((y_std, level_value * np.ones(covs.shape[0])))

    return X, y, y_std

def reg_to_classif(y_reg, level_values):

    level_stdvalues = standardize(level_values)
    level_thresholds = (level_stdvalues[1:] + level_stdvalues[:-1]) / 2

    y_classif = np.zeros_like(y_reg)

    for i, y in enumerate(y_reg):
        if y < level_thresholds[0]:
            y_classif[i] = level_values[0]
        elif y > level_thresholds[1]:
            y_classif[i] = level_values[2]
        else:
            y_classif[i] = level_values[1]

    return y_classif


###############################################################################

training_accuracies = np.empty((n_subjects))
testing_accuracies = np.empty((n_subjects))

for sub_n in range(n_subjects):

    print('\nSubject:', sub_n)
    sub = "P{0:02d}".format(sub_n + 1)


    ### training on session 1
    sess = "S1"
    X1, y1, y1_std = extract_covs(sub, sess, level_values, level_names)
        
    #whit1 = Whitening(dim_red={'max_cond': 100})
    whit1 = Whitening(dim_red={'expl_var': 0.999})
    X1_w = whit1.fit_transform(X1)
    print(' Dimension reduction on',  whit1.n_components_, 'components (over',
          n_chan, 'channels)')
    spoc = SPoC(nfilter=1, metric='euclid', log=True)
    y1_train_reg = np.squeeze(spoc.fit_transform(X1_w, y1_std))
    print(' Training MSE =', mean_squared_error(y1_std, y1_train_reg))
    y1_train_classif = reg_to_classif(y1_train_reg, level_values)
    training_accuracies[sub_n] = accuracy_score(y1, y1_train_classif)
    print(' Training accuracy score =', training_accuracies[sub_n])


    ### testing on session 2
    sess = "S2"
    X2, y2, y2_std = extract_covs(sub, sess, level_values, level_names)

    whit2 = Whitening(dim_red={'n_components': whit1.n_components_})
    X2_w = whit2.fit_transform(X2)
    y2_test_reg = np.squeeze(spoc.transform(X2_w))
    print(' Testing MSE =', mean_squared_error(y2_std, y2_test_reg))
    y2_test_classif = reg_to_classif(y2_test_reg, level_values)
    testing_accuracies[sub_n] = accuracy_score(y2, y2_test_classif)
    print(' Testing accuracy score =', testing_accuracies[sub_n])


    ### plot results
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(title="Regression with a whitened SPoC, subject " + str(sub_n+1),
           xlabel='Trial index', ylabel='Mental workload level')
    plt.scatter(np.arange(1, len(y1)+1), y1, c='b', label='GT training')
    plt.scatter(np.arange(1, len(y1)+1), y1_train_reg, c='c',
                label='Pred training', alpha=0.5)
    plt.scatter(np.arange(len(y1), len(y1)+len(y2)), y2, c='r', label='GT test')
    plt.scatter(np.arange(len(y1), len(y1)+len(y2)), y2_test_reg, c='m',
                label='Pred test', alpha=0.5)
    ax.legend()
    plt.show()



print('\nAll subjects:')
print(' Training accuracy score =', training_accuracies.mean(),
      '+/-', training_accuracies.std())
print(' Testing accuracy score =', testing_accuracies.mean(),
      '+/-', testing_accuracies.std())


###############################################################################

