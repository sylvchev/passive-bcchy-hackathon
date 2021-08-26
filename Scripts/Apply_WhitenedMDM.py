# Regression with a Whitened MDM, cross-session (and intra-subject)

# Remarques générales:
# - SotA on mental workload
# theta power [4, 8] in frontal sensors
# and alpha power [8, 12] in parietal sensors

# Inconvénients de Whitened MDM:
# - Whitening avec dim_red (non-supervisée): on ne projette pas sur le même
# espace source entre la session 1 et la session 2


###############################################################################

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.classification import MDM
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


###############################################################################

data_dir="../Database/"
data_file="MNE_Raw_SingleTrial_AllSubjects.npy"
level_names = ["MATBeasy", "MATBmed", "MATBdiff"]
level_values = [1, 2, 3]

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


def extract_covs(sub, sess, level_values, level_names):
    """ Extract covariance matrices
    after a band-pass filtering (bilateral IIR filtering)
    """
    X, y = np.empty((0, n_chan, n_chan)), np.empty((0))

    for level_value, level_name in zip(level_values, level_names):
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

    return X, y


###############################################################################

training_accuracies = np.empty((n_subjects))
testing_accuracies = np.empty((n_subjects))

for sub_n in range(n_subjects):

    print('\nSubject:', sub_n)
    sub = "P{0:02d}".format(sub_n + 1)


    ### training on session 1
    sess = "S1"
    X1, y1 = extract_covs(sub, sess, level_values, level_names)
        
    #whit1 = Whitening(dim_red={'max_cond': 100})
    whit1 = Whitening(dim_red={'expl_var': 0.999})
    X1_w = whit1.fit_transform(X1)
    print(' Dimension reduction on',  whit1.n_components_, 'components (over',
          n_chan, 'channels)')
    mdm = MDM(metric='riemann')
    mdm.fit_transform(X1_w, y1)
    y1_train = mdm.predict(X1_w)
    training_accuracies[sub_n] = accuracy_score(y1, y1_train)
    print(' Training accuracy score =', training_accuracies[sub_n])


    ### testing on session 2
    sess = "S2"
    X2, y2 = extract_covs(sub, sess, level_values, level_names)

    whit2 = Whitening(dim_red={'n_components': whit1.n_components_})
    X2_w = whit2.fit_transform(X2)
    y2_test = mdm.predict(X2_w)
    testing_accuracies[sub_n] = accuracy_score(y2, y2_test)
    print(' Testing accuracy score =', testing_accuracies[sub_n])


    ### plot results
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(title="Regression with a whitened MDM, subject " + str(sub_n+1),
           xlabel='Trial index', ylabel='Mental workload level')
    plt.scatter(np.arange(1, len(y1)+1), y1, c='b', label='GT training')
    plt.scatter(np.arange(1, len(y1)+1), y1_train, c='c', marker='+',
                label='Pred training', alpha=0.5)
    plt.scatter(np.arange(len(y1), len(y1)+len(y2)), y2, c='r', label='GT test')
    plt.scatter(np.arange(len(y1), len(y1)+len(y2)), y2_test, c='m', marker='+',
                label='Pred test', alpha=0.5)
    ax.legend()
    plt.show()



print('\nAll subjects:')
print(' Training accuracy score =', training_accuracies.mean(),
      '+/-', training_accuracies.std())
print(' Testing accuracy score =', testing_accuracies.mean(),
      '+/-', testing_accuracies.std())


###############################################################################

