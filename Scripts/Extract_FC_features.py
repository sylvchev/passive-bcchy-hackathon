# Script to load the data competition in .npy, compute the FC metrics, & save results in .npz file

import os.path as osp
import numpy as np
from tqdm import tqdm, trange
from sklearn.covariance import ledoit_wolf

from mne import set_log_level, EpochsArray
from mne.epochs import BaseEpochs
from mne.connectivity import spectral_connectivity
from mne.connectivity import envelope_correlation

#%%
def _compute_fc_subtrial(epoch, delta=1, ratio=0.5, method="coh", fmin=8, fmax=35):
    """Compute single trial functional connectivity (FC)

    Most of the FC estimators are already implemented in mne-python (and used here from
    mne.connectivity.spectral_connectivity and mne.connectivity.envelope_correlation).
    The epoch is split into subtrials.

    Parameters
    ----------
    epoch: MNE epoch
        Epoch to process
    delta: float
        length of the subtrial in seconds
    ratio: float, in [0, 1]
        ratio overlap of the sliding windows
    method: string
        FC method to be applied, currently implemented methods are: "coh", "plv",
        "imcoh", "pli", "pli2_unbiased", "wpli", "wpli2_debiased", "cov", "plm", "aec"
    fmin: real
        filtering frequency, lowpass, in Hz
    fmax: real
        filtering frequency, highpass, in Hz

    Returns
    -------
    connectivity: array, (nb channels x nb channels)


    TODO: see if the lagged coherence could be interesting :
        - M. Congedo's paper: https://hal.archives-ouvertes.fr/hal-00423717/file/Congedo_et_al_2009_Brain_Topography.pdf
        - Integration within MNE: https://neurodsp-tools.github.io/neurodsp/auto_examples/plot_mne_example.html


    TODO: compare matlab/python plm's output
    The only exception is the Phase Linearity Measurement (PLM). In this case, it is a
    Python version of the ft_connectivity_plm MATLAB code [1] of the Fieldtrip
    toolbox [2], which credits [3], with the "translation" into Python made by
    M.-C. Corsi.

    references
    ----------
    .. [1] https://github.com/fieldtrip/fieldtrip/blob/master/connectivity/ft_connectivity_plm.m  # noqa
    .. [2] R. Oostenveld, P. Fries, E. Maris, J.-M. Schoffelen, and  R. Oostenveld,
    "FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive
    Electrophysiological  Data" (2010): https://doi.org/10.1155/2011/156869
    .. [3] F. Baselice, A. Sorriso, R. Rucco, and P. Sorrentino, "Phase Linearity
    Measurement: A Novel Index for Brain Functional Connectivity" (2019):
    https://doi.org/10.1109/TMI.2018.2873423
    """
    lvl = set_log_level("CRITICAL")
    L = epoch.times[-1] - epoch.times[0]
    sliding = ratio * delta
    # fmt: off
    spectral_met = ["coh", "plv", "imcoh", "pli", "pli2_unbiased",
                    "wpli", "wpli2_debiased", ]
    other_met = ["cov", "plm", "aec"]
    # fmt: on
    if not method in spectral_met + other_met:
        raise NotImplemented("this spectral connectivity method is not implemented")

    sfreq, nb_chan = epoch.info["sfreq"], epoch.info["nchan"]
    win = delta * sfreq
    nb_subtrials = int(L * (1 / (sliding + delta) + 1 / delta))
    nbsamples_subtrial = delta * sfreq

    # TODO:
    #  - reboot computation options depending on the frequency options, faveage=False, but issue on AEC /PLM :/
    #  - robust estimators : bootstrap over subtrials, sub-subtrials & z-score, ways to remove outliers

    # X, total nb trials over the session(s) x nb channels x nb samples
    X = np.squeeze(epoch.get_data())
    subtrials = np.empty((nb_subtrials, nb_chan, int(win)))

    for i in range(0, nb_subtrials):
        idx_start = int(sfreq * i * sliding)
        idx_stop = int(sfreq * i * sliding + nbsamples_subtrial)
        subtrials[i, :, :] = np.expand_dims(X[:, idx_start:idx_stop], axis=0)
    sub_epoch = EpochsArray(np.squeeze(subtrials), info=epoch.info)
    if method in spectral_met:
        r = spectral_connectivity(
            sub_epoch,
            method=method,
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            tmin=0,
            mt_adaptive=False,
            n_jobs=1,
        )
        c = np.squeeze(r[0])
        c = c + c.T - np.diag(np.diag(c)) + np.identity(nb_chan)
    elif method == "aec":
        # filter in frequency band of interest
        sub_epoch.filter(
            fmin,
            fmax,
            n_jobs=1,
            l_trans_bandwidth=1,  # make sure filter params are the same
            h_trans_bandwidth=1,
        )  # in each band and skip "auto" option.
        # apply hilbert transform first
        h_sub_epoch = sub_epoch.apply_hilbert()
        c = envelope_correlation(h_sub_epoch, verbose=True)
        # by default, combine correlation estimates across epochs by peforming an average
        # output : nb_channels x nb_channels -> no need to rearrange the matrix
    elif method == "cov":
        c = ledoit_wolf(X.T)[0]  # oas ou fast_mcd
    elif method == "plm":
        # adapted from the matlab code from https://github.com/fieldtrip/fieldtrip/blob/master/connectivity/ft_connectivity_plm.m  # noqa
        # TODO: compare matlab/python plm's output
        # no need to filter before because integration in the frequency band later on

        # apply hilbert transform first
        h_sub_epoch = sub_epoch.apply_hilbert()
        input = h_sub_epoch.get_data()

        ph_min = 0.1  # Eps of Eq.(17) of the manuscript
        f = [
            int(sfreq / nbsamples_subtrial) * x
            for x in range(int(nbsamples_subtrial) - 1)
        ]
        # B: bandwidth, in hertz
        B = fmax - fmin
        f_diff = np.zeros((len(f), 1))
        for i in range(len(f)):
            f_diff[i] = f[i] - sfreq
        idx_f_integr_temp = np.where(
            np.logical_and(np.abs(f) < B, np.abs(f_diff) < B) == True
        )
        idx_f_integr = idx_f_integr_temp[1]

        p = np.zeros((nb_chan, nb_chan, len(input)))
        for i in range(len(input)):
            for kchan1 in range(nb_chan - 2):
                for kchan2 in range((kchan1 + 1), nb_chan):
                    temp = np.fft.fft(
                        input[i, kchan1, :] * np.conjugate(input[i, kchan2, :])
                    )
                    temp[0] = temp[0] * (abs(np.angle(temp[0])) > ph_min)
                    # TODO: check temp values, they are really low
                    temp = np.power((abs(temp)), 2)
                    p_temp = np.sum(temp[idx_f_integr,]) / np.sum(temp)
                    p[kchan1, kchan2, i] = p_temp
                    p[kchan2, kchan1, i] = p_temp
                    # to make it symmetrical i guess
        # new, not in the matlab code, average over the
        # subtrials + normalization:
        m = np.mean(p, axis=2)
        c1 = m / np.max(m) + np.identity(nb_chan)
        c = np.moveaxis(c1, -1, 0)
    return c


#%% parameters
spectral_met = [
    "coh",
    "plv",
    "cov",
    "imcoh",
    "wpli",
]
# "pli",
# "pli2_unbiased",
# "wpli2_debiased",
FreqBands = dict()
FreqBands = {
    "delta": [2, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [15, 30],
    "gamma": [30, 45],
    "defaultBand": [8, 35],
}
#%% retrieve data
data_dir = "../Database/"
data_file = "MNE_Raw_SingleTrial_AllSubjects_AllLevels.npy"
diff = ["MATBeasy", "MATBmed", "MATBdiff"]

with open(data_dir + data_file, "rb") as f:
    Database = np.load(f, allow_pickle=True).item()

n_subjects = len(Database)
n_sessions = len(Database["P01"])
n_chan = len(Database["P01"]["S1"][diff[0]].ch_names)
# tmin, tmax = None, None
tmin, tmax = 0.1, 1.8
set_log_level("ERROR")

FC_Database = dict()
for sub_n in trange(n_subjects, desc="subjects"):
    temp_sess = dict()
    sub = "P{0:02d}".format(sub_n + 1)
    for session_n in trange(n_sessions, desc="sessions"):
        sess = f"S{session_n+1}"
        for lab_idx, level in enumerate(diff):
            epochs = Database[sub][sess][level].crop(tmin, tmax)
            for fmin, fmax in FreqBands.values():
                for sm in spectral_met:
                    delta, ratio = 1, 0.5
                    lp = [sub, sess, level, sm, delta, ratio, fmin, fmax]
                    cname = "-".join([str(e) for e in lp])
                    X = Database[sub][sess][level]
                    Xfc = np.empty((len(X), n_chan, n_chan))
                    for i in range(len(X)):
                        Xfc[i, :, :] = _compute_fc_subtrial(
                            X[i], delta, ratio, sm, fmin, fmax
                        )
                    FC_Database[cname] = Xfc

#%% save concatenated features, compressed format, in case...
data_dir = "../Database/"
data_file = "MNE_FC_crop_AllEstim_6Freqs.npz"
np.savez_compressed(data_dir + data_file, FC_Database)

#%% plot an adjency matrix, just a test
# plt.imshow(Xfc[1, :, :])
# plt.savefig('FC-test.png')
