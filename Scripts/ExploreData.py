# i just redo the main steps proposed in the competition doc
# More info available here: https://zenodo.org/record/4917218#.YNGIVi3pODW

import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

plt.ion()

#%% read electrodes infos
data_path = "Database/"
n_subjects = 9
n_sessions = 2
diff = ["MATBeasy", "MATBmed", "MATBdiff"]

electrodes = pd.read_csv(
    data_path + "chan_locs_standard.dms",
    header=None,
    sep="\t",
    names=["ch_names", "x", "y", "z"],
)
electrodes.head()


#%% look at data from a given subject - subset of channels (frontal ones)
# fmt: off
ch_slice = ["F7", "F5", "F3", "F1", "F2", "F4", "F6", "AF3", "AFz", "AF4",
            "Fp1", "Fp2", ]  # , 'FPz'] -> FPz not present !

ch_select_mcc = ["Fp1", "Fz", "F3", "F7", "FC5", "FC1", "C3", "CP5", "CP1",
                 "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "CP6", "CP2",
                 "FCz", "C4", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3",
                 "AFz", "F1", "F5", "FC3", "C1", "C5", "CP3", "P1", "P5",
                 "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2", "CPz", "CP4",
                 "C6", "C2", "FC4", "F6", "AF8", "AF4", "F2", ]
# fmt: on

event_ids = {}
all_epochs = []
for sub_n, session_n in itertools.product(range(n_subjects), range(n_sessions)):
    epochs_data = []
    labels = []
    for lab_idx, level in enumerate(diff):
        sub = "P{0:02d}".format(sub_n + 1)
        sess = f"S{session_n+1}"
        path = (
            os.path.join(data_path, sub, sess, "eeg", 
            f"alldata_sbj{str(sub_n+1).zfill(2)}_sess{session_n+1}_{level}.set")
        )
        # Read the epoched data with MNE
        epochs = mne.io.read_epochs_eeglab(path, verbose=False)

        # Add the montage from read electrodes
        elect = pd.read_csv(os.path.join(data_path, sub, sess, "electrode_positions", "get_chanlocs.txt"), header=None, sep="\s+", names=["ch_names", "x", "y", "z"])
        ch_pos = {elect.loc[i]['ch_names']: np.array([elect.loc[i]['x'], elect.loc[i]['y'], elect.loc[i]['z']]) for i in range(len(elect))}
        montage = mne.channels.make_dig_montage(ch_pos, nasion=ch_pos['nas'])
        if 'FT9' in epochs.ch_names or 'CP5' in epochs.ch_names or 'FT10' in epochs.ch_names:
            epochs = epochs.drop_channels(['FT9', 'CP5', 'FT10'])  # because they are not in the epochs!
        epochs.set_montage(montage)

        # Use standard montage
        # montage = mne.channels.make_standard_montage("standard_1005")
        # epochs.set_montage(montage)

        # Use event correlated with subj, sess and level
        ev_code = (sub_n+1)*100 + (session_n+1)*10 + lab_idx
        epochs.events[:, -1] = ev_code
        event_ids['{}/{}/{}'.format(sub, sess, level)] = ev_code
        epochs.event_id = {'{}/{}/{}'.format(sub, sess, level): ev_code}
        all_epochs.append(epochs)

        # Time-frequency analysis
        # Voir https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html#sphx-glr-auto-examples-time-frequency-time-frequency-erds-py
        # Utiliser une référence, comme le resting state
        # Estimating evoked responses

        # plots
        picks = mne.pick_channels(epochs.info["ch_names"], ch_select_mcc)
        epoch_plot_psd=epochs.plot_psd(fmin=0, fmax=45)
        epoch_plot_psd.savefig('Figures/PSD_{}_{}_{}.png'.format(sub, sess, level), dpi=300)
        epoch_plot_psd_topo = epochs.plot_psd_topomap()
        epoch_plot_psd_topo.savefig('Figures/PSD_topo_{}_{}_{}.png'.format(sub, sess, level) , dpi=300)
        plt.close("all")

        # You could add some pre-processing here with MNE
        # We will just select some channels (mostly frontal ones)
        # epochs = epochs.drop_channels(list(set(ch_slice)))

        # Get the data and concatenante with others MATB levels
        # tmp = epochs.get_data()
        # epochs_data.extend(tmp)
        # labels.extend([lab_idx] * len(tmp))

    # epoch_plot=epochs.plot() # to visually check that SOBI works
    # epoch_plot.savefig ( 'Figures/EEG-Signals_'+ sub + '_' + sess + '.pdf' , dpi=300 )

    # epochs_data = np.array(epochs_data)
    # labels = np.array(labels)
all_epochs = mne.concatenate_epochs(all_epochs)
