# Script to load the data competition and save them in .npy

import mne
import pandas as pd
import numpy as np

import os
from mne.epochs import BaseEpochs
#%% read electrodes infos
data_path = "./Database/"
n_subjects = 15
n_sessions = 2
diff = ["MATBeasy", "MATBmed", "MATBdiff"]

electrodes = pd.read_csv(
    data_path + "chan_locs_standard.dms",
    header=None,
    sep="\t",
    names=["ch_names", "x", "y", "z"],
)
electrodes.head()

#%% For the moment, we take into account all the channels
Database=dict()
temp_sub = dict ()
for sub_n in range(n_subjects):
    temp_sess = dict ()
    for session_n in range(n_sessions):
        epochs_data = []
        labels = []
        temp_database = dict ()

        for lab_idx, level in enumerate(diff):
            sub = "P{0:02d}".format(sub_n + 1)
            sess = f"S{session_n+1}"
            path = (
                os.path.join(os.path.join(data_path, sub), sess)
                + f"/eeg/alldata_sbj{str(sub_n+1).zfill(2)}_sess{session_n+1}_{level}.set"
            )
            # Read the epoched data with MNE
            epochs = mne.io.read_epochs_eeglab(path, verbose=False)

            if isinstance(epochs,BaseEpochs):
                temp_database[ level ] = epochs
                ## Get the data and concatenate with others MATB levels --> in comments here
                # tmp = epochs.get_data()
                # epochs_data.extend(tmp)
                # labels.extend([lab_idx] * len(tmp))
            else:
                print('WARNING: epochs not compatible with MNE')
        # epochs_data = np.array(epochs_data)
        # labels = np.array(labels)

        #temp_database["data"] = temp_diff #epochs_data
        #temp_database["labels"] = labels
        temp_sess[sess]=temp_database
    temp_sub[sub]=temp_sess

Database=temp_sub

# save concatenated "raw" data, in case...
data_dir="./Database/"
data_file="MNE_Raw_SingleTrial_AllSubjects_AllLevels.npy"

#%%
with open(data_dir+data_file, 'wb') as f:
    np.save(f, Database, allow_pickle=True)
