# Script to load the data competition and save them in .npy

import mne
import pandas as pd
import numpy as np

import os
from mne.epochs import BaseEpochs
import matplotlib.pyplot as plt
%matplotlib

#%% read electrodes infos
data_path = "../Database/"
n_subjects_tot = 15
n_sessions_tot = 2
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


sub_n = 15 # need to do it manually
# subj 5, 6, 9, 11, 14 super noisy btw
# subj 3, 10, 13 not noisy at all
session_n= 2 # need to do it manually
level= diff[2] # need to do it manually

temp_sess = dict ()
epochs_data = []
labels = []
temp_database = dict ()


sub = "P{0:02d}".format(sub_n)
sess = f"S{session_n}"
path = (
        os.path.join(os.path.join(data_path, sub), sess)
        + f"/eeg/alldata_sbj{str(sub_n).zfill(2)}_sess{session_n}_{level}.set"
        )
# reject to be adapted to each subject & session

# Read the epoched data with MNE, between 0.10 & 1.80 to avoid pre/post distorsions
epochs_old = mne.io.read_epochs_eeglab(path, verbose=False)
epochs = epochs_old.copy().crop(tmin=0.10, tmax=1.80, include_tmax=True)

title_before='Before reject, 0.10-1.80s, Subj'+sub+', Sess'+sess+ ', '+ level+ ' level'
epochs.plot_image (title=title_before) # to identify the threshold to apply
plt.savefig ( os.path.join(os.path.join(data_path, sub), sess)
        + f"/eeg/"+title_before+'.pdf' , dpi=300 )

#%%
data = epochs.get_data ()
info = epochs.info
events = epochs.events
event_id = epochs.event_id


reject = dict ( eeg=90e-6 ) # by default:90e-6
epochs_bccy = mne.EpochsArray ( data , info=info , events=events ,
                                        event_id=event_id,  reject=reject)
#epochs_bccy.plot()

percent_epochs_dropped=epochs_bccy.drop_log_stats()
title_after = 'After Reject, 0.10-1.80s, Subj' + sub + ' Sess' + sess + ', ' + level + ' level' + ', ' + "\n" + str(round(percent_epochs_dropped,1)) + '% dropped epochs' + ', Threshold of ' + str(reject["eeg"]) + ' V'

epochs_bccy.plot_image (title=title_after) # to check the results
plt.savefig ( os.path.join(os.path.join(data_path, sub), sess)
        + f"/eeg/"+title_after+'.pdf' , dpi=300 )

np.save(data_path + '/'"Trials_Selection_Subj"+sub+'_Sess_'+sess+'_level_'+level+".npy",epochs_bccy.selection)

#%% Considering concatenate all the selections afterwards to make it easier to manipulate

Trials_selection=dict()
for file in os.listdir(data_path + '/'):
    if file.startswith("Trials_Selection_Subj"):
        Trials_selection[file]=np.load(data_path + '/'+file)

np.save(data_path + '/'"Trials_Selection_AllSubj_Sess1Sess2_AllLevels_0.21-1.75s.npy",Trials_selection)

#%% create a config file with the list of trials to use & the threshold applied
import configparser
config = configparser.ConfigParser()

n_subjects = 15
n_sessions = 2
diff = ["MATBeasy", "MATBmed", "MATBdiff"]

#Trials_selection_test=dict()

for sub_n in range(n_subjects):
    for session_n in range(n_sessions):
        sub = "P{0:02d}".format ( sub_n + 1 )
        sess = f"S{session_n + 1}"
        for lab_idx , level in enumerate ( diff ):
            data=dict()
            path = (
                os.path.join(os.path.join(data_path, sub), sess)
                + f"/eeg/"
            )
            for file in os.listdir(path):
                if file.startswith("After Reject"):
                    tmp=file.split(" ")
                    data["Threshold"]=tmp[len(tmp)-2]

            data["ListOfTrials"] = np.load(data_path + '/'"Trials_Selection_Subj" + sub + '_Sess_' + sess + '_level_' + level + ".npy").tolist()

            #Trials_selection_test["Subj" + sub + '_Sess_' + sess + '_level_' + level ]=data
            config["Subj" + sub + '_Sess_' + sess + '_level_' + level] = data



with open(data_path + '/drop_epoch.ini', 'w') as configfile:
    config.write(configfile)
