# passive-bccy-hackathon

Submission for Passive BCI Hackathon of Neuroergonomics conference

[Competition website](https://www.neuroergonomicsconference.um.ifi.lmu.de/pbci/)

## Data

Data & instructions available [here](https://zenodo.org/record/4917218#.YNGIVi3pODW)

## Installation

```shell 
pip install -r requirements.txt
```

## Code

To obtain the prediction for the competition, you should run
1. `Scripts/Extract_FC_features.py`
1. `Scripts/FormatData.py`
1. `Scripts/predict_session3.py`


```shell 
root/
-Scripts/
--ExploreData.py
--FormatData.py
--...
-Database/
--chan_locs_standard.dms
--...
--test/
---test_sample.foo
---...
--train/
---training_sample.bar
---...
```

