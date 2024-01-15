# Example 3: CBPM



## Requirements

To run this example, please make sure you have the following libraries installed

`julearn`, with the visualization dependencies:

```
pip install julearn[viz]
```

`seaborn`:
```
pip install seaborn
```


## Obtaining the data

This example requires accces to the Human Connectome Project WU-Minn HCP 1200 databse.

Unfortunately, due to restrictions on the database, we cannot provide the features used for the example. The user must download and compute the required features.

The example requires 3 files, placed in the `data/3_gf` directory

1. `dataset-HCPREST1_parc-Finn2015Shen268_marker-empiricalFC.csv`

This file contains the functional connectivity as a pandas dataframe. The first column `subject` contains the subject identifier. The rest of the columns should be named `X~Y`, where `X` and `Y` represents an ROI in the Shen parcellation, as described in the julearn publication.

2. `unrestricted.csv`

This file contains subject-specific information from the HCP dataset. It must contain two columns: `subject` and `PMAT24_A_CR`.

3. `unrelated_subjects_rms_filtered.txt`

This file defines the list of subjects that will be used in the analysis. Since the HCP dataset contains family members, related subjects must be excluded. Each line must contain one subject id.

## Running the example

1. Execute `2_run_all_parameters.sh` to run all the analyses used in the manuscript.

```
./2_run_all_parameters.sh
```

2. Plot the reults

```
python 3_plot
```