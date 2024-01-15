# Example 2: Confound removal

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

`statannotations`:
```
pip install statannotations
```


## Downloading the data

Before running the example, you first need to download the data from LONI.
If you already have an account and access to ADNI, you can skip to step 6
 
1) Create an account in LONI: https://ida.loni.usc.edu
2) Once you login, select "ADNI" in feature studies.
3) Click on the "Apply for access" button.
4) Complete the application form and submit it.
5) Wait for the approval email. This may take a few days.

6) Login into LONI: https://ida.loni.usc.edu
7) Select ADNI
8) On the top bar, select Download -> Study data
9) Download each one of the following options:
   * Assessments -> Diagnosis -> Diagnosis Summary [ADNI1,GO,2,3]
   * Imaging -> MR Image Analysis -> UCSF - Cross-Sectional FreeSurfer (6.0) [ADNI3] 
   * Subject characteristics -> Subject demographics -> Subject demographics[ADNI1,GO,2,3]
10) Place the downloaded files in the `data/2_confounds` directory.
11) Rename the files so the last date component (download date) is removed. They should be name as follows:
    * `DXSUM_PDXCONV_ADNIALL.csv`
    * `UCSFFSX6_08_17_22.csv`
    * `PTDEMOG.csv`

## Running the example

1. Execute `1_process_data.py` to download the required data.

```
python 1_get_data.py
```

2. Run the main CV script:

```
python 2_run_models.py
```

3. Plot the reults

```
python 3_plot
```