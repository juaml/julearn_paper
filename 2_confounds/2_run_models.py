# %%
from julearn import run_cross_validation, PipelineCreator
from julearn.utils import configure_logging, logger, raise_error
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import numpy as np

from pathlib import Path


configure_logging(level="INFO")
data_dir = Path(__file__).parent.parent / "data" / "2_confounds"
all_data_df = pd.read_csv(data_dir / "ADNI_subonce_qc.csv")

out_dir = Path(__file__).parent.parent / "results" / "2_confounds"
out_dir.mkdir(exist_ok=True, parents=True)

# %%
# Balance the data as we have more healthy than unhealthy
healthy = all_data_df["current_diagnosis"] == "NL"
unhealthy = all_data_df["current_diagnosis"] != "NL"
# Get the count for each group
n_healthy = healthy.sum()
n_unhealthy = unhealthy.sum()
n_min = min(n_healthy, n_unhealthy)

# Sample the data to get the same number of healthy and unhealthy
df_healthy = all_data_df[healthy].sample(n_min, random_state=824)
df_unhealthy = all_data_df[unhealthy].sample(n_min, random_state=2947)

# Create a new dataframe with the balanced data
data_df = pd.concat([df_healthy, df_unhealthy]).reset_index(drop=True)
# Compute age
exam_year = data_df["EXAMDATE_fs"].str[:4].astype(int)
data_df["age"] = exam_year - data_df["PTDOBYY"].astype(int)

# %% Set up the model without confound removal
svm_Cs = np.arange(0.1, 4, 0.2)

pipeline_raw = PipelineCreator(problem_type="classification")
pipeline_raw.add("zscore")
pipeline_raw.add("svm", kernel="rbf", C=svm_Cs)
cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=60, random_state=890234)

# %% Run the model
scores_raw, _, inspect_raw = run_cross_validation(
    X=[".*TA$"],
    y="current_diagnosis",
    data=data_df,
    X_types={"continuous": [".*TA$"]},
    return_estimator="all",
    model=pipeline_raw,
    cv=cv,
    return_inspector=True,
    pos_labels=["MCI", "AD"],
)
# %% Set up the model with confound removal on healthy subjects
pipeline_corrected = PipelineCreator(problem_type="classification")
pipeline_corrected.add("zscore")
pipeline_corrected.add(
    "confound_removal",
    confounds="confound",
    row_select_col_type="grouper",
    row_select_vals=["NL"],
)
pipeline_corrected.add("svm", kernel="rbf", C=svm_Cs)

scores_corrected, _, inspect_corrected = run_cross_validation(
    X=[".*TA$", "current_diagnosis", "age"],
    y="current_diagnosis",
    data=data_df,
    X_types={
        "confound": ["age"],
        "grouper": ["current_diagnosis"],
        "continuous": [".*TA$"],
    },
    return_estimator="all",
    model=pipeline_corrected,
    cv=cv,
    return_inspector=True,
    pos_labels=["MCI", "AD"],
)

# %%
preds_raw = inspect_raw.folds.predict()
preds_corrected = inspect_corrected.folds.predict()

# %%
scores_raw.to_csv(out_dir / "scores_raw.csv")
preds_raw.to_csv(out_dir / "preds_raw.csv")
scores_corrected.to_csv(out_dir/ "scores_corrected.csv")
preds_corrected.to_csv(out_dir / "preds_corrected.csv")
pruned_data_df = data_df[["age", "current_diagnosis"]]
pruned_data_df.to_csv(out_dir / "used_data.csv")
# %%
