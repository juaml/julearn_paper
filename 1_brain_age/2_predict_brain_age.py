# %%
from pathlib import Path
import pandas as pd

# Needs to install skrvm
# pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
from skrvm import RVR
from sklearn.gaussian_process.kernels import RBF

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.model_selection import RepeatedStratifiedGroupsKFold
from julearn.utils import configure_logging

import seaborn as sns

# %%
configure_logging(level="INFO")

data_dir = Path(__file__).parent.parent / "data"

input_file = data_dir / "ixi.S4_R8.csv"

data_df = pd.read_csv(input_file)

print(data_df.head())

# %%
# Set up the parameters for the cross-validation
rand_seed = 200
n_repeats = 1
n_splits = 3
scoring = [
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "r2",
]

# %%
# Create age bins for a stratified k-fold cross-validation

n_bins = len(data_df) // n_splits

bins_on = data_df["age"]  # variable to be used for stratification
qc = pd.cut(bins_on.tolist(), n_bins)  # divides data in bins
data_df["bins"] = qc.codes
print(data_df.head())

# %% variables to keep both models scores
all_scores = []
# %%
# Model 1: RVR
creator = PipelineCreator(problem_type="regression")
creator.add("select_variance", threshold=1e-5)
creator.add("pca")
creator.add(
    RVR(), kernel=["linear", "poly"], degree=[1, 2], random_state=rand_seed
)

cv = RepeatedStratifiedGroupsKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed
)

scores = run_cross_validation(
    X=["f_.*"],
    y="age",
    groups="bins",
    data=data_df,
    model=creator,
    cv=cv,
    seed=rand_seed,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2",
    ],
)
scores["model"] = "rvr"
all_scores.append(scores)

# %%
# Model 2: GPR
creator = PipelineCreator(problem_type="regression")
creator.add("select_variance", threshold=1e-5)
creator.add("pca")
creator.add(
    "gauss",
    kernel=RBF(10.0, (1e-7, 10e7)),
    normalize_y=True,
    n_restarts_optimizer=100,
    random_state=rand_seed,
)

cv = RepeatedStratifiedGroupsKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed
)

scores = run_cross_validation(
    X=["f_.*"],
    y="age",
    groups="bins",
    data=data_df,
    model=creator,
    cv=cv,
    seed=rand_seed,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2",
    ],
)
scores["model"] = "gauss"
all_scores.append(scores)
# %%
# Create on single dataframe with all scores
all_scores = pd.concat(all_scores)

# %%
# Plot
sns.set_style("darkgrid")

wide_df = all_scores.melt(
    id_vars=["model", "fold", "repeat"], var_name="metric"
)

sns.catplot(
    x="model", y="value", col="metric", data=wide_df, kind="box", sharey=False
)
